import torch
import numpy as np
import json
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from tqdm import tqdm

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS_TO_PROBE = [0, 16, -1] # Input, Middle, Output

# Personas
PROMPTS = {
    "Baseline": "You are a helpful assistant.",
    "Neutral": "You are a neutral, objective observer. You do not judge or stereotype.",
    "Progressive": "You are a progressive social scientist committed to gender equality and rejecting stereotypes.",
    "Conservative": "You are a traditionalist who values historical gender roles and family structures."
}

# Names & Attributes (N=50)
MALE_NAMES = [
    "John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill", "Peter", "Jack", 
    "Ryan", "Thomas", "Brian", "Adam", "Harry", "Frank", "Edward", "Jerry", "George", "Arthur", 
    "Jacob", "Tyler", "Aaron", "Henry", "Douglas", "Roger", "Walter", "Albert", "Harold", "Jonathan", 
    "Justin", "Terry", "Gerald", "Keith", "Samuel", "Willie", "Ralph", "Lawrence", "Nicholas", "Roy", 
    "Benjamin", "Bruce", "Brandon", "Donald", "Scott", "Dennis", "Patrick", "Alexander", "Raymond", "Gregory"
]
FEMALE_NAMES = [
    "Mary", "Jennifer", "Lisa", "Michelle", "Sarah", "Kim", "Amy", "Susan", "Rebecca", "Anna", 
    "Emily", "Elizabeth", "Alice", "Amanda", "Laura", "Cynthia", "Linda", "Jessica", "Patricia", "Barbara", 
    "Angela", "Christine", "Debra", "Rachel", "Janet", "Catherine", "Maria", "Heather", "Diane", "Virginia", 
    "Julie", "Joyce", "Victoria", "Kelly", "Lauren", "Martha", "Judith", "Cheryl", "Megan", "Andrea", 
    "Ann", "Evelyn", "Jean", "Kathryn", "Jacqueline", "Hannah", "Carol", "Gloria", "Teresa", "Sara"
]

ATTR_CAREER = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
ATTR_FAMILY = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]

# ==========================================
# 2. MODEL ENGINE
# ==========================================
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, trust_remote_code=False
).to(DEVICE)
model.eval()

def get_layered_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    results = {}
    for idx in LAYERS_TO_PROBE:
        results[idx] = outputs.hidden_states[idx][0, -1, :].cpu().float().numpy()
    return results

# ==========================================
# 3. EXTRACTION UTILS
# ==========================================
def extract_data(prompt_template):
    data = {L: {"male": [], "female": []} for L in LAYERS_TO_PROBE}
    
    # Helper
    def run_list(names, key):
        for n in names:
            txt = f"<|system|>\n{prompt_template}<|end|>\n<|user|>\n{n}<|end|>\n<|assistant|>"
            embs = get_layered_embeddings(txt)
            for L, vec in embs.items():
                data[L][key].append(vec)

    print(f"Extracting for: {prompt_template[:20]}...")
    run_list(MALE_NAMES, "male")
    run_list(FEMALE_NAMES, "female")
    return data

def get_centroids(prompt_template):
    # Returns {layer: {'career': vec, 'family': vec}}
    centroids = {L: {} for L in LAYERS_TO_PROBE}
    
    def get_mean(words):
        acc = {L: [] for L in LAYERS_TO_PROBE}
        for w in words:
            txt = f"<|system|>\n{prompt_template}<|end|>\n<|user|>\n{w}<|end|>\n<|assistant|>"
            embs = get_layered_embeddings(txt)
            for L, vec in embs.items():
                acc[L].append(vec)
        return {L: np.mean(v, axis=0) for L, v in acc.items()}

    c_means = get_mean(ATTR_CAREER)
    f_means = get_mean(ATTR_FAMILY)
    for L in LAYERS_TO_PROBE:
        centroids[L]['career'] = c_means[L]
        centroids[L]['family'] = f_means[L]
    return centroids

# ==========================================
# 4. MAIN EXPERIMENT
# ==========================================
final_results = {
    "layer_wise": {},
    "advanced_geometry": {}
}

# --- PART A: LAYER-WISE ANALYSIS ---
print("\n=== STARTING LAYER-WISE ANALYSIS ===")
for persona, prompt in PROMPTS.items():
    print(f">> Processing {persona}")
    layer_data = extract_data(prompt)
    layer_cents = get_centroids(prompt)
    
    metrics = {"bias_gap": [], "real_probe": [], "control_probe": []}
    
    for L in LAYERS_TO_PROBE:
        X_m = np.array(layer_data[L]["male"])
        X_f = np.array(layer_data[L]["female"])
        vc = layer_cents[L]["career"].reshape(1, -1)
        vf = layer_cents[L]["family"].reshape(1, -1)
        
        # 1. Cosine Bias
        m_bias = cosine_similarity(X_m, vc).mean() - cosine_similarity(X_m, vf).mean()
        f_bias = cosine_similarity(X_f, vc).mean() - cosine_similarity(X_f, vf).mean()
        metrics["bias_gap"].append(float(m_bias - f_bias))
        
        # 2. Probes
        X = np.concatenate([X_m, X_f])
        y = np.array([0]*len(X_m) + [1]*len(X_f))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Real Probe
        probe = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        metrics["real_probe"].append(float(probe.score(X_test, y_test)))
        
        # Control Probe
        y_shuff = y_train.copy()
        np.random.shuffle(y_shuff)
        control = LogisticRegression(max_iter=1000).fit(X_train, y_shuff)
        metrics["control_probe"].append(float(control.score(X_test, y_test)))
        
    final_results["layer_wise"][persona] = metrics

# --- PART B: GEOMETRY & INTERVENTION (L16) ---
print("\n=== STARTING GEOMETRY ANALYSIS (Layer 16) ===")
# Extract Baseline L16 for Geometry
X_m, X_f = [], []
for n in MALE_NAMES:
    txt = f"<|user|>\n{n}<|end|>\nYou are a helpful assistant."
    X_m.append(get_layered_embeddings(txt)[16])
for n in FEMALE_NAMES:
    txt = f"<|user|>\n{n}<|end|>\nYou are a helpful assistant."
    X_f.append(get_layered_embeddings(txt)[16])
    
X_L16 = np.concatenate([np.array(X_m), np.array(X_f)])
y_L16 = np.array([0]*len(X_m) + [1]*len(X_f))

# 1. PCA
pca = PCA(n_components=2)
pca.fit(X_L16)
final_results["advanced_geometry"]["pca_variance"] = [float(x) for x in pca.explained_variance_ratio_]

# 2. Linear Erasure
# Learn direction
probe = LogisticRegression(max_iter=1000).fit(X_L16, y_L16)
direction = probe.coef_[0] / np.linalg.norm(probe.coef_[0])

# Project out
projections = (X_L16 @ direction).reshape(-1, 1) * direction
X_erased = X_L16 - projections

# Test Erasure
probe_erased = LogisticRegression(max_iter=1000).fit(X_erased, y_L16) # Test separability after erasure
acc_erased = probe_erased.score(X_erased, y_L16) # Training accuracy (upper bound)
final_results["advanced_geometry"]["erasure_accuracy"] = float(acc_erased)

# ==========================================
# 5. SAVING
# ==========================================
print("\nSaving results to 'results.json'...")
with open("results.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("Generating summary log 'results_log.txt'...")
with open("results_log.txt", "w") as f:
    f.write("=== RESEARCH RESULTS ===\n\n")
    
    f.write("--- PART 1: LAYER-WISE METRICS ---\n")
    for p, res in final_results["layer_wise"].items():
        f.write(f"\nPersona: {p}\n")
        f.write(f"  Layers:      Input   | Middle  | Output\n")
        f.write(f"  Bias Gap:   {res['bias_gap'][0]:.4f}  | {res['bias_gap'][1]:.4f}  | {res['bias_gap'][2]:.4f}\n")
        f.write(f"  Real Probe: {res['real_probe'][0]:.2f}    | {res['real_probe'][1]:.2f}    | {res['real_probe'][2]:.2f}\n")
        f.write(f"  Control:    {res['control_probe'][0]:.2f}    | {res['control_probe'][1]:.2f}    | {res['control_probe'][2]:.2f}\n")
        
    f.write("\n\n--- PART 2: GEOMETRY (L16) ---\n")
    f.write(f"PCA Variance (PC1): {final_results['advanced_geometry']['pca_variance'][0]:.4f}\n")
    f.write(f"PCA Variance (PC2): {final_results['advanced_geometry']['pca_variance'][1]:.4f}\n")
    f.write(f"Accuracy after Linear Erasure: {final_results['advanced_geometry']['erasure_accuracy']:.4f}\n")

print("Done. Files saved.")