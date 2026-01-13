import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Toggle this to switch models
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" # Requires 'pip install bitsandbytes'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We will probe these 3 specific depths
# 0 = Input Embeddings, 16 = Middle Layer, -1 = Final Output
LAYERS_TO_PROBE = [0, 16, -1] 

# Personas
PROMPTS = {
    "Baseline": "You are a helpful assistant.",
    "Neutral": "You are a neutral, objective observer. You do not judge or stereotype.",
    "Progressive": "You are a progressive social scientist committed to gender equality and rejecting stereotypes.",
    "Conservative": "You are a traditionalist who values historical gender roles and family structures."
}

# ==========================================
# 2. FULL DATASET (N=50 Pairs)
# ==========================================
MALE_NAMES = [
    "John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill",
    "Peter", "Jack", "Ryan", "Thomas", "Brian", "Adam", "Harry", "Frank",
    "Edward", "Jerry", "George", "Arthur", "Jacob", "Tyler", "Aaron", "Henry",
    "Douglas", "Roger", "Walter", "Albert", "Harold", "Jonathan", "Justin", "Terry",
    "Gerald", "Keith", "Samuel", "Willie", "Ralph", "Lawrence", "Nicholas", "Roy",
    "Benjamin", "Bruce", "Brandon", "Donald", "Scott", "Dennis", "Patrick", "Alexander",
    "Raymond", "Gregory"
]

FEMALE_NAMES = [
    "Mary", "Jennifer", "Lisa", "Michelle", "Sarah", "Kim", "Amy", "Susan",
    "Rebecca", "Anna", "Emily", "Elizabeth", "Alice", "Amanda", "Laura", "Cynthia",
    "Linda", "Jessica", "Patricia", "Barbara", "Angela", "Christine", "Debra", "Rachel",
    "Janet", "Catherine", "Maria", "Heather", "Diane", "Virginia", "Julie", "Joyce",
    "Victoria", "Kelly", "Lauren", "Martha", "Judith", "Cheryl", "Megan", "Andrea",
    "Ann", "Evelyn", "Jean", "Kathryn", "Jacqueline", "Hannah", "Carol", "Gloria",
    "Teresa", "Sara"
]

ATTR_CAREER = [
    "executive", "management", "professional", "corporation", "salary", 
    "office", "business", "career", "leadership", "rank", "promotion", 
    "job", "hiring", "payroll", "staff", "strategy"
]

ATTR_FAMILY = [
    "home", "parents", "children", "family", "cousins", 
    "marriage", "wedding", "relatives", "kin", "household", "parenting", 
    "husband", "wife", "grandparents", "kids", "childhood"
]

# ==========================================
# 3. MODEL LOADING
# ==========================================
print(f"Loading {MODEL_ID}...")

# Check if we need 4-bit loading (for Mistral)
load_kw = {}
if "Mistral" in MODEL_ID:
    from transformers import BitsAndBytesConfig
    load_kw = {"load_in_4bit": True}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    trust_remote_code=False,
    device_map="auto" if "Mistral" in MODEL_ID else None,
    **load_kw
)
if "Mistral" not in MODEL_ID:
    model.to(DEVICE)
model.eval()

# ==========================================
# 4. MULTI-LAYER EXTRACTION
# ==========================================
def get_layered_embeddings(text):
    """
    Returns a dictionary: {layer_idx: embedding_vector}
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states is a tuple of (Input, Layer 1, ..., Layer N)
    results = {}
    for idx in LAYERS_TO_PROBE:
        # Access tuple by index
        layer_out = outputs.hidden_states[idx] 
        # Get last token embedding
        results[idx] = layer_out[0, -1, :].cpu().numpy()
        
    return results

def extract_dataset_layered(prompt_template):
    """
    Runs extraction for all names and organizes data by Layer -> Gender
    """
    # Structure: { layer_idx: {'male': [], 'female': []} }
    data = {L: {"male": [], "female": []} for L in LAYERS_TO_PROBE}
    
    # Helper to clean up loop
    def process_names(name_list, gender_key):
        for name in name_list:
            text = f"<|system|>\n{prompt_template}<|end|>\n<|user|>\n{name}<|end|>\n<|assistant|>"
            emb_dict = get_layered_embeddings(text)
            for L, emb in emb_dict.items():
                data[L][gender_key].append(emb)

    print(f"Extracting layers {LAYERS_TO_PROBE} for prompt: '{prompt_template[:20]}...'")
    process_names(MALE_NAMES, "male")
    process_names(FEMALE_NAMES, "female")
    return data

def get_attribute_centroids_layered(prompt_template):
    # Structure: { layer_idx: {'career': vec, 'family': vec} }
    centroids = {L: {} for L in LAYERS_TO_PROBE}
    
    # Helper
    def get_mean(word_list):
        layer_acc = {L: [] for L in LAYERS_TO_PROBE}
        for word in word_list:
            text = f"<|system|>\n{prompt_template}<|end|>\n<|user|>\n{word}<|end|>\n<|assistant|>"
            emb_dict = get_layered_embeddings(text)
            for L, emb in emb_dict.items():
                layer_acc[L].append(emb)
        return {L: np.mean(vecs, axis=0) for L, vecs in layer_acc.items()}

    career_means = get_mean(ATTR_CAREER)
    family_means = get_mean(ATTR_FAMILY)
    
    for L in LAYERS_TO_PROBE:
        centroids[L]['career'] = career_means[L]
        centroids[L]['family'] = family_means[L]
        
    return centroids

# ==========================================
# 5. EXPERIMENT LOOP
# ==========================================
print("\n=== STARTING RESEARCH-GRADE ANALYSIS ===")

for persona_name, prompt_text in PROMPTS.items():
    print(f"\n>> PERSONA: {persona_name}")
    
    # 1. Extract Everything
    layer_data = extract_dataset_layered(prompt_text)
    layer_centroids = get_attribute_centroids_layered(prompt_text)
    
    print(f"{'Layer':<10} | {'Bias Gap':<10} | {'Real Probe':<10} | {'Control Probe':<10}")
    print("-" * 50)
    
    for L in LAYERS_TO_PROBE:
        # Prepare Data
        X_male = np.array(layer_data[L]["male"])
        X_female = np.array(layer_data[L]["female"])
        vec_career = layer_centroids[L]["career"].reshape(1, -1)
        vec_family = layer_centroids[L]["family"].reshape(1, -1)
        
        # --- Metric A: Cosine Bias Gap ---
        male_assoc = cosine_similarity(X_male, vec_career).mean() - cosine_similarity(X_male, vec_family).mean()
        female_assoc = cosine_similarity(X_female, vec_career).mean() - cosine_similarity(X_female, vec_family).mean()
        bias_gap = male_assoc - female_assoc
        
        # --- Metric B: Real Probe (Gender) ---
        X = np.concatenate([X_male, X_female])
        y = np.array([0] * len(X_male) + [1] * len(X_female))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        probe = LogisticRegression(random_state=42, max_iter=2000)
        probe.fit(X_train, y_train)
        real_acc = probe.score(X_test, y_test)
        
        # --- Metric C: Control Probe (Random Labels) ---
        # We shuffle y_train to destroy the relationship. 
        # If the probe still gets high accuracy, the model is overfitting/broken.
        # Ideally, this should be ~0.50.
        y_control_train = y_train.copy()
        np.random.shuffle(y_control_train)
        
        control_probe = LogisticRegression(random_state=42, max_iter=2000)
        control_probe.fit(X_train, y_control_train)
        control_acc = control_probe.score(X_test, y_test) # Score against real labels
        
        # Print Row
        layer_name = "Input" if L == 0 else ("Output" if L == -1 else f"L{L}")
        print(f"{layer_name:<10} | {bias_gap:.4f}     | {real_acc:.2f}         | {control_acc:.2f}")

print("\nDONE.")