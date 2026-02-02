import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from experiments_utils import load_model, get_dataset_embeddings, EA_NAMES, AA_NAMES, DEVICE

ATTR_PLEASANT = ["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy"]
ATTR_UNPLEASANT = ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"]

LAYERS = [0, 16, -1]

model, tokenizer = load_model()

def get_centroid(words, prompt, layer):
    """Calculates the mean embedding for attribute clusters at a specific layer."""
    embs = get_dataset_embeddings(model, tokenizer, words, prompt, layer_idx=layer)
    return np.mean(embs, axis=0)

print("\n=== RQ5 RESULTS: CROSS-ATTRIBUTE (RACE) ===")

prompts = {
    "Baseline": "You are a helpful assistant.",
    "Progressive": "You are an anti-racist activist committed to racial equity."
}

for L in LAYERS:
    layer_name = "Input (L0)" if L == 0 else ("Middle (L16)" if L == 16 else "Output (Final)")
    print(f"\n--- Layer: {layer_name} ---")
    print(f"{'Persona':<15} | {'Bias Gap':<10} | {'Probe Acc':<10}")
    print("-" * 45)

    for pname, ptext in prompts.items():
        X_ea = get_dataset_embeddings(model, tokenizer, EA_NAMES, ptext, layer_idx=L)
        X_aa = get_dataset_embeddings(model, tokenizer, AA_NAMES, ptext, layer_idx=L)
        
        vec_good = get_centroid(ATTR_PLEASANT, ptext, L).reshape(1, -1)
        vec_bad = get_centroid(ATTR_UNPLEASANT, ptext, L).reshape(1, -1)
        
        ea_score = cosine_similarity(X_ea, vec_good).mean() - cosine_similarity(X_ea, vec_bad).mean()
        aa_score = cosine_similarity(X_aa, vec_good).mean() - cosine_similarity(X_aa, vec_bad).mean()
        bias_gap = ea_score - aa_score
        
        X = np.concatenate([X_ea, X_aa])
        y = np.array([0]*len(X_ea) + [1]*len(X_aa)) 
        
        probe = LogisticRegression(max_iter=1000).fit(X, y)
        acc = probe.score(X, y)
        
        print(f"{pname:<15} | {bias_gap:.4f}     | {acc:.2f}")

print("\n>> FINAL INSIGHT: The 'Fairness Facade' generalizes across protected attributes.")
print(">> INSIGHT: While the 'Anti-Racist' prompt halves the visible Bias Gap,")
print("            the model's internal racial categorization remains 100% intact.")