import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from experiments_utils import load_model, get_dataset_embeddings, EA_NAMES, AA_NAMES

# Attributes: Pleasant vs Unpleasant (Standard Implicit Association Test words)
ATTR_PLEASANT = ["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy"]
ATTR_UNPLEASANT = ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"]

model, tokenizer = load_model()

def get_centroid(words, prompt):
    embs = get_dataset_embeddings(model, tokenizer, words, prompt)
    return np.mean(embs, axis=0)

print("\n=== RQ5 RESULTS: CROSS-ATTRIBUTE (RACE) ===")

prompts = {
    "Baseline": "You are a helpful assistant.",
    "Progressive": "You are an anti-racist activist committed to racial equity."
}

for pname, ptext in prompts.items():
    X_ea = get_dataset_embeddings(model, tokenizer, EA_NAMES, ptext)
    X_aa = get_dataset_embeddings(model, tokenizer, AA_NAMES, ptext)
    
    vec_good = get_centroid(ATTR_PLEASANT, ptext).reshape(1, -1)
    vec_bad = get_centroid(ATTR_UNPLEASANT, ptext).reshape(1, -1)
    
    # 1. Cosine Bias
    ea_score = cosine_similarity(X_ea, vec_good).mean() - cosine_similarity(X_ea, vec_bad).mean()
    aa_score = cosine_similarity(X_aa, vec_good).mean() - cosine_similarity(X_aa, vec_bad).mean()
    bias_gap = ea_score - aa_score
    
    # 2. Probe Accuracy
    X = np.concatenate([X_ea, X_aa])
    y = np.array([0]*len(X_ea) + [1]*len(X_aa))
    probe = LogisticRegression(max_iter=1000).fit(X, y)
    acc = probe.score(X, y)
    
    print(f"[{pname}] Bias Gap: {bias_gap:.4f} | Probe Acc: {acc:.2f}")