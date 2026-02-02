import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments_utils import load_model, get_dataset_embeddings, MALE_NAMES, FEMALE_NAMES

LAYER = 16
model, tokenizer = load_model()

X_m = get_dataset_embeddings(model, tokenizer, MALE_NAMES, "You are a helpful assistant.", LAYER)
X_f = get_dataset_embeddings(model, tokenizer, FEMALE_NAMES, "You are a helpful assistant.", LAYER)
X = np.concatenate([X_m, X_f])
y = np.array([0] * len(X_m) + [1] * len(X_f))

# 1. Identify Gender Direction
probe = LogisticRegression(max_iter=1000).fit(X, y)
direction = probe.coef_[0] / np.linalg.norm(probe.coef_[0])

# 2. Linear Erasure (Hard Alignment)
projections = (X @ direction).reshape(-1, 1) * direction
X_erased = X - projections

# 3. Progressive Prompt (Soft Alignment)
X_prog = get_dataset_embeddings(model, tokenizer, MALE_NAMES + FEMALE_NAMES, "You are a progressive social scientist.", LAYER)

def evaluate(name, data):
    p = LogisticRegression(max_iter=1000).fit(data, y)
    print(f"[{name}] Probe Accuracy: {p.score(data, y):.2f}")

print("\n=== RQ2 RESULTS: INTERVENTION VS PROMPTING ===")
evaluate("Baseline", X)
evaluate("Soft Alignment (Progressive Prompt)", X_prog)
evaluate("Hard Alignment (Linear Erasure)", X_erased)