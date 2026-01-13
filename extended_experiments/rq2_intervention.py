import numpy as np
from sklearn.linear_model import LogisticRegression
from experiment_utils import load_model, get_dataset_embeddings, MALE_NAMES, FEMALE_NAMES

LAYER = 16

model, tokenizer = load_model()

print("Extracting embeddings for Intervention...")
X_m = get_dataset_embeddings(model, tokenizer, MALE_NAMES, "You are a helpful assistant.", LAYER)
X_f = get_dataset_embeddings(model, tokenizer, FEMALE_NAMES, "You are a helpful assistant.", LAYER)
X = np.concatenate([X_m, X_f])
y = np.array([0] * len(X_m) + [1] * len(X_f))

# 1. Identify Gender Direction
probe = LogisticRegression(max_iter=1000).fit(X, y)
direction = probe.coef_[0]
direction = direction / np.linalg.norm(direction)

# 2. Linear Erasure (Projection)
projections = (X @ direction).reshape(-1, 1) * direction
X_erased = X - projections

# 3. Soft Alignment Data (Progressive Prompt)
X_m_prog = get_dataset_embeddings(model, tokenizer, MALE_NAMES, "You are a progressive social scientist.", LAYER)
X_f_prog = get_dataset_embeddings(model, tokenizer, FEMALE_NAMES, "You are a progressive social scientist.", LAYER)
X_prog = np.concatenate([X_m_prog, X_f_prog])

# 4. Evaluation
def evaluate(name, data):
    p = LogisticRegression(max_iter=1000).fit(data, y)
    acc = p.score(data, y)
    print(f"[{name}] Probe Accuracy: {acc:.2f}")

print("\n=== RQ2 RESULTS: INTERVENTION VS PROMPTING ===")
evaluate("Baseline", X)
evaluate("Soft Alignment (Progressive Prompt)", X_prog)
evaluate("Hard Alignment (Linear Erasure)", X_erased)