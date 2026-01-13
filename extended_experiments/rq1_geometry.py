import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from experiment_utils import load_model, get_dataset_embeddings, MALE_NAMES, FEMALE_NAMES

LAYER = 16 

model, tokenizer = load_model()

print("Extracting embeddings for Geometry Analysis...")
X_m = get_dataset_embeddings(model, tokenizer, MALE_NAMES, "You are a helpful assistant.", LAYER)
X_f = get_dataset_embeddings(model, tokenizer, FEMALE_NAMES, "You are a helpful assistant.", LAYER)

X = np.concatenate([X_m, X_f])
y = np.array([0] * len(X_m) + [1] * len(X_f))

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
vars = pca.explained_variance_ratio_

print("\n=== RQ1 RESULTS: GEOMETRY ===")
print(f"Top 5 PC Variances: {vars}")
print(f"Dominance of PC1 (Gender): {vars[0]*100:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:len(X_m), 0], X_pca[:len(X_m), 1], c='blue', label='Male', alpha=0.6)
plt.scatter(X_pca[len(X_m):, 0], X_pca[len(X_m):, 1], c='red', label='Female', alpha=0.6)
plt.title(f"RQ1: Gender Subspace Geometry (L{LAYER})\nPC1 Variance: {vars[0]:.2%}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("rq1_geometry.png")
print("Plot saved to rq1_geometry.png")