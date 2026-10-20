import numpy as np, matplotlib.pyplot as plt 
from ktree import build_ktree
from metrics import pca_fit, pca_transform
K, D = 8, 3
N_RANDOM = 500
SEED = 7
data = np.load("features.npz")
Xtr, ytr = data["Xtr"], data["ytr"]
kmeans_models, _ = build_ktree(Xtr, K, D)
leaf_centers = [c for km in kmeans_models[-1] if km is not None for c in km.cluster_centers_]
centers = np.asarray(leaf_centers)
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Xtr), min(N_RANDOM, len(Xtr)), replace=False)
pts, labels = Xtr[idx], ytr[idx]
mean, comp = pca_fit(np.vstack([centers, pts]), n_components=2)
centers_2d = pca_transform(centers, mean, comp)
pts_2d = pca_transform(pts, mean, comp)
plt.figure(figsize=(6,6))
plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6, label="egitim ozniteligi")
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="s", c="black", s=15, alpha=0.4, label="leaf merkezi")
plt.legend()
plt.title(f"K-Tree - PCA (K={K}, D={D}, {len(centers)} leaf)")
plt.savefig("ktree_pca_clean.png", dpi=160)
plt.close()
