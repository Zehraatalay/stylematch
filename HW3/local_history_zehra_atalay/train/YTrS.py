import numpy as np, pickle, time 
from ktree import build_ktree, build_leaf_class_counts
K, D = 8, 3 
def train( K = K, D = D, num_classes = 10, path = "features.npz") : 
    data = np.load(path)
    Xtr, ytr = data["Xtr"], data["ytr"]
    t0 = time.time()
    kmeans_models, cluster_label = build_ktree(Xtr, K, D)
    leaf_counts, class_totals = build_leaf_class_counts(cluster_label, ytr, num_classes=num_classes)
    train_time = time.time() - t0