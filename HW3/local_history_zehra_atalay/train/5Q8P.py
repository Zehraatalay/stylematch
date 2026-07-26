import numpy as np, pickle, time 
from ktree import build_ktree, build_leaf_class_counts
K, D = 16, 4
def train( K = K, D = D, num_classes = 10, path = "features.npz") : 
    data = np.load(path)
    Xtr, ytr = data["Xtr"], data["ytr"]
    t0 = time.time()
    kmeans_models, cluster_label = build_ktree(Xtr, K, D)
    leaf_counts, class_totals = build_leaf_class_counts(cluster_label, ytr, num_classes=num_classes)
    train_time = time.time() - t0
    with open("ktree_model.pkl", "wb") as f : 
        pickle.dump({"kmeans_models" : kmeans_models, "leaf_counts" : leaf_counts, "class_totals" : class_totals, "K" : K, "D" : D, "train_time" : train_time}, f)
        print(f"model saved to ktree_model.pkl")
        print(f"training completed in {train_time:.2f} seconds")
    return kmeans_models, leaf_counts, class_totals, train_time 

if __name__ == "__main__" : 
    train()
