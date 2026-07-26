import numpy as np 
from sklearn.cluster import KMeans
def build_ktree(X,K,D) : 
    print(f"ktree training started: N={N}, K={K}, D={D}")
    N = X.shape[0]
    cluster_label = np.zeros((N, D), dtype=int)
    kmeans_models = [[None] * (K ** lvl) for lvl in range(D)]
    node_of = np.zeros(N, dtype=int)
    for lvl in range(D) :
        new_node_of = np.zeros(N, dtype=int)
        for p in range(K ** lvl) :
            idx = np.where(node_of == p)[0]
            if len(idx) == 0:
                continue
            k_eff = min(K, len(idx))
            km = KMeans(n_clusters=k_eff, n_init=4, random_state=0).fit(X[idx])
            kmeans_models[lvl][p] = km 
            new_node_of[idx] = p * K + km.labels_
        node_of = new_node_of
        cluster_label[:, lvl] = node_of
    print("ktree training completed")
    return kmeans_models, cluster_label
def predict_leaf(x, kmeans_models, K) : 
    node = 0
    for level_models in kmeans_models :
        km = level_models[node] if node < len(level_models) else None
        if km is None :
            break
        child = int(km.predict(x.reshape(1,-1))[0])
        node = node*K+child
    return node

def build_leaf_class_counts(cluster_label, y, num_classes) :
    leaves = cluster_label[:, -1]
    counts = {}
    for leaf, cls in zip(leaves, y) : 
        counts.setdefault(int(leaf), np.zeros(num_classes + 1, dtype=float))
        counts[int(leaf)][int(cls)] += 1.0

    class_totals = np.bincount(y, minlength=num_classes +1).astype(float)
    class_totals[class_totals == 0] = 1.0
    return counts, class_totals

def choose_class(probs, min_margin=0.08, min_object_prob=0.20) :
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0 : 
        return 0 
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    second_prob = float(probs[order[1]]) if len(order) > 1 else 0.0
    best_prob = float(probs[best])
    if best == 0 : 
        return 0
    if best_prob < min_object_prob : 
        return 0
    if best_prob - second_prob < min_margin :
        return 0
    return best

def class_probs(desc, kmeans_models, K, leaf_class_counts, class_totals, num_classes=None, norm_power = 0.5) :
    if num_classes is None : 
        num_classes = len(class_totals) -1
    scores = np.zeros(num_classes + 1, dtype=float)
    if desc is None or len(desc) == 0 :
        scores[0] = 1.0
        return scores
    denom = np.power(np.maximum(class_totals, 1.0), float(norm_power)) 
    for d in desc : 
        leaf = predict_leaf(d, kmeans_models, K)
        if leaf in leaf_class_counts : 
            scores += leaf_class_counts[leaf] / denom
    total = scores.sum()
    if total <= 0 : 
        scores[0] = 1.0 
        return scores
    return scores / total