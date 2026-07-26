import numpy as np 
from sklearn.cluster import KMeans
def build_ktree(X,K,D) : 
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