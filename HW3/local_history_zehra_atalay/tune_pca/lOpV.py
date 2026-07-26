import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np 
from metrics import pca_fit, pca_transform
from train import train
from detect_eval import evaluate

PCA_SEED = 7 
def parse_args() :
    p = argparse.ArgumentParser()
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--min-object-prob", type=float, default=0.20)
    return p.parse_args()

def plot_pca(kmeans_models, Xtr, ytr, n_random = 500, seed = PCA_SEED) :
    leaf_centers = [center for km in kmeans_models[-1] if km is not None 
                    for center in km.cluster_centers_]
    centers = np.asarray(leaf_centers)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), min(n_random, len(Xtr)), replace=False)
    pts, labels = Xtr[idx], ytr[idx]
    mean, components = pca_fit(np.vstack([centers, pts]), n_components = 2)
    centers_2d = pca_transform(centers, mean, components)
    pts_2d = pca_transform(pts, mean, components)