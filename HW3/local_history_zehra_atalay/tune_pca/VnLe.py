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
    plt.figure()
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="s", c="black", s=60)
    plt.title("K-Tree - PCA")
    plt.savefig("ktree_pca.png", dpi = 160)
    plt.close()

def suggest_best(rows) :
    best = max(rows, key=lambda r : (r[6], -r[2]))
    return int(best[0]), int(best[1])
def plot_sweep(rows) : 
    labels = [f"K{int(r[0])}D{int(r[1])}" for r in rows]
    times = [r[2] for r in rows]
    macro_f1 = [r[6] for r in rows]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(rows)), times, marker="o", label="training time")
    ax2.plot(range(len(rows)), macro_f1, marker="s", label="macro F1")
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel("sure (s)")
    ax2.set_ylabel("macro F1 (object classes)")
    fig.tight_layout()
    plt.savefig("hyperparam_sweep.png", dpii=160)
    plt.close()

def hyperparam_sweep(K_list, D_list, iou_thresh = 0.3, margin = 0.08, min_object_prob = 0.20) : 
    with open("raw_images.pkl", "rb") as f : 
        val_raw = pickle.load(f)["val"]
    rows = []
    best_payload = None
    best_macro_f1 = -1.0
    for K in K_list : 
        for D in D_list : 
            kmeans_models, leaf_counts, class_totals, elapsed = train(K=K, D=D)
            cm, report_rows, acc = evaluate(val_raw, kmeans_models, K, leaf_counts, class_totals, iou_thresh=iou_thresh, min_margin=margin, min_object_prob=min_object_prob,)
            object_rows = report_rows[1 :]
            macro_precision = float(np.mean([r[1] for r in object_rows]))
            macro_recall = float(np.mean([r[2] for r in object_rows]))
            macro_f1 = float(np.mean([r[3] for r in object_rows]))
            rows.append(((K, D, elapsed, acc, macro_precision, macro_recall, macro_f1)))

            if macro_f1 > best_macro_f1 :
                best_macro_f1 = macro_f1
                best_payload = (K, D, kmeans_models, leaf_counts, class_totals, elapsed)

    if best_payload is not None : 
        K, D, models, leaf_counts, class_totals, elapsed=best_payload
        with open("ktree_model.pkl", "wb") as f : 
            pickle.dump({"kmeans_models" : models, "leaf_counts" : leaf_counts, "class_totals" : class_totals, "K" : K, "D" : D, "train_time" : elapsed, }, f)

    return rows

if __name__ == "__main__" : 
    args = parse_args()
    rows = hyperparam_sweep(K_list=[4, 8, 16], D_list=[2, 3, 4], iou_thresh=args.iou, margin=args.margin, min_object_prob=args.min_object_prob,)
    plot_sweep(rows)