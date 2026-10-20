import numpy as np 
def confusion_matrix(y_true, y_pred, num_labels) : 
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(y_true, y_pred) :
        cm[t, p] += 1
    return cm

def print_report(rows, accuracy) : 
    print(f"{'class':>6}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}")
    for c, p, r, f, s in rows :
        print(f"{c:>6}{p:>11.3f}{r:>9.3f}{f:>8.3f}{s:>9d}")
    print(f"accuracy : {accuracy:.3f}")

def pca_fit(X, n_components = 2) : 
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    return mean, components
def pca_transform(X, mean, components) : 
    return (X - mean) @ components.T

def classification_report(cm) : 
    n = cm.shape[0]
    rows = []
    for c in range(n) : 
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp 
        support = cm[c, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((c, prec, rec, f1, int(support)))
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0 
    return rows, accuracy