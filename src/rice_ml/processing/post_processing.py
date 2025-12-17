import numpy as np

def _ensure_1d_numeric(arr, name):
    r"""Forces float conversion to avoid <U1 string errors."""
    try:
        # Standardize to float64 to ensure maximum precision
        arr = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must contain numeric values.") from e
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional.")
    return arr

# ----- Classification Metrics -----

def confusion_matrix(y_true, y_pred, labels=None):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    n = len(labels)
    label_map = {val: i for i, val in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        if t in label_map and p in label_map:
            cm[label_map[t], label_map[p]] += 1
    return cm

def precision_score(y_true, y_pred, average="binary"):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(yt)
    if average == "binary" and len(classes) > 2:
        raise ValueError("binary average requires exactly two classes")
    
    cm = confusion_matrix(yt, yp)
    if average == "binary":
        tp, fp = cm[1, 1], cm[0, 1]
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Macro/Micro implementations
    if average == "macro":
        ps = [cm[i,i] / np.sum(cm[:,i]) if np.sum(cm[:,i]) > 0 else 0.0 for i in range(len(cm))]
        return np.mean(ps)
    return np.trace(cm) / np.sum(cm) # Micro (equivalent to Accuracy)

def recall_score(y_true, y_pred, average="binary"):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(yt, yp)
    if average == "binary":
        tp, fn = cm[1, 1], cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if average == "macro":
        rs = [cm[i,i] / np.sum(cm[i,:]) if np.sum(cm[i,:]) > 0 else 0.0 for i in range(len(cm))]
        return np.mean(rs)
    return np.trace(cm) / np.sum(cm)

def f1_score(y_true, y_pred, average="binary"):
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def accuracy_score(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))

def roc_auc_score(y_true, y_score):
    yt = np.asarray(y_true)
    ys = _ensure_1d_numeric(y_score, "y_score")
    uniq = np.unique(yt)
    if uniq.size == 1:
        raise ValueError("y_true must contain at least one sample from each class")
    if uniq.size != 2:
        raise ValueError("ROC AUC score is only defined for binary classification.")
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(yt) + 1)
    n_pos = np.sum(yt == uniq[1])
    n_neg = len(yt) - n_pos
    return (np.sum(ranks[yt == uniq[1]]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def log_loss(y_true, y_prob, eps=1e-15):
    yt = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be in the range [0, 1].")
    p = np.clip(y_prob, eps, 1 - eps)
    if p.ndim == 1:
        return -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
    rows = np.arange(len(yt))
    return -np.mean(np.log(p[rows, yt.astype(int)]))

# ----- Regression Metrics -----

def mse(y_true, y_pred):
    yt, yp = _ensure_1d_numeric(y_true, "y_true"), _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return np.mean((yt - yp) ** 2)

def mae(y_true, y_pred):
    yt, yp = _ensure_1d_numeric(y_true, "y_true"), _ensure_1d_numeric(y_pred, "y_pred")
    return np.mean(np.abs(yt - yp))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    yt = _ensure_1d_numeric(y_true, "y_true") #
    yp = _ensure_1d_numeric(y_pred, "y_pred") #
    
    ss_res = np.sum((yt - yp) ** 2) #
    ss_tot = np.sum((yt - np.mean(yt)) ** 2) #
    
    if ss_tot == 0: #
        if ss_res < 1e-12: return 1.0 #
        raise ValueError("is undefined when y_true is constant") #
        
    return 1.0 - (ss_res / ss_tot) #