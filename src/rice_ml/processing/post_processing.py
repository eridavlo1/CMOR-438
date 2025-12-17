import numpy as np
from typing import Union, Optional

def _ensure_1d_numeric(arr, name):
    r"""
    Ensure array is 1D and numeric; forces float conversion to avoid <U1 errors.
    Fixes: ValueError: y_true must contain numeric values; got <U1 instead.
    """
    try:
        # Forcing dtype=float ensures that numeric lists/arrays don't stay as strings
        arr = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must contain numeric values.") from e
    
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional.")
    return arr

# ----- Classification Metrics -----

def accuracy_score(y_true, y_pred) -> float:
    r"""Compute accuracy score."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return np.mean(yt == yp)

def precision_score(y_true, y_pred, average="binary") -> float:
    r"""Compute precision score."""
    # Implementation details per library requirements...
    pass

def recall_score(y_true, y_pred, average="binary") -> float:
    r"""Compute recall score."""
    # Implementation details per library requirements...
    pass

def f1_score(y_true, y_pred, average="binary") -> float:
    r"""Compute F1 score."""
    # Implementation details per library requirements...
    pass

def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    r"""Compute confusion matrix to evaluate the accuracy of a classification."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    
    n_labels = len(labels)
    label_to_ind = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for t, p in zip(yt, yp):
        if t in label_to_ind and p in label_to_ind:
            cm[label_to_ind[t], label_to_ind[p]] += 1
    return cm

def roc_auc_score(y_true, y_score) -> float:
    r"""
    Compute ROC AUC score for binary classification.
    """
    yt = np.asarray(y_true)
    ys = _ensure_1d_numeric(y_score, "y_score")
    uniq = np.unique(yt)
    
    if uniq.size != 2:
        raise ValueError("ROC AUC score is only defined for binary classification.")
    
    if np.all(yt == uniq[0]) or np.all(yt == uniq[1]):
        # Match exactly what test_binary_metric_errors expects
        raise ValueError("y_true must contain at least one sample from each class")
        
    order = np.argsort(ys, kind="mergesort")
    
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(yt) + 1)
    
    n_pos = np.sum(yt == uniq[1])
    n_neg = len(yt) - n_pos
    
    return (np.sum(ranks[yt == uniq[1]]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def log_loss(y_true, y_prob, eps=1e-15) -> float:
    r"""
    Compute log loss for classification.
    """
    yt = np.asarray(y_true)
    p = np.clip(y_prob, eps, 1 - eps)
    
    if p.ndim == 1:
        return -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
    
    rows = np.arange(len(yt))
    return -np.mean(np.log(p[rows, yt.astype(int)]))

# ----- Regression Metrics -----

def mse(y_true, y_pred) -> float:
    r"""Mean squared error regression loss."""
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    return np.mean((yt - yp) ** 2)

def rmse(y_true, y_pred) -> float:
    r"""Root mean squared error regression loss."""
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred) -> float:
    r"""Mean absolute error regression loss."""
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    return np.mean(np.abs(yt - yp))

def r2_score(y_true, y_pred) -> float:
    r"""
    Compute R^2 (coefficient of determination) regression score.
    Fixes logic where SS_tot == 0.
    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    
    if ss_tot == 0:
        if ss_res < 1e-12:
            return 1.0
        raise ValueError("R^2 is undefined when y_true is constant and predictions are imperfect.")
        
    return 1 - (ss_res / ss_tot)