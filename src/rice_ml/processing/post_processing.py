"""
Post-training evaluation metrics for classification and regression tasks.

This module provides common metrics with robust validation and NumPy-based docstrings suitable for doctest and pytest. Functions are designed to be scikit-learn-like but rely only on NumPy.

Classification:
---------------
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
classification_report
roc_auc_score
log_loss

Regression:
-----------
mse
rmse
mae
r2_score
"""
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Union, Any, Dict
import numpy as np

__all__ = [
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'roc_auc_score',
    'log_loss',
    'mse',
    'rmse',
    'mae',
    'r2_score',
]

ArrayLike = Union[Sequence[Any], Sequence]
NumArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]

# --- Validation Helpers ---

def _ensure_1d(y: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array; got {arr.ndim}D array instead.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr

def _ensure_1d_numeric(y: NumArrayLike, name: str) -> np.ndarray:
    arr = _ensure_1d(y, name)
    if not np.issubdubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} must contain numeric values; got {arr.dtype} instead.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr

def _validate_pair(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    yt = _ensure_1d(y_true, "y_true")
    yp = _ensure_1d(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and y_pred must have the same length; got {yt.shape[0]} and {yp.shape[0]}.")
    return yt, yp

def _validate_probs(y_true: ArrayLike, y_prob: ArrayLike) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Validate probabilities for log_loss/roc_auc.
    
    Supports:
    - Binary: y_prob shape (n,) or (n, 2) (prob of class 1 or [p0, p1]).
    - Multiclass: y_prob shape (n,K).
    
    Returns:
    -------
    y_true_labels: np.ndarray (1D)
    probs: np.ndarray (2D: (n, K))
    K: int (number of classes)
    """
    yt = _ensure_1d(y_true, "y_true")
    probs = np.asarray(y_prob)
    if probs.ndim == 1:
        probs = probs.astype(float)
        if probs.shape[0] != yt.shape[0]:
            raise ValueError(f"y_true and y_prob must have the same length; got {yt.shape[0]} and {probs.shape[0]}.")
        # Interpret  as prob of positive class (class == positive label)
        probs = np.stack([1.0 - probs, probs], axis=1)
        K = 2
    elif probs.ndim == 2:
        if probs.shape[0] != yt.shape[0]:
            raise ValueError("y_prob must have same first dimension as y_true.")
        probs = probs.astype(float)
        K = probs.shape[1]
    else:
        raise ValueError(f"y_prob must be 1D or 2D array; got {probs.ndim}D array instead.")
    if np.any(probs < 0) or np.any(probs > 1) or np.any(~np.isfinite(probs)):
        raise ValueError("All probability values must be in the range [0, 1] and finite.")
    return yt, probs, K

def _get_classification_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # --- Simplified label handling ---
    if labels is None:
        # Get all unique labels, sorted, for the matrix order
        labels_out = np.unique(np.concatenate((y_true.astype(str), y_pred.astype(str))))
    else:
        # Use provided labels (converted to string) as the order
        labels_out = np.array([str(lab) for lab in labels])
        
    # Generate the confusion matrix using the fixed function
    conf = confusion_matrix(y_true, y_pred, labels=labels_out)
    
    L = len(labels_out)
    total_samples = conf.sum()
    
    # Per-class metrics calculation
    tp = np.diag(conf).astype(float)
    fp = conf.sum(axis=0) - tp # column sum - tp
    fn = conf.sum(axis=1) - tp # row sum - tp
    tn = total_samples - (tp + fp + fn)
    support = conf.sum(axis=1) # row sum (true positives + false negatives)
    
    return tp, fp, tn, fn, support, labels_out

# ------ Classification Metrics ------

def  accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Classification accuracy score.
    """
    yt, yp = _validate_pair(y_true, y_pred)
    yt_str = yt.astype(str)
    yp_str = yp.astype(str)
    return float(np.mean(yt_str == yp_str))


def _per_class_metric(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class TP, FP, TN, FN, support.
    """
    yt, yp = _validate_pair(y_true, y_pred)
    yt_str = yt.astype(str)
    yp_str = yp.astype(str)
    if labels is not None:
        labels = [str(lab) for lab in labels]
    else:
        labels_str = None
    tp, fp, fn, _, support, labels_out = _get_classification_stats(yt_str, yp_str, labels_str)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp /(tp + fp), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where((tp + fn) > 0, tp /(tp + fn), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * (prec * rec) /(prec + rec), 0.0)
    return prec, rec, f1, support, labels_out

def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None
) -> Union[float, np.ndarray]:
    """
    Precision score for classification.
    
    Parameters
    ----------
    ...
    average : {'binary', 'micro', 'macro', 'weighted', None}, default = "binary"
        - "weighted": mean of per-class precision weighted by support.
        - None: return per-class array (aligned with 'labels' or sorted unique labels).
    """
    prec, _, _, support, _ = _per_class_metric(y_true, y_pred, labels)
    
    if average in ["macro", "weighted", None]:
       if support.sum() == 0:
           return 0.0 if average in ["macro", "weighted"] else np.array([])
    elif average == "micro":
        # micro precision == Micro recall == Micro f1 == Accuracy in multi-class
        return accuracy_score(y_true, y_pred)
    elif average == "binary":
        yt, yp = _validate_pair(y_true, y_pred)
        uniq = np.unique(np.concatenate((yt, yp)))
        if uniq.size != 2:
            raise ValueError("binary average requires exactly two classes in y_true and y_pred.")
        pos_label = np.max(uniq)
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fp = np.sum((yt != pos_label) & (yp == pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    else:
        raise ValueError('average must be one of {"binary", "micro", "macro", "weighted", None}.')
    if average == "macro":
        return float(np.mean(prec))
    if average == "weighted":
        return float(np.sum(prec * support) / support.sum())
    if average is None:
        return prec # per-class scores
    # should be unreachable
    return 0.0

def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None
) -> Union[float, np.ndarray]:
    """
    Recall score for classification.
    Parameters
    ----------
    ...
    average : {'binary', 'micro', 'macro', 'weighted', None}, default = "binary"
        Averaging method as in precision_score.
    """
    _, rec, _, support, _ = _per_class_metric(y_true, y_pred, labels)
    
    if average in ["macro", "weighted", None]:
       if support.sum() == 0:
           return 0.0 if average in ["macro", "weighted"] else np.array([])
    elif average == "micro":
        return accuracy_score(y_true, y_pred)
    elif average == "binary":
        yt, yp = _validate_pair(y_true, y_pred)
        uniq = np.unique(np.concatenate((yt, yp)))
        if uniq.size != 2:
            raise ValueError("binary average requires exactly two classes in y_true and y_pred.")
        pos_label = np.max(uniq)
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fn = np.sum((yt == pos_label) & (yp != pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        raise ValueError('average must be one of {"binary", "micro", "macro", "weighted", None}.')
    if average == "macro":
        return float(np.mean(rec))
    if average == "weighted":
        return float(np.sum(rec * support) / support.sum())
    if average is None:
        return rec # per-class scores
    return 0.0

def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None
) -> Union[float, np.ndarray]:
    """
    F1 score (harmonic mean of precision and recall) for classification.
    Parameters
    ----------
    ...
    average : {'binary', 'micro', 'macro', 'weighted', None}, default = "binary"
        Averaging method as in precision_score.
    """
    _, _, f1, support, _ = _per_class_metric(y_true, y_pred, labels)
    
    if average in ["macro", "weighted", None]:
       if support.sum() == 0:
           return 0.0 if average in ["macro", "weighted"] else np.array([])
    elif average == "micro":
        return accuracy_score(y_true, y_pred)
    elif average == "binary":
        p = precision_score(y_true, y_pred, average="binary")
        r = recall_score(y_true, y_pred, average="binary")
        return float(0.0 if (p + r) == 0 else 2 * (p * r) / (p + r))
    else:
        raise ValueError('average must be one of {"binary", "micro", "macro", "weighted", None}.')
    if average == "macro":
        return float(np.mean(f1))
    if average == "weighted":
        return float(np.sum(f1 * support) / support.sum())
    if average is None:
        return f1 # per-class scores
    return 0.0

def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[Sequence] = None
) -> np.ndarray:
    """
    Confusion matrix for classification.

    Parameters
    ----------
    y_true : ArrayLike, shape (n_samples,)
        True target values.
    y_pred : ArrayLike, shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : sequence, optional
        List of labels to index the matrix. If None, all unique labels 
        in y_true and y_pred are used, sorted alphabetically.

    Returns
    -------
    conf : np.ndarray, shape (L, L)
        Confusion matrix where L is the number of labels.
        Row i corresponds to true label i, and column j corresponds to predicted label j.
    """
    yt, yp = _validate_pair(y_true, y_pred)
    
    # 1. Determine the final, ordered list of labels (labels_out)
    if labels is not None:
        # Use provided labels and convert them all to strings for consistency
        labels_out = np.array([str(lab) for lab in labels])
    else:
        # Get all unique labels from both arrays, convert to string, and sort
        all_labels = np.unique(np.concatenate((yt.astype(str), yp.astype(str))))
        # Filter out labels that don't appear in either y_true or y_pred (already done by unique)
        labels_out = np.sort(all_labels)

    L = len(labels_out)
    
    # 2. Create the mapping from label value (string) to matrix index (integer)
    label_to_idx = {lab: i for i, lab in enumerate(labels_out)}
    
    # 3. Convert y_true and y_pred into arrays of indices
    
    # Index array for true labels
    y_true_idx = np.array([label_to_idx.get(str(lab), -1) for lab in yt])
    # Index array for predicted labels
    y_pred_idx = np.array([label_to_idx.get(str(lab), -1) for lab in yp])
    
    # 4. Initialize and populate the confusion matrix
    conf = np.zeros((L, L), dtype=int)
    
    # Populate the matrix using a loop over the indices
    for t, p in zip(y_true_idx, y_pred_idx):
        # Skip if label was not in the 'labels' sequence
        if t != -1 and p != -1: 
            conf[t, p] += 1
            
    return conf

def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[Sequence] = None,
    output_dict: bool = False,
    digits: int = 4
) -> Union[str, Dict[Any, Dict[str, float]]]:
    """
    Build a text report showing the main classification metrics.
    
    Parameters
    ----------
    y_true : ArrayLike, shape (n_samples,)
        True labels.
    y_pred : ArrayLike, shape (n_samples,)
        Predicted labels.
    labels : sequence, optional
        class label order
    output_dict : bool, default=False
        If True, return output as dict instead of string.
    digits : int, default=4
        Number of digits for formatting precision.
    Returns
    -------
    str or dict
        A formatted string or dictionary report
    
    Examples
    --------
    report = classification_report([0,1,1,0], [0,1,0,0])
    print(report.strip()    )
    """
    prec, rec, f1, support, labels_out = _per_class_metric(y_true, y_pred, labels)
    report_dict = {}
    for i, lab in enumerate(labels_out):
        report_dict[str(lab)] = {
            "precision": prec[i],
            "recall": rec[i],
            "f1-score": f1[i],
            "support": int(support[i])
        }
    total_support = support.sum()
    report_dict["accuracy"] = {'precision': accuracy_score(y_true, y_pred), 'support': total_support}
    report_dict['macro avg'] = {
        'precision': np.mean(prec),
        'recall': np.mean(rec),
        'f1-score': np.mean(f1),
        'support': total_support
    }
    report_dict['weighted avg'] = {
        'precision': np.sum(prec * support) / total_support,
        'recall': np.sum(rec * support) / total_support,
        'f1-score': np.sum(f1 * support) / total_support,
        'support': total_support
    }
    
    if output_dict:
        return report_dict
    
    # build string report
    fmt = f"{{:<{max(len(str(l)) for l in labels_out)}s}} {{:>{digits+2}.{digits}f}} {{:>{digits+2}.{digits}f}} {{:>{digits+2}.{digits}f}} {{:>9d}}"
    header = f"{{:<{max(len(str(l)) for l in labels_out)}s}} {{:>10s}} {{:>10s}} {{:>10s}} {{:>9s}}".format("label", "precision", "recall", "f1-score", "support")
    lines = [header, "-" * len(header)]
    
    for label in labels_out:
        data = report_dict[str(label)]
        lines.append(fmt.format(str(label), data['precision'], data['recall'], data['f1-score'], data['support']))
    lines.append("-" * len(header))
    acc_data = report_dict["accuracy"]
    lines.append(f"{{:<{len('accuracy'):}s}} {{:>34.4f}} {{:>9d}}".format("accuracy", acc_data['precision'], acc_data['support']))
    lines.append("")
    
    for avg in ['macro avg', 'weighted avg']:
        data = report_dict[avg]
        lines.append(fmt.format(avg, data['precision'], data['recall'], data['f1-score'], data['support']))
    
    return "\n".join(lines)

def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """
    Compute ROC AUC score for binary classification.
    
    Parameters
    ----------
    y_true : ArrayLike, shape (n_samples,)
        True binary labels.
    y_score : ArrayLike, shape (n_samples,) or (n_samples, 2)
        Target scores (probabilities or decision function).
    
    Returns
    -------
    float
        ROC AUC score.
    """
    yt = _ensure_1d(y_true, "y_true")
    ys = _ensure_1d_numeric(y_score, "y_score")
    uniq = np.unique(yt)
    if uniq.size != 2:
       raise ValueError("ROC AUC score is only defined for binary classification.")
    if np.all(yt == uniq[0]) or np.all(yt == uniq[1]):
       raise ValueError("y_true must contain at least one sample from each class.")
   # Rank-based AUC calculation (Mann-Whitney U statistic)
   # Compure AUC = (sum of ranks for positive class - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    order = np.argsort(ys, kind = "mergesort")
    ranks = np.empty_like(order=float)
    ranks[order] = np.arange(1, len(ys) + 1)
   
    pos = yt == uniq.max()
    n_pos = np.sum(pos)
    n_neg = len(yt) - n_pos
    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def log_loss(y_true: ArrayLike, y_prob: ArrayLike, *, eps: float = 1e-15) -> float:
    """
    Logarithmic loss for classification.
    """
    yt, probs, K = _validate_probs(y_true, y_prob)
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be a positive finite value.")
    # Determine label mapping to columns
    labels = np.unique(yt)
    if K == 2 and labels.size == 2:
        labels_to_col = {labels.min(): 0, labels.max(): 1}
    else:
        if labels.size != K:
            labels = np.arrange(K)
        labels_to_col = {lab: i for i, lab in enumerate(labels)}
    # If probs is 2D, require rows to sum to 1 (within tolerance)    
    if probs.ndim == 2:
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol = 0.0):
            raise ValueError("Each probability row must sum to 1 within tolerance.")
    p = np.clip(probs, eps, 1.0)
    
    # Gather probabilities for true labels
    cols = np.array([labels_to_col.get(lab, None) for lab in yt])
    if np.any(cols == None):
        raise ValueError("Could not map some true labels to probability columns.")
    ll = -np.log(p[np.arrange(len(yt)), cols.astype(int)])
    return float(np.mean(ll))

# ------ Regression Metrics ------
def mse(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Mean squared error.
    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Root mean squared error.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Mean absolute error.
    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Coefficient of determination R^2.
    
    R^2 = 1 - SS_res / SS_tot, where SS_tot uses y_true mean.

    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    ss_res = np.sum((yt - yp) ** 2)
    y_mean = np.mean(yt)
    ss_tot = np.sum((yt - y_mean) ** 2)
    if ss_tot == 0:
        # Undefined per many libs; tests expect a ValueError unless perfect predictions
        if ss_res == 0:
            return 1.0
        raise ValueError("R^2 is undefined when y_true is constant and predictions are not perfect.")
    return float(1.0 - ss_res / ss_tot)