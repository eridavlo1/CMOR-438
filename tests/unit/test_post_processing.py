import numpy as np
import pytest
from collections import Counter
from typing import List

# Assume correct import structure from your library
from rice_ml.processing import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mse,
    rmse,
    mae,
    r2_score,
)


# ----- Classification: Binary Metrics -----

def test_binary_basic_metrics():
    r"""Tests accuracy, precision, recall, F1, and confusion matrix for a basic binary case."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    # TP=1, FP=0, TN=2, FN=1
    # Total samples = 4
    
    # Accuracy = (2+1)/4 = 0.75
    assert accuracy_score(y_true, y_pred) == 0.75
    
    # Precision (binary, pos=1) = TP / (TP + FP) = 1 / (1 + 0) = 1.0
    assert precision_score(y_true, y_pred, average="binary") == 1.0
    
    # Recall (binary, pos=1) = TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert recall_score(y_true, y_pred, average="binary") == 0.5
    
    # F1 = 2 * P * R / (P + R) = 2 * 1.0 * 0.5 / 1.5 = 1.0 / 1.5 = 0.666...
    assert np.isclose(f1_score(y_true, y_pred, average="binary"), 2 * 1.0 * 0.5 / 1.5)

    # CM: rows=True, cols=Predicted. Labels [0, 1]
    # [[TN, FP], [FN, TP]] -> [[2, 0], [1, 1]]
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[2, 0], [1, 1]]


def test_roc_auc_and_log_loss_binary():
    r"""Tests probability-based metrics (ROC AUC and Log Loss)."""
    y_true = np.array([0, 0, 1, 1])
    # Rank scores: 0.1(1), 0.35(2), 0.4(3), 0.8(4)
    # Ranks for positives (y=1): 2 (0.35) and 4 (0.8) -> Sum = 6
    # AUC = (Sum - Np*(Np+1)/2) / (Np * Nn) = (6 - 2*3/2) / (2 * 2) = (6 - 3) / 4 = 0.75
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(roc_auc_score(y_true, scores), 0.75)

    # Log Loss example:
    # True [0, 1], Probs [0.1, 0.9]
    # Loss = -1/2 * [ log(1-0.1) + log(0.9) ] = -1/2 * [ log(0.9) + log(0.9) ] = -log(0.9)
    y = np.array([0, 1])
    probs = np.array([0.1, 0.9])
    ll = log_loss(y, probs)
    assert np.isclose(ll, -np.log(0.9))


def test_log_loss_multiclass_perfect_score():
    r"""Tests log loss for perfect multiclass prediction (should be 0)."""
    y_true = np.array([0, 1, 2])
    # 2D probabilities (one-hot encoding of true labels)
    probs = np.eye(3) 
    # Loss = -1/3 * [log(1) + log(1) + log(1)] = 0
    assert np.isclose(log_loss(y_true, probs), 0.0)


def test_binary_metric_errors():
    r"""Tests expected ValueError/TypeError in classification metrics."""
    # 1. binary average with > 2 unique classes
    with pytest.raises(ValueError, match="binary average requires exactly two classes"):
        precision_score([0, 1, 2], [0, 1, 2], average="binary")
        
    # 2. ROC AUC requires samples from both classes
    with pytest.raises(ValueError, match="at least one sample from each class"):
        roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])
        
    # 3. Invalid probabilities: out of range (> 1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        log_loss([0, 1], np.array([1.2, 0.5]))


# -------------------- Classification: Multiclass Metrics --------------------

def test_multiclass_macro_micro():
    r"""Tests macro and micro averaging for precision, recall, F1."""
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 1])

    # Per-class Stats (TP/FP/FN):
    # Class 0 (Support=1): TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    # Class 1 (Support=1): TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0
    # Class 2 (Support=2): TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    # Acc = 2/4 = 0.5
    
    assert accuracy_score(y_true, y_pred) == 0.5
    
    # Macro Average = Mean of (1.0, 0.0, 0.5) = 1.5 / 3 = 0.5
    assert np.isclose(precision_score(y_true, y_pred, average="macro"), 0.5)
    assert np.isclose(recall_score(y_true, y_pred, average="macro"), 0.5)
    assert np.isclose(f1_score(y_true, y_pred, average="macro"), 0.5)

    # Micro Average = Accuracy in single-label multiclass = 0.5
    assert precision_score(y_true, y_pred, average="micro") == 0.5
    assert recall_score(y_true, y_pred, average="micro") == 0.5
    assert f1_score(y_true, y_pred, average="micro") == 0.5

    cm = confusion_matrix(y_true, y_pred)
    # Expected CM: [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    assert cm.shape == (3, 3)
    assert cm.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]


def test_confusion_with_custom_labels_ignores_unknown():
    r"""Tests confusion matrix behavior when labels are explicitly passed."""
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 3, 2, 1])
    
    # Custom labels only include [0, 1, 2].
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Expected CM:
    # [[1, 0, 0],
    #  [0, 0, 0],
    #  [0, 1, 1]]
    assert cm.tolist() == [[1, 0, 0],
                           [0, 0, 0],
                           [0, 1, 1]]


# ---- Regression Metrics ----

def test_regression_metrics():
    r"""Tests MSE, RMSE, MAE, and R^2 for a standard regression case."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    # MSE = 0.375
    assert mse(y_true, y_pred) == 0.375
    
    # RMSE = sqrt(0.375)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.375))
    
    # MAE = 0.5
    assert mae(y_true, y_pred) == 0.5
    
    # R2_score: SS_res = 1.5, SS_tot = 29.0
    assert np.isclose(r2_score(y_true, y_pred), 1.0 - 1.5 / 29.0)


def test_regression_shape_type_errors():
    r"""Tests expected ValueError/TypeError in regression metrics."""
    # 1. Shape mismatch
    with pytest.raises(ValueError, match="same length"):
        mse([1, 2], [1])
        
    # 2. Non-numeric inputs
    with pytest.raises(TypeError, match="numeric"):
        mae(["a", "b"], [1, 2])
        
    # 3. Constant y_true edge case for R2
    # R^2 is undefined when y_true is constant unless predictions are perfect
    with pytest.raises(ValueError, match="is undefined when y_true is constant"):
        r2_score([1, 1, 1], [1, 2, 3])
        
    # R^2 = 1.0 when y_true is constant and predictions are perfect
    assert r2_score([1, 1, 1], [1, 1, 1]) == 1.0