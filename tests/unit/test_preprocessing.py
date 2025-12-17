import numpy as np
import pytest
from collections import Counter
from rice_ml.processing import (
    standardize,
    minmax_scale,
    maxabs_scale,
    l2_normalize_rows,
    train_test_split,
    train_val_test_split,
)

# ----- Scaling & Normalization Tests -----

def test_standardize_basic_and_params():
    X = np.array([[1., 2.], [3., 2.], [5., 2.]])
    Z, params = standardize(X, return_params=True)
    assert Z.shape == X.shape
    assert np.allclose(Z[:, 1], 0.0)
    assert params["scale"][1] == 1.0
    assert np.allclose(Z.mean(axis=0), 0.0)

def test_minmax_scale_range_and_params():
    X = np.array([[0., 10.], [5., 10.], [10., 10.]])
    X2, params = minmax_scale(X, feature_range=(2, 3), return_params=True)
    assert np.allclose(X2[:, 0], [2.0, 2.5, 3.0])
    assert np.allclose(X2[:, 1], 2.0)

# ---- Splitting Tests ----

def test_train_test_split_shapes_and_determinism():
    """Tests shapes and ensures random_state leads to deterministic results."""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    # Test Determinism: Same seed must produce identical splits
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 50 samples * 0.3 = 15 test, 35 train
    assert X_tr1.shape == (35, 2)
    assert X_te1.shape == (15, 2)
    
    # Arrays must be identical across runs with the same seed
    assert np.array_equal(X_tr1, X_tr2)
    assert np.array_equal(y_te1, y_te2)
    
    # Test shuffle=False: Must be sequential
    X_tr3, X_te3 = train_test_split(X, test_size=0.2, shuffle=False)
    assert np.array_equal(X_tr3, X[:40])

def test_train_test_split_stratify():
    X = np.arange(60).reshape(30, 2)
    y = np.array([0, 1, 2] * 10)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    
    c_te = Counter(y_te)
    assert c_te[0] == 3
    assert c_te[1] == 3
    assert c_te[2] == 3