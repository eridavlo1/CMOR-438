import numpy as np
import pytest
from collections import Counter
from typing import Tuple, List

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
    """Tests z-score calculation, near-zero mean, and zero-variance handling."""
    X = np.array([[1., 2.], [3., 2.], [5., 2.]])
    Z, params = standardize(X, return_params=True)
    
    assert Z.shape == X.shape
    
    # Column 2 is constant (mean=2.0, std=0.0). Z should be all 0.0.
    assert np.allclose(Z[:, 1], 0.0)
    # The scale param for the zero-variance column must be 1.0
    assert params["scale"][1] == 1.0
    
    # Final array should be centered (mean ~0)
    assert np.allclose(Z.mean(axis=0), 0.0)


def test_standardize_no_std_or_mean():
    """Tests the with_mean=False and with_std=False parameters."""
    X = np.array([[1., 2.], [3., 4.]])
    
    # 1. No changes applied
    Z = standardize(X, with_mean=False, with_std=False)
    assert np.allclose(Z, X)
    
    # 2. Only centering applied (mean must be 0)
    Z = standardize(X, with_mean=True, with_std=False)
    assert not np.allclose(Z, X)
    assert np.allclose(Z.mean(axis=0), 0.0)


def test_minmax_scale_range_and_params():
    """Tests scaling to a non-default range and zero-range feature handling."""
    X = np.array([[0., 10.], [5., 10.], [10., 10.]])
    X2, params = minmax_scale(X, feature_range=(2, 3), return_params=True)
    
    assert X2.shape == X.shape
    
    # 1. First feature: [0, 10] -> [2.0, 3.0]. 5.0 -> 2.5
    assert np.allclose(X2[:, 0], [2.0, 2.5, 3.0])
    
    # 2. Second feature: zero-range -> mapped to lower bound (2.0)
    assert np.allclose(X2[:, 1], 2.0)
    assert params["scale"][1] == 1.0 # Scale is 1.0 where range=0
    assert params["feature_range"] == (2.0, 3.0)


def test_maxabs_scale_basic():
    """Tests scaling by max absolute value."""
    X = np.array([[-2., 0.], [1., 0.], [2., 0.]])
    X2, params = maxabs_scale(X, return_params=True)
    
    # Feature 1 max abs is 2.0. Result: [-1.0, 0.5, 1.0]
    assert np.allclose(X2[:, 0], [-1.0, 0.5, 1.0])
    
    # Feature 2 max abs is 0.0. Scale is 1.0. Result: [0.0, 0.0, 0.0]
    assert np.allclose(X2[:, 1], [0.0, 0.0, 0.0])
    assert params["scale"][1] == 1.0


def test_l2_normalize_rows_behavior():
    """Tests L2 normalization (Euclidean norm) and edge cases."""
    X = np.array([[3., 4.], [0., 0.]])
    Xn = l2_normalize_rows(X)
    
    # Row 1: norm(3, 4) = 5. Result: [0.6, 0.8]. Norm should be 1.0
    assert np.isclose(np.linalg.norm(Xn[0]), 1.0)
    
    # Row 2: zero row stays zero
    assert np.allclose(Xn[1], [0.0, 0.0])
    
    # Test eps validation
    with pytest.raises(ValueError, match="eps must be > 0"):
        l2_normalize_rows(X, eps=0.0)


def test_scalers_input_validation():
    """Tests shared validation logic (2D, numeric, range checking)."""
    
    # Not 2D input
    with pytest.raises(ValueError, match="2D array"):
        standardize(np.array([1., 2., 3.]))
        
    # Non-numeric input
    with pytest.raises(TypeError, match="numeric"):
        standardize([["a", "b"], ["c", "d"]])
        
    # Empty input
    with pytest.raises(ValueError, match="non-empty"):
        minmax_scale(np.empty((0, 2)))
        
    # Invalid minmax feature range (min >= max)
    with pytest.raises(ValueError, match="min < max"):
        minmax_scale(np.ones((2, 2)), feature_range=(1, 1))
        
    # Invalid eps
    with pytest.raises(ValueError, match="eps must be > 0"):
        l2_normalize_rows(np.ones((2, 2)), eps=-1.0)


# ---- Splitting Tests ----

def test_train_test_split_shapes_and_determinism():
    """Tests shapes and ensures random_state leads to deterministic results."""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    # Test Determinism: Same seed must produce identical splits
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Expected sizes: 50 * 0.3 = 15 test, 35 train
    assert X_tr1.shape == (35, 2)
    assert X_te1.shape == (15, 2)
    
    # Arrays must be identical across runs with the same seed
    assert np.array_equal(X_tr1, X_tr2)
    assert np.array_equal(y_te1, y_te2)
    
    # Test shuffle=False (split occurs strictly sequentially)
    X_tr3, X_te3 = train_test_split(X, test_size=0.2, shuffle=False)
    # Expected: First 40 for train, next 10 for test
    assert X_tr3.shape == (40, 2) and X_te3.shape == (10, 2)
    assert np.array_equal(X_tr3, X[:40])


def test_train_test_split_stratify():
    """Tests that class proportions are maintained in the split."""
    X = np.arange(60).reshape(30, 2)
    y = np.array([0, 1, 2] * 10) # 10 samples per class
    
    # Expected sizes: 30 * 0.3 = 9 test, 21 train
    # Expected per class: 10 * 0.3 = 3 test, 7 train
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    
    # Must contain all classes
    assert set(np.unique(y_tr)) == {0, 1, 2}
    
    c_te = Counter(y_te)
    
    # Check that test set has 3 samples for each class
    assert c_te[0] == 3
    assert c_te[1] == 3
    assert c_te[2] == 3
    assert len(y_te) == 9 # Total test size


def test_train_val_test_split_shapes_and_stratify():
    """Tests 3-way split shapes and stratification."""
    X = np.arange(90).reshape(45, 2)
    y = np.array([0, 1, 2] * 15) # 15 samples per class
    
    # Total samples: 45
    # Test size: 0.2 * 45 = 9 samples
    # Val size: 0.2 * 45 = 9 samples
    # Train size: 45 - 9 - 9 = 27 samples
    
    parts = train_val_test_split(X, y, val_size=0.2, test_size=0.2, stratify=y, random_state=123)
    X_tr, X_va, X_te, y_tr, y_va, y_te = parts
    
    assert X_tr.shape == (27, 2)
    assert X_va.shape == (9, 2)
    assert X_te.shape == (9, 2)
    
    # Check stratification sanity: all classes must be present
    assert set(np.unique(y_tr)) == {0, 1, 2}
    assert set(np.unique(y_va)) == {0, 1, 2}
    assert set(np.unique(y_te)) == {0, 1, 2}


def test_split_input_validation():
    """Tests validation for sizes and shapes."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    
    # 1. Not 2D X
    with pytest.raises(ValueError, match="2D array"):
        train_test_split(np.arange(10), y)
        
    # 2. Shape mismatch (X rows vs y elements)
    with pytest.raises(ValueError, match="compatible first dimension"):
        train_test_split(X, y[:-1])
        
    # 3. Invalid test_size
    with pytest.raises(ValueError, match="float in \\(0, 1\\)"):
        train_test_split(X, y, test_size=1.5)
        
    # 4. Invalid random_state
    with pytest.raises(TypeError, match="integer or None"):
        train_test_split(X, y, random_state="seed")
        
    # 5. Invalid val_size + test_size >= 1.0
    with pytest.raises(ValueError, match="val_size \\+ test_size must be < 1.0"):
        train_val_test_split(X, y, val_size=0.6, test_size=0.5)


def test_train_val_test_split_without_y():
    """Tests 3-way split when only X is provided."""
    X = np.arange(30).reshape(15, 2)
    
    # Returns (X_train, X_val, X_test)
    parts = train_val_test_split(X, val_size=0.2, test_size=0.2, random_state=7)
    X_tr, X_va, X_te = parts
    
    # Total samples: 15. Test: 0.2*15=3. Val: 0.2*15=3. Train: 9.
    assert X_tr.shape == (9, 2)
    assert X_va.shape == (3, 2)
    assert X_te.shape == (3, 2)