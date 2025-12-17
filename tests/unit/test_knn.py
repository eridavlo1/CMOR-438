import numpy as np
import pytest
from rice_ml.supervised_learning.k_nearest_neighbors import KNNClassifier, KNNRegressor

# ... (Keep Classifier Tests as they were) ...

# ---- Regressor Tests -----

def test_regressor_basic_predict_and_r2_score():
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])
    
    reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    pred = reg.predict([[1.5]])[0]
    assert np.allclose(pred, 1.25) 
    
    reg2 = KNNRegressor(n_neighbors=1).fit(X, y)
    assert reg2.score(X, y) == 1.0

def test_regressor_constant_y_score_error():
    """Tests handling of constant y_true (R^2 definition)."""
    X = np.array([[0],[1],[2]], dtype=float)
    y = np.array([5.0, 5.0, 5.0])
    reg = KNNRegressor(n_neighbors=1).fit(X, y)
    
    # If the model predicts perfectly (5.0), R^2 should be 1.0
    # even if y_true is constant.
    assert reg.score(X, y) == 1.0
    
    # Test perturbed X (still predicting 5.0)
    assert reg.score(X + 0.1, y) == 1.0