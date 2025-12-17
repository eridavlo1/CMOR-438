import numpy as np
import pytest
from rice_ml.supervised_learning.k_nearest_neighbors import KNNClassifier, KNNRegressor

# ---- Classifier Tests -----

def test_classifier_basic_predict_and_proba_uniform_euclidean():
    """Test standard prediction, probability output format, and classification logic."""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,1,1])
    clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)

    # Query 1: (0.1, 0.1) -> Neighbors are (0,0), (0,1), (1,0) -> Labels {0, 0, 1} -> Predict 0
    # Query 2: (0.9, 0.9) -> Neighbors are (1,1), (0,1), (1,0) -> Labels {1, 1, 0} -> Predict 1
    preds = clf.predict([[0.1, 0.1], [0.9, 0.9]])
    assert preds.tolist() == [0, 1]

    proba = clf.predict_proba([[0.1, 0.1], [0.9, 0.9]])
    # 1. Proba rows must sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0)
    # 2. Predicted class must match the highest probability index
    assert (proba.argmax(axis=1) == preds).all()
    # 3. Check specific probabilities for (0.1, 0.1): 2/3 Class 0, 1/3 Class 1
    assert np.allclose(proba[0], [2/3, 1/3])


def test_classifier_manhattan_distance_weighted():
    """Test distance-weighted voting with Manhattan metric."""
    X = np.array([[0,0],[2,0],[0,2],[2,2]], dtype=float)
    y = np.array(["A","A","B","B"], dtype=object)
    
    clf = KNNClassifier(n_neighbors=3, metric="manhattan", weights="distance").fit(X, y)
    pred = clf.predict([[0.1, 0.2]])
    assert pred.tolist() == ["A"]
    
    # Check probabilities based on inverse distance:
    p = clf.predict_proba([[0.1, 0.2]])[0]
    assert p[0] > p[1]  # classes_ sorted -> ["A","B"]


def test_classifier_errors_and_kneighbors():
    """Test runtime errors and the kneighbors method output shape."""
    X = np.array([[0,0],[1,1],[2,2]], dtype=float)
    y = np.array([0,1,1])
    clf = KNNClassifier(n_neighbors=2).fit(X, y)
    
    # 1. Wrong feature count
    with pytest.raises(ValueError, match="features, expected 2"):
        clf.predict([[0.0, 0.0, 0.0]])
        
    # 2. kneighbors returns shapes (nq, k)
    d, idx = clf.kneighbors([[1.0, 1.0]])
    assert d.shape == (1, 2) and idx.shape == (1, 2)
    
    # 3. k value validation
    assert np.allclose(d[0][0], 0.0)
    assert idx[0][0] == 1
    
    # 4. KNeighbors before fit
    clf_unfitted = KNNClassifier(n_neighbors=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf_unfitted.kneighbors([[0, 0]])


def test_classifier_score_accuracy():
    """Test that the score method returns correct accuracy."""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,1,1])
    clf = KNNClassifier(n_neighbors=1).fit(X, y)
    assert clf.score(X, y) == 1.0
    
    # Test imperfect score
    clf2 = KNNClassifier(n_neighbors=4).fit(X, y) 
    assert clf2.score(X, y) == 0.5


# ---- Regressor Tests -----

def test_regressor_basic_predict_and_r2_score():
    """Test prediction for regression and R^2 score calculation."""
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
    
    # Test perturbed X (if model still predicts 5.0 accurately)
    # The R2 score should still be 1.0 if predictions match the constant y
    assert reg.score(X + 0.1, y) == 1.0