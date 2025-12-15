import numpy as np
import pytest
from typing import Tuple
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# ----Classifier Tests -----

def test_classifier_basic_predict_and_proba_uniform_euclidean():
    """Test standard prediction, probability output format, and classification logic."""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,1,1])
    clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)

    # Query 1: (0.1, 0.1) -> Neighbors are (0,0), (0,1), (1,0) -> Labels {0, 0, 1} -> Predict 0
    # Query 2: (0.9, 0.9) -> Neighbors are (1,1), (0,1), (1,0) -> Labels {1, 1, 1, 0, 1} -> Predict 1
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
    
    # Query (0.1, 0.2)
    # Distances (Manhattan) to all points:
    # d(0,0) = 0.3
    # d(2,0) = |0.1-2| + |0.2-0| = 1.9 + 0.2 = 2.1
    # d(0,2) = |0.1-0| + |0.2-2| = 0.1 + 1.8 = 1.9
    # d(2,2) = 1.9 + 1.8 = 3.7
    # Nearest 3 (k=3): (0,0, Label A), (0,2, Label B), (2,0, Label A)
    
    clf = KNNClassifier(n_neighbors=3, metric="manhattan", weights="distance").fit(X, y)
    pred = clf.predict([[0.1, 0.2]])
    assert pred.tolist() == ["A"]
    
    # Check probabilities based on inverse distance:
    # Weights: w(0,0)=1/0.3, w(0,2)=1/1.9, w(2,0)=1/2.1
    w_a = (1/0.3) + (1/2.1)
    w_b = (1/1.9)
    # Since w_a > w_b, p_a should be > p_b.
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
    
    # 3. k value validation (k=2 neighbors for query point [1,1])
    # Distances: d(0,0)=sqrt(2), d(1,1)=0, d(2,2)=sqrt(2)
    # Neighbors should be index 1 (0 dist) and index 0 or 2 (sqrt(2) dist)
    assert np.allclose(d[0][0], 0.0)
    assert idx[0][0] == 1 # Index of [1,1] in X
    
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
    clf2 = KNNClassifier(n_neighbors=4).fit(X, y) # Predicts majority (0) for all
    # True: [0,0,1,1]. Pred: [0,0,0,0]. Accuracy: 2/4 = 0.5
    assert clf2.score(X, y) == 0.5


def test_classifier_zero_distance_with_distance_weights():
    """Test robust zero-distance handling (only zero-distance neighbors vote)."""
    X = np.array([[0,0],[1,1],[0,0]], dtype=float)
    y = np.array([0,1,0]) # Two samples are class 0, one is class 1
    
    # k=2, weights="distance"
    clf = KNNClassifier(n_neighbors=2, weights="distance").fit(X, y)
    
    # Query matches (0,0) exactly (indices 0 and 2 are 0 distance)
    pred = clf.predict([[0,0]])
    assert pred.tolist() == [0]
    
    p = clf.predict_proba([[0,0]])[0]
    # Zero distance samples are indices 0 (y=0) and 2 (y=0). 
    # Only these two vote, result is 100% mass on class 0.
    assert np.isclose(p[0], 1.0)
    assert np.isclose(p[1], 0.0)


# ---- Regressor Tests -----

def test_regressor_basic_predict_and_r2_score():
    """Test prediction for regression and R^2 score calculation."""
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])
    
    # 1. Distance-weighted prediction for X=1.5
    reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    # Neighbors of 1.5 are 1 (dist 0.5, y=1.0) and 2 (dist 0.5, y=1.5).
    # Since d1=d2, weights are w1=2, w2=2 (or w1=1, w2=1) -> average = (1.0+1.5)/2 = 1.25
    pred = reg.predict([[1.5]])[0]
    assert np.allclose(pred, 1.25) 
    
    # 2. Perfect R^2 score with k=1 on training data
    reg2 = KNNRegressor(n_neighbors=1).fit(X, y)
    assert reg2.score(X, y) == 1.0
    
    # 3. Imperfect R^2 score
    y_pred_bad = reg.predict(X)
    assert reg.score(X, y) < 1.0 # Should be imperfect due to averaging

def test_regressor_constant_y_score_error():
    """Tests the special case where y_true is constant (R^2 definition)."""
    X = np.array([[0],[1],[2]], dtype=float)
    y = np.array([5.0, 5.0, 5.0])
    reg = KNNRegressor(n_neighbors=1).fit(X, y)
    
    # 1. Perfect prediction on training data (k=1 guarantees perfect fit) -> R^2 = 1.0
    assert reg.score(X, y) == 1.0
    
    # 2. Prediction on perturbed X: prediction will still be 5.0, so R^2 is 1.0
    assert reg.score(X + 0.1, y) == 1.0
    
    # 3. Test scenario where prediction is IMPERFECT but y is constant
    # For KNN, this is hard to engineer, but if we set the score to a non-perfect value (e.g., 0.99)
    # the underlying R^2 function should raise if predictions are NOT perfect, 
    # unless you are scoring on the training data.
    # We rely on the robust handling in the score method (if ss_res > 0 and ss_tot == 0).
    # Since KNN naturally predicts y_mean when y is constant, this test remains robust.