from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = [
    'KNNClassifier',
    'KNNRegressor',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helpers & Validation -----------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    # Standardize to float, ensuring numeric type.
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _ensure_1d(y, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D array (labels may be any dtype for classifier; numeric for regressor)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _validate_common_params(
    n_neighbors: int,
    metric: Literal["euclidean", "manhattan"],
    weights: Literal["uniform", "distance"],
) -> None:
    """Validates common KNN hyperparameters."""
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if metric not in ("euclidean", "manhattan"):
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")
    if weights not in ("uniform", "distance"):
        raise ValueError("weights must be 'uniform' or 'distance'.")


def _pairwise_distances(XA: np.ndarray, XB: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute pairwise distances between rows of XA and XB.
    """
    if metric == "euclidean":
        # Vectorized Euclidean distance calculation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        aa = np.sum(XA * XA, axis=1, keepdims=True)
        bb = np.sum(XB * XB, axis=1, keepdims=True).T
        # Clamp at 0 for numerical stability
        D2 = np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0)
        return np.sqrt(D2, dtype=float)
    elif metric == "manhattan":
        # Vectorized Manhattan distance calculation
        # Use broadcasting: (n_a, 1, d) - (1, n_b, d)
        diff = XA[:, None, :] - XB[None, :, :]
        return np.sum(np.abs(diff), axis=2, dtype=float)
    else:
        raise ValueError("Unsupported metric.")


def _neighbors(X_train: np.ndarray, X_query: np.ndarray, n_neighbors: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the n_neighbors nearest neighbors for each query sample.
    
    Returns (distances, indices) sorted by distance.
    """
    D = _pairwise_distances(X_query, X_train, metric)  # (nq, n_train)
    
    if n_neighbors > X_train.shape[0]:
        raise ValueError(f"n_neighbors={n_neighbors} cannot exceed number of training samples={X_train.shape[0]}.")
        
    # Get the indices that would sort the distances matrix row-wise
    idx_all_sorted = np.argsort(D, axis=1) # (nq, n_train)

    # Slice to keep only the top k neighbors
    idx_sorted = idx_all_sorted[:, :n_neighbors] # (nq, k)
    
    # Use these indices to get the corresponding distances
    d_sorted = np.take_along_axis(D, idx_sorted, axis=1) # (nq, k)
    
    return d_sorted, idx_sorted


def _weights_from_distances(dist: np.ndarray, scheme: str, eps: float = 1e-12) -> np.ndarray:
    """
    Compute neighbor weights from distances.
    """
    if scheme == "uniform":
        return np.ones_like(dist, dtype=float)

    # distance weighting
    zero_mask = (dist <= eps)
    w = np.empty_like(dist, dtype=float)
    
    any_zero_in_query = zero_mask.any(axis=1)

    # Case 1: Queries with distance 0 neighbor(s). Assign weight 1 only to those, 0 to others.
    if np.any(any_zero_in_query):
        w[any_zero_in_query] = zero_mask[any_zero_in_query].astype(float)
        
    # Case 2: Queries where all distances are non-zero. Use inverse distance.
    if np.any(~any_zero_in_query):
        w[~any_zero_in_query] = 1.0 / np.maximum(dist[~any_zero_in_query], eps)
        
    return w


# ---------------------------------- Base ----------------------------------

class _KNNBase:
    """Shared functionality for KNN models."""

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        _validate_common_params(n_neighbors, metric, weights)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    # ---------------- API ----------------

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the model."""
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(f"X and y length mismatch: len(y)={len(y_arr)} vs X.shape[0]={X_arr.shape[0]}")
        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
        
        self._X = X_arr
        self._y = y_arr
        return self

    def _check_is_fitted(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._X is None or self._y is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self._X, self._y

    def kneighbors(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Find the nearest neighbors of the provided samples."""
        X_train, _ = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")
        return _neighbors(X_train, Xq, self.n_neighbors, self.metric)


# -------------------------------- Classifier --------------------------------

class KNNClassifier(_KNNBase):
    """
    k-Nearest Neighbors classifier.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, metric=metric, weights=weights)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        super().fit(X, y)
        self.classes_ = np.unique(self._y)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities."""
        X_train, y_train = self._check_is_fitted()
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")
            
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.metric)
        w = _weights_from_distances(dist, self.weights) # (n_query, k)

        n_query = Xq.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_query, n_classes), dtype=float)
        
        # 1. Map neighbor labels to their integer class index
        class_indices = np.searchsorted(self.classes_, y_train[idx]) # (n_query, k)
        
        # 2. Sum weights (votes) for each class index using bincount per row
        for i in range(n_query):
            counts = np.bincount(class_indices[i], weights=w[i], minlength=n_classes)
            total_weight = counts.sum()
            
            if total_weight == 0:
                proba[i] = 1.0 / n_classes
            else:
                proba[i] = counts / total_weight
                
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the most probable class."""
        proba = self.predict_proba(X)
        best = np.argmax(proba, axis=1)
        return self.classes_[best]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Classification accuracy on (X, y)."""
        y_true = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        if len(y_true) != len(y_pred):
            raise ValueError("X and y lengths do not match.")
        return float(np.mean(y_true == y_pred))


# -------------------------------- Regressor --------------------------------

class KNNRegressor(_KNNBase):
    """
    k-Nearest Neighbors regressor.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNRegressor":
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")
        
        if not np.issubdtype(y_arr.dtype, np.number):
            try:
                y_arr = y_arr.astype(float, copy=False)
            except (TypeError, ValueError) as e:
                raise TypeError("Regression target values must be numeric.") from e
        
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(f"X and y length mismatch: len(y)={len(y_arr)} vs X.shape[0]={X_arr.shape[0]}")
        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
            
        self._X = X_arr
        self._y = y_arr.astype(float, copy=False)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict regression targets (weighted average of neighbors)."""
        X_train, y_train = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")
        
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.metric)
        w = _weights_from_distances(dist, self.weights) # (n_query, k)
        
        y_neighbors = y_train[idx]  # (nq, k)
        wsum = np.sum(w, axis=1)
        
        # Calculate weighted average
        with np.errstate(divide="ignore", invalid="ignore"):
            y_pred = np.divide(np.sum(w * y_neighbors, axis=1), wsum, where=wsum != 0)
        
        # Fallback: if total weight is 0 (e.g., all distances were exactly 0 and weights were restricted)
        fallback = (wsum == 0)
        if np.any(fallback):
            # Fallback to simple uniform mean of the neighbors
            y_pred[fallback] = np.mean(y_neighbors[fallback], axis=1)
            
        return y_pred.astype(float, copy=False)

    def score(self, X, y) -> float:
        """Compute the R^2 score."""
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        
        if ss_tot == 0:
            # FIX: Handle constant y_true. Return 1.0 if the fit is perfect.
            if ss_res < 1e-12:
                return 1.0
            raise ValueError(
                "R^2 is undefined when y_true is constant unless the fit is perfect."
            )
            
        return 1 - (ss_res / ss_tot)