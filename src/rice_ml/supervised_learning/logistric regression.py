import numpy as np
from typing import Optional, Union, Sequence, Any, Tuple, List
import warnings
from rice_ml.utils import ArrayLike, ensure_2d_numeric, ensure_1d_vector, check_Xy_shapes
from ._linear_helpers import sigmoid, add_bias_unit 

class LogisticRegression:
    """
    Logistic Regression Classifier using Batch Gradient Descent.

    Performs binary classification by estimating probabilities using the 
    sigmoid function. 

    Parameters
    ----------
    eta : float, default=0.01
        The learning rate (step size) for gradient descent.
    epochs : int, default=100
        Maximum number of training iterations.
    random_state : Optional[int], default=None
        Seed for reproducibility.

    Attributes
    ----------
    weights_ : np.ndarray
        Weights (including bias term) learned by the model.
    cost_history_ : List[float]
        Binary Cross-Entropy loss calculated at each epoch.
    """
    def __init__(self, eta: float = 0.01, epochs: int = 100, random_state: Optional[int] = None):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.weights_: Optional[np.ndarray] = None
        self.cost_history_: List[float] = []

    # NOTE: sigmoid is now imported from _linear_helpers

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the net input (Z = X * W + b).
        Uses the shared helper `add_bias_unit`.
        """
        X_biased = add_bias_unit(X, how='col')
        # Z = X_biased @ self.weights_ (already includes bias)
        return X_biased @ self.weights_

    def _cost(self, y: np.ndarray, y_pred_proba: np.ndarray, eps: float = 1e-15) -> float:
        """Calculates the Binary Cross-Entropy Loss (Log Loss)."""
        n_samples = y.shape[0]
        # Clip probabilities to prevent log(0)
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps) 
        
        # Loss = -1/N * sum(y*log(p) + (1-y)*log(1-p))
        loss = - np.sum(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
        return float(loss / n_samples)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Trains the model using batch gradient descent.
        """
        X_arr = ensure_2d_numeric(X, name="X")
        y_arr = ensure_1d_vector(y, name="y").astype(float)
        check_Xy_shapes(X_arr, y_arr)

        if len(np.unique(y_arr)) != 2 or np.any((y_arr != 0) & (y_arr != 1)):
             raise ValueError("Target y must be binary and encoded as {0, 1}.")

        n_samples, n_features = X_arr.shape
        rng = np.random.default_rng(self.random_state)
        
        # Initialize weights (w) and bias (b) as zeros or small random values
        # We initialize (n_features + 1) weights for w and b
        self.weights_ = rng.standard_normal(size=n_features + 1) * 0.01 
        self.cost_history_ = []

        X_biased = add_bias_unit(X_arr, how='col')
        
        for epoch in range(self.epochs):
            # Forward pass
            net_input = X_biased @ self.weights_
            y_pred_proba = sigmoid(net_input)

            # Gradient calculation
            # Error = (y_pred_proba - y_true)
            error = y_pred_proba - y_arr
            
            # Gradient = X_T * Error / N
            gradient = X_biased.T @ error / n_samples
            
            # Update weights: W = W - eta * Gradient
            self.weights_ -= self.eta * gradient
            
            # Calculate and store cost
            cost = self._cost(y_arr, y_pred_proba)
            self.cost_history_.append(cost)

            
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts class membership probabilities (P(y=1|X)).
        """
        X_arr = ensure_2d_numeric(X)
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
            
        # P(y=1|X) = sigmoid(Z)
        return sigmoid(self._net_input(X_arr))

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, 0)