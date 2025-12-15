import numpy as np
import warnings
from typing import Optional, Union, Sequence, List, Callable, Tuple, Literal 
from rice_ml.post_processing import log_loss, roc_auc_score

class LogisticRegression:
    """
    Binary Logistic Regression classifier using Batch Gradient Descent.

    Model Capabilities
    ------------------
    - Binary classification (labels {0,1})
    - L2 regularization (Ridge) controlled by 'C' (inverse of alpha)
    - Optional intercept term
    - Gradient descent optimizer with convergence checks
    
    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength (lambda). Must be positive.
    eta : float, default=0.01
        Learning rate for Gradient Descent.
    epochs : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence check (stops if cost change < tol).
    fit_intercept : bool, default=True
        Specifies whether to include the bias/intercept term.
    random_state : Optional[int], default=None
        Seed for weight and bias initialization.
        
    Attributes
    ----------
    w_ : 1d-array
        Optimized weight vector (coefficients) after fitting.
    b_ : float
        Optimized bias term (intercept) after fitting (0.0 if fit_intercept=False).
    cost_history_ : List[float]
        Binary Cross-Entropy loss calculated after each epoch.
    """
    def __init__(self, C: float = 1.0, eta: float = 0.01, epochs: int = 1000, 
                 tol: float = 1e-4, fit_intercept: bool = True, random_state: Optional[int] = None):
        
        if C <= 0:
            raise ValueError("C (Inverse of regularization strength) must be positive.")
            
        self.C = C
        self.eta = eta
        self.epochs = epochs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.cost_history_: List[float] = []
        self.n_iter_: int = 0
        self.classes_: Optional[np.ndarray] = None
        
        self._lambda = 1.0 / C

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + exp(-Z))."""
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    def decision_function(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Calculates the linear combination (net input): z = X * w + b.
        
        Returns
        -------
        z : np.ndarray, shape (n_samples,)
            The decision function value.
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted.")
            
        X_arr = np.atleast_2d(X)
        if X_arr.shape[1] != self.w_.shape[0]:
            raise ValueError("Input features mismatch fitted model.")
            
        return X_arr @ self.w_ + self.b_

    def _cost(self, y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Binary Cross-Entropy Loss with L2 Regularization.
        """
        n_samples = y_true.shape[0]
        
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        log_loss_term = - (y_true @ np.log(y_prob_clipped) + (1 - y_true) @ np.log(1 - y_prob_clipped)) / n_samples
        
        # L2 Regularization Term (excludes bias 'b') 
        if self._lambda > 0 and self.w_ is not None:
            l2_term = (self._lambda / 2.0) * np.sum(self.w_**2)
        else:
            l2_term = 0.0
            
        return float(log_loss_term + l2_term)

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "LogisticRegression":
        """
        Trains the Logistic Regression model using Batch Gradient Descent.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).flatten()
        
        if X_arr.ndim != 2: raise ValueError("X must be a 2D array.")
        if np.any((y_arr != 0) & (y_arr != 1)):
             raise ValueError("Target y must be binary and encoded as {0, 1}.")
            
        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X_arr.shape
            
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Initialization
        self.w_ = np.random.randn(n_features) * 0.01
        self.b_ = np.random.randn() * 0.01 if self.fit_intercept else 0.0
        
        self.cost_history_ = []
        
        # 2. Gradient Descent Loop
        for i in range(self.epochs):
            self.n_iter_ = i + 1
            
            # Forward Pass
            z = self.decision_function(X_arr)
            y_prob = self._sigmoid(z)
            
            # Calculate Error Signal
            errors = y_prob - y_arr 
            
            # Calculate Gradients
            dw = (1 / n_samples) * (X_arr.T @ errors)
            db = (1 / n_samples) * np.sum(errors)
            
            # Apply Regularization (penalty term on dw)
            if self._lambda > 0:
                dw += self._lambda * self.w_
            
            # Update Weights and Bias
            self.w_ -= self.eta * dw
            if self.fit_intercept:
                self.b_ -= self.eta * db
            
            # Calculate and Store Cost
            current_cost = self._cost(y_prob, y_arr)
            self.cost_history_.append(current_cost)
            
            # Check for Convergence (Early Stopping)
            if i > 0 and self.tol is not None:
                if abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tol:
                    break
            
        return self

    def predict_proba(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Predicts the probability of belonging to class 1.
        """
        z = self.decision_function(X)
        y_prob = self._sigmoid(z)
        return y_prob

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Predicts the class label (0 or 1) using a threshold of 0.5.
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)

    def score(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence], 
              scoring: Literal['roc_auc', 'log_loss', 'accuracy'] = 'roc_auc') -> float:
        """
        Returns a selected evaluation score (default is ROC AUC).
        
        Parameters
        ----------
        scoring : {'roc_auc', 'log_loss', 'accuracy'}
            The metric to compute.
            
        Returns
        -------
        float
            The calculated score.
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
            
        y_true = np.asarray(y).flatten()
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        
        if scoring == 'roc_auc':
            # ROC AUC requires probabilities
            return float(roc_auc_score(y_true, y_prob))
        elif scoring == 'log_loss':
            # Log Loss requires probabilities
            return float(log_loss(y_true, y_prob))
        elif scoring == 'accuracy':
            # Accuracy requires hard predictions
            return float(np.mean(y_true == y_pred))
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")