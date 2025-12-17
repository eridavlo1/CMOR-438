import numpy as np
from typing import Optional, Union, Sequence, List, Literal, Callable
from ..processing.post_processing import r2_score
import warnings


class LinearRegression:
    r"""
    A comprehensive Linear Regression model supporting OLS, Ridge, and Gradient Descent.

    Parameters
    ----------
    method : {'ols', 'ridge', 'gd'}, default='ols'
        The optimization method to use.
    alpha : float, default=0.0
        Regularization strength (lambda) for Ridge regression. Ignored if method='ols' or 'gd'.
    eta : float, default=0.01
        Learning rate for Gradient Descent (if method='gd').
    epochs : int, default=1000
        Maximum iterations for Gradient Descent (if method='gd').
    random_state : int, optional
        Seed for reproducibility.
    """
    def __init__(self, method: Literal['ols', 'ridge', 'gd'] = 'ols', 
                 alpha: float = 0.0, eta: float = 0.01, epochs: int = 1000, 
                 random_state: Optional[int] = None):
        
        self.method = method
        self.alpha = alpha
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        self.w_: Optional[np.ndarray] = None  # Weights (coefficients)
        self.b_: Optional[float] = None       # Bias (intercept)
        self.cost_history_: List[float] = []  # Cost history for GD

    def _add_intercept_column(self, X: np.ndarray) -> np.ndarray:
        r"""Adds a column of ones to X for intercept calculation."""
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def _fit_closed_form(self, X_biased: np.ndarray, y: np.ndarray):
        r"""Fits using the analytical solution (OLS or Ridge)."""
        n_features_biased = X_biased.shape[1]
        
        # 1. Calculate X_biased.T @ X_biased
        XTX = X_biased.T @ X_biased
        
        # 2. Add regularization for Ridge
        if self.method == 'ridge':
            # Identity matrix (I) for Ridge: size is (n_features+1, n_features+1)
            # We DON'T regularize the intercept term (the last column of W), 
            # so the last element of the diagonal is 0.
            I = np.eye(n_features_biased)
            I[-1, -1] = 0.0 
            XTX += self.alpha * I

        # 3. Calculate (XTX + alpha*I)^-1
        try:
            XTX_inv = np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
            # Handle case where XTX is singular (e.g., collinearity, few samples)
            warnings.warn("XTX matrix is singular. Using pseudo-inverse (least squares).", RuntimeWarning)
            XTX_inv = np.linalg.pinv(XTX) 

        # 4. Calculate final weights W = (XTX + alpha*I)^-1 @ X.T @ y
        W_full = XTX_inv @ X_biased.T @ y
        
        # 5. Separate weights and bias
        self.w_ = W_full[:-1]
        self.b_ = W_full[-1]

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        r"""Fits using Batch Gradient Descent (reusing prior GD logic)."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize weights (w) and bias (b) separately
        self.w_ = np.random.randn(n_features) * 0.01
        self.b_ = np.random.randn() * 0.01
        self.cost_history_ = []

        for _ in range(self.epochs):
            # Calculate hypothesis
            y_pred = X @ self.w_ + self.b_
            errors = y_pred - y
            
            # Calculate Gradients
            dw = (1 / n_samples) * X.T @ errors
            db = (1 / n_samples) * np.sum(errors)
            
            # Update Weights and Bias
            self.w_ -= self.eta * dw
            self.b_ -= self.eta * db
            
            # Calculate and Store Cost (MSE scaled by 1/2)
            cost = (1 / (2 * n_samples)) * np.sum(errors**2)
            self.cost_history_.append(cost)

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "LinearRegression":
        r"""
        Trains the Linear Regression model using the specified method.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).flatten()
        
        if X_arr.ndim != 2: raise ValueError("X must be a 2D array.")
        if y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("y must be a 1D array matching the number of rows in X.")

        if self.method in ['ols', 'ridge']:
            X_biased = self._add_intercept_column(X_arr)
            self._fit_closed_form(X_biased, y_arr)
        elif self.method == 'gd':
            self._fit_gradient_descent(X_arr, y_arr)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Must be 'ols', 'ridge', or 'gd'.")
            
        return self

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        r"""Predicts continuous target values."""
        if self.w_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
            
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
            
        # Prediction: X @ w + b
        return X_arr @ self.w_ + self.b_

    def score(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> float:
        r"""Returns the coefficient of determination (R^2) of the prediction."""
        if self.w_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
            
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=float).flatten()
        
        return float(r2_score(y_true, y_pred))
