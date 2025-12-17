# CMOR-438/src/rice_ml/supervised_learning/gradient_descent.py

import numpy as np
from typing import Callable, Tuple, Optional, Sequence, Union, List
import warnings
from ..utils import ArrayLike, ensure_2d_numeric, ensure_1d_vector 

class GradientDescent:
    r"""
    This class implements the core logic for Batch Gradient Descent (BGD) 
    optimization. It is designed to be model-agnostic, relying on external 
    cost and gradient functions defined within specific supervised learning 
    models (e.g., LinearRegression, LogisticRegression).
    
    """
    
    def __init__(self, eta: float = 0.01, epochs: int = 1000, tol: Optional[float] = None, random_state: Optional[int] = None):
        r"""
        Parameters
        ----------
        eta : float, default=0.01
            The learning rate (step size).
        epochs : int, default=1000
            The maximum number of training iterations.
        tol : float, optional
            Tolerance for early stopping. Training stops if the cost change 
            between epochs is less than `tol`.
        random_state : int, optional
            Seed for initializing weights and bias.
        """
        self.eta = eta
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state
        
        # Parameters initialized during the optimize method
        self.weights_ = None
        self.bias_ = None
        self.cost_history_ = []

    def optimize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cost_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float], float],
        gradient_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, float]],
    ) -> Tuple[np.ndarray, float, List[float]]:
        r"""
        Performs the optimization loop.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.
        y : array_like, shape (n_samples,)
            Target vector.
        cost_func : Callable
            Function that computes the cost (e.g., MSE) given (X, y, weights, bias).
        gradient_func : Callable
            Function that computes the gradients (dw, db) given (X, y, weights, bias).

        Returns
        -------
        weights : np.ndarray
            Optimized weight vector.
        bias : float
            Optimized bias term.
        cost_history : list
            List of cost values recorded after each epoch.
        """
        
        # --- 1. Validation and Data Preparation ---
        X_arr = ensure_2d_numeric(X)
        y_arr = ensure_1d_vector(y)
        n_samples, n_features = X_arr.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # 2. Initialize weights (w) and bias (b)
        self.weights_ = np.random.randn(n_features) * 0.01
        self.bias_ = np.random.randn() * 0.01
        self.cost_history_ = []
        
        # Calculate and store the initial cost (Epoch 0)
        self.cost_history_.append(cost_func(X_arr, y_arr, self.weights_, self.bias_)) 
        
        # --- 3. Start Gradient Descent Loop ---
        for _ in range(self.epochs):
            
            # 3a. Compute Gradients
            # The model-specific logic is encapsulated in gradient_func
            dw, db = gradient_func(X_arr, y_arr, self.weights_, self.bias_)
            
            # 3b. Update Weights and Bias 
            self.weights_ -= self.eta * dw
            self.bias_ -= self.eta * db
            
            # 3c. Calculate and Store Cost
            current_cost = cost_func(X_arr, y_arr, self.weights_, self.bias_)
            self.cost_history_.append(current_cost)
            
            # 3d. Check for Convergence (Early Stopping)
            # Check against the previous cost (cost_history[-2])
            if self.tol is not None and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tol:
                break
                
        return self.weights_, self.bias_, self.cost_history_