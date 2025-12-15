import numpy as np
from typing import Callable, Tuple, Optional, Any, Sequence, Union

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

def batch_gradient_descent(
    X: ArrayLike,
    y: ArrayLike,
    *,
    cost_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float], float],
    gradient_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, float]],
    eta: float = 0.01,
    epochs: int = 1000,
    random_state: Optional[int] = None,
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, float, list]:
    """
    Performs Batch Gradient Descent optimization to minimize a cost function.

    This function is model-agnostic and relies on external cost and gradient 
    functions tailored to a specific model (e.g., Linear Regression).

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
    eta : float, default=0.01
        Learning rate.
    epochs : int, default=1000
        Maximum number of iterations.
    random_state : int, optional
        Seed for initial weight/bias generation.
    tol : float, optional
        Tolerance for early stopping (if the cost change between epochs is < tol).

    Returns
    -------
    weights : np.ndarray
        Optimized weight vector (w).
    bias : float
        Optimized bias term (b).
    cost_history : list
        List of cost values recorded after each epoch.

    Examples
    --------
    # Requires defining _lr_cost_func and _lr_gradient_func outside
    >>> import numpy as np
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> y = np.array([3.0, 5.0, 7.0]) # y = 2x + 1
    >>> w_final, b_final, _ = batch_gradient_descent(X, y, cost_func=lambda *args: np.mean((args[0] @ args[2] + args[3] - args[1])**2), gradient_func=lambda *args: ((args[0] @ args[2] + args[3] - args[1]) @ args[0] / len(args[1]), np.mean(args[0] @ args[2] + args[3] - args[1])), eta=0.01, epochs=100)
    >>> round(w_final[0], 2)
    2.17
    >>> round(b_final, 2)
    0.72
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    if random_state is not None:
        np.random.seed(random_state)
        
    # 1. Initialize weights (w) and bias (b)
    # Weights and bias are initialized randomly, scaled down to prevent early divergence.
    weights = np.random.randn(n_features) * 0.01
    bias = np.random.randn() * 0.01
    
    cost_history = []
    
    # 2. Start Gradient Descent Loop
    for _ in range(epochs):
        
        # 2a. Compute Gradients
        # This function call encapsulates the model's core math (h(x) and its derivative)
        dw, db = gradient_func(X_arr, y_arr, weights, bias)
        
        # 2b. Update Weights and Bias 
        weights -= eta * dw
        bias -= eta * db
        
        # 2c. Calculate and Store Cost
        current_cost = cost_func(X_arr, y_arr, weights, bias)
        cost_history.append(current_cost)
        
        # 2d. Check for Convergence (Early Stopping)
        if tol is not None and len(cost_history) > 1:
            if abs(cost_history[-2] - cost_history[-1]) < tol:
                break
            
    return weights, bias, cost_history
