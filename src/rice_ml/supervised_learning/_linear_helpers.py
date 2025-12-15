import numpy as np
from typing import Union, Optional, Sequence, Any

# --- Activation Functions ---

def sigmoid(Z: np.ndarray, clip_range: float = 500.0) -> np.ndarray:
    """
    Computes the Sigmoid activation function: 1 / (1 + exp(-Z)).

    Used for probability estimation in the output layer of binary
    classification models. 
    Parameters
    ----------
    Z : np.ndarray
        The input matrix/vector (weighted sum of inputs).
    clip_range : float, default=500.0
        A value to clip Z to prevent overflow/underflow errors in exp().

    Returns
    -------
    np.ndarray
        The activation output.
    """
    Z_clipped = np.clip(Z, -clip_range, clip_range)
    return 1.0 / (1.0 + np.exp(-Z_clipped))


def sigmoid_derivative(A: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Sigmoid function: A * (1 - A).

    Parameters
    ----------
    A : np.ndarray
        The output of the sigmoid function (sigmoid(Z)).

    Returns
    -------
    np.ndarray
        The derivative value.
    """
    return A * (1.0 - A)


def relu(Z: np.ndarray) -> np.ndarray:
    """
    Computes the Rectified Linear Unit (ReLU) activation: max(0, Z).     
    Parameters
    ----------
    Z : np.ndarray
        The input matrix/vector.

    Returns
    -------
    np.ndarray
        The ReLU activation output.
    """
    return np.maximum(0, Z)


def relu_derivative(A: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the ReLU function: 1 if A > 0, else 0.
    
    Parameters
    ----------
    A : np.ndarray
        The output of the ReLU function (relu(Z)).

    Returns
    -------
    np.ndarray
        The derivative value.
    """
    return (A > 0).astype(float)


# --- Utility Functions for Linear Models ---

def add_bias_unit(X: np.ndarray, how: str = 'col') -> np.ndarray:
    """
    Adds a bias unit (a column of 1s) to the input matrix X.

    This is necessary for linear models (like Linear Regression, Perceptron, 
    and MLP) to introduce an intercept term.

    Parameters
    ----------
    X : np.ndarray
        The input matrix (n_samples, n_features).
    how : {'col', 'row'}, default='col'
        Orientation of addition. 'col' adds a column of ones (most common).

    Returns
    -------
    np.ndarray
        The matrix with the added bias unit.
    """
    if how == 'col':
        return np.hstack((X, np.ones((X.shape[0], 1))))
    elif how == 'row':
        return np.vstack((X, np.ones((1, X.shape[1]))))
    raise ValueError("Bias unit addition method must be 'col' or 'row'.")


# --- Public API  ---

__all__ = [
    'sigmoid', 
    'sigmoid_derivative', 
    'relu', 
    'relu_derivative', 
    'add_bias_unit'
]