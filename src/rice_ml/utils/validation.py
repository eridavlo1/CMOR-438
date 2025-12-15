import numpy as np
from typing import Optional, Sequence, Union, Any, Tuple

# --- Type Definitions ---
ArrayLike = Union[np.ndarray, Sequence[Any]]


# --- Internal Safety Checks ---

def _check_for_nan_inf(arr: np.ndarray, name: str) -> None:
    """Check for NaN and Inf values and raise ValueError."""
    if np.isnan(arr).any():
        raise ValueError(f"Input array {name} contains NaN values. Please handle missing data.")
    if np.isinf(arr).any():
        raise ValueError(f"Input array {name} contains Infinite values. Please handle extreme data.")


# --- Primary Validation Functions ---

def ensure_2d_numeric(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Ensure X is a 2D numeric NumPy array (n_samples, n_features).
    
    This function performs necessary type conversion, checks dimensionality, 
    and verifies for NaN/Inf values.
    
    Parameters
    ----------
    X : array_like
        The input feature matrix or array.
    name : str, default="X"
        The name of the variable (used in error messages).

    Returns
    -------
    np.ndarray
        The validated and converted 2D float array.

    Raises
    ------
    ValueError
        If input is empty, contains NaN/Inf, or has incorrect dimensionality.
    TypeError
        If input contains non-numeric elements.
    """
    arr = np.asarray(X)
    
    # Handle 1D input by reshaping to (n, 1) for consistent feature matrix handling
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Convert to float, attempting to catch non-numeric elements
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        # If already numeric, ensure it's float for consistent calculations
        arr = arr.astype(float, copy=False)

    _check_for_nan_inf(arr, name)

    return arr


def ensure_1d_vector(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Ensure y is a 1D vector (numeric or categorical).
    
    This function is primarily used for target vectors (y) or stratification vectors. 
    It checks dimensionality and length but allows non-numeric dtypes (for labels).
    
    Parameters
    ----------
    y : array_like
        The input target vector.
    name : str, default="y"
        The name of the variable (used in error messages).

    Returns
    -------
    np.ndarray
        The validated 1D array.

    Raises
    ------
    ValueError
        If input is empty or not 1D.
    """
    arr = np.asarray(y)
    
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")
    
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
        
    # NOTE: No explicit numeric check or NaN/Inf check, as 'y' may contain non-numeric labels.
    
    return arr


def check_Xy_shapes(X: np.ndarray, y: np.ndarray) -> None:
    """
    Check that X and y have compatible leading dimensions (n_samples).
    """
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y must have compatible first dimension; got len(y)={len(y)} "
            f"and X.shape[0]={X.shape[0]}."
        )