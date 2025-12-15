"""
Distance metrics module.

This module provides common distance functions for numerical vectors,
implemented using NumPy for efficient computation and robust error handling.
"""

from __future__ import annotations
from typing import Tuple, Any, Sequence, Union
import numpy as np

# Define types for clarity in the function signatures
ArrayLike = Union[Sequence[Union[int, float]], np.ndarray]

__all__ = ["euclidean_distance", "manhattan_distance"]


def _to_1d_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input to a 1D float NumPy array with clear, consistent errors.

    Parameters
    ----------
    x : array_like
        Input vector.
    name : str
        Name used in error messages ("a" or "b").

    Returns
    -------
    np.ndarray
        1D array of dtype float.

    Raises
    ------
    ValueError
        If the array is not 1-dimensional.
    TypeError
        If the input contains non-numeric elements.
    """
    # Use np.asarray to handle various input types (list, tuple, np.ndarray)
    arr = np.asarray(x)

    # Dimensionality check first (so shape errors surface consistently)
    if arr.ndim != 1:
        raise ValueError(f"Input array '{name}' must be 1-dimensional; got {arr.ndim}D.")

    # Check for non-numeric dtypes (e.g., object from mixed/str, or string dtype)
    if not np.issubdtype(arr.dtype, np.number):
        # A more robust check might involve iterating or trying astype, but 
        # for standard NumPy array-like inputs, this usually catches the issue.
        # We perform a try-catch for astype below which is the ultimate safeguard.
        pass

    # Safe cast to float, which raises ValueError/TypeError if elements are non-numeric
    # (e.g., strings that can't be parsed as floats).
    try:
        # Use copy=False to avoid unnecessary copies if the dtype is already float
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as e:
        # Re-raise as TypeError for non-numeric content
        raise TypeError(f"All elements of '{name}' must be numeric (int or float). "
                        f"Original error: {e}")
    
    return arr


def _validate_arrays(a: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and align two 1D arrays for distance computation.

    This is a helper function to ensure both inputs are valid, 1D, 
    numeric NumPy arrays of the same shape.
    """
    a_arr = _to_1d_float_array(a, "a")
    b_arr = _to_1d_float_array(b, "b")

    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Arrays must have the same shape: a.shape={a_arr.shape}, b.shape={b_arr.shape}.")

    return a_arr, b_arr


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the Euclidean distance between two 1D arrays.

    The Euclidean distance (or L2 norm) is the 'straight-line' distance 
    between two points in Euclidean space. 

    The Euclidean distance is defined as:
    $$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i} (a_i - b_i)^2}$$

    Parameters
    ----------
    a : array_like
        First input vector (1D NumPy array, list, or tuple of numeric values).
    b : array_like
        Second input vector (1D NumPy array, list, or tuple of numeric values).

    Returns
    -------
    float
        The Euclidean distance between vectors `a` and `b`.

    Raises
    ------
    TypeError
        If `a` or `b` contains non-numeric elements.
    ValueError
        If `a` or `b` is not 1D or if shapes differ.

    Examples
    --------
    >>> import numpy as np
    >>> euclidean_distance(np.array([0, 0]), np.array([3, 4]))
    5.0
    >>> euclidean_distance([1, 2, 3], [1, 2, 3])
    0.0
    """
    a_arr, b_arr = _validate_arrays(a, b)
    
    # Implementation using np.linalg.norm for efficiency and numerical stability
    # The L2 norm of the difference vector (a_arr - b_arr) is the Euclidean distance.
    return float(np.linalg.norm(a_arr - b_arr))


def manhattan_distance(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the Manhattan (L1) distance between two 1D arrays.

    The Manhattan distance (or L1 norm, taxi-cab distance) is the sum of the 
    absolute differences of their Cartesian coordinates. 

    The Manhattan distance is defined as:
    $$d(\mathbf{a}, \mathbf{b}) = \sum_{i} |a_i - b_i|$$

    Parameters
    ----------
    a : array_like
        First input vector (1D NumPy array, list, or tuple of numeric values).
    b : array_like
        Second input vector (1D NumPy array, list, or tuple of numeric values).

    Returns
    -------
    float
        The Manhattan distance between vectors `a` and `b`.

    Raises
    ------
    TypeError
        If `a` or `b` contains non-numeric elements.
    ValueError
        If `a` or `b` is not 1D or if shapes differ.

    Examples
    --------
    >>> import numpy as np
    >>> manhattan_distance(np.array([1, 2, 3]), np.array([4, 0, 3]))
    5.0
    >>> manhattan_distance([0, 0], [0, 0])
    0.0
    """
    a_arr, b_arr = _validate_arrays(a, b)
    
    # Implementation: Sum of the absolute differences
    return float(np.sum(np.abs(a_arr - b_arr)))
