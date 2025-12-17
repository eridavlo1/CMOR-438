import numpy as np

def _validate_numeric(a, b):
    r"""
    Internal helper to validate that inputs are numeric, 1D, and have matching shapes.
    
    Raises
    ------
    TypeError
        If inputs are not numeric (e.g., contain strings or objects).
    ValueError
        If inputs have different shapes or are not 1-dimensional.
    """
    # Convert inputs to numpy arrays for inspection
    a_arr = np.asanyarray(a)
    b_arr = np.asanyarray(b)

    # 1. Strict Numeric Check: Ensures dtypes like <U1 (strings) or object raise TypeError
    if not (np.issubdtype(a_arr.dtype, np.number) and np.issubdtype(b_arr.dtype, np.number)):
        raise TypeError("Input arrays must contain numeric values.")

    # 2. Shape Mismatch Check
    if a_arr.shape != b_arr.shape:
        raise ValueError("Input arrays must have the same shape.")

    # 3. Dimensionality Check (Strictly 1D vectors for these metrics)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        # Note: Empty arrays np.array([]) are 1D with size 0, so they pass this check.
        raise ValueError("Inputs must be 1-dimensional vectors.")

    return a_arr.astype(float), b_arr.astype(float)

def euclidean_distance(a, b):
    r"""
    Compute the Euclidean distance between two 1D numeric vectors.
    
    The Euclidean distance is defined as:
    $d(a, b) = \sqrt{\sum (a_i - b_i)^2}$
    """
    a_f, b_f = _validate_numeric(a, b)
    
    # Handle empty array edge case
    if a_f.size == 0:
        return 0.0
        
    return np.sqrt(np.sum((a_f - b_f) ** 2))

def manhattan_distance(a, b):
    r"""
    Compute the Manhattan distance between two 1D numeric vectors.
    
    The Manhattan distance is defined as:
    $d(a, b) = \sum |a_i - b_i|$
    """
    a_f, b_f = _validate_numeric(a, b)
    
    # Handle empty array edge case
    if a_f.size == 0:
        return 0.0
        
    return np.sum(np.abs(a_f - b_f))