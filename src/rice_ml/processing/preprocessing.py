from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, List
import numpy as np

__all__ = [
    'ArrayLike',
    'standardize',
    'minmax_scale',
    'maxabs_scale',
    'l1_normalize_rows', 
    'l2_normalize_rows',
    'train_test_split',
    'train_val_test_split',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

# ---- Internal Validation Helper ---- #

def _check_for_nan_inf(arr: np.ndarray, name: str) -> None:
    r"""Check for NaN and Inf values and raise ValueError."""
    if np.isnan(arr).any():
        raise ValueError(f"Input array {name} contains NaN values. Please handle missing data before preprocessing.")
    if np.isinf(arr).any():
        raise ValueError(f"Input array {name} contains Infinite values. Please handle extreme data before preprocessing.")

def _ensure_2d_numeric(X: ArrayLike, name: str = "X") -> np.ndarray:
    r"""Ensure X is a 2D numeric NumPy array."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    _check_for_nan_inf(arr, name)

    return arr

def _ensure_1d_vector(y: Optional[ArrayLike], name: str = "y") -> Optional[np.ndarray]:
    """Ensure y is a 1D array (numeric or categorical), or None."""
    if y is None:
        return None
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")
    return arr

def _check_Xy_shapes(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    if y is not None and len(y) != X.shape[0]:
        raise ValueError(
            f"X and y must have compatible first dimension; got len(y)={len(y)} "
            f"and X.shape[0]={X.shape[0]}."
        )

def _rng_from_seed(random_state: Optional[int]) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if not (isinstance(random_state, (int, np.integer))):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(random_state))

# --- Scaling / Normalization Functions ---
def standardize(
    X: ArrayLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    return_params: bool = False,
    # --- ENHANCED PARAMETERS FOR TRANSFORMATION ---
    mean: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    r"""
    Z-score standardization (feature-wise).
    
    This function can be used in two modes:
    1. Fit and Transform (default): Calculates mean/std from X and scales X.
    2. Transform only: Uses provided 'mean' and 'scale' arrays to transform X.

    Each feature column is transformed to `(X - mean) / scale` when enabled.
    Columns with zero variance are centered (if `with_mean=True`) but not scaled, 
    and the corresponding scaling factor used is 1.0.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    with_mean : bool, default=True
        Center features by subtracting the column mean (if fitting).
    with_std : bool, default=True
        Scale features by dividing by the column standard deviation (if fitting).
    ddof : int, default=0
        Delta degrees of freedom for variance/std calculation during fitting.
    return_params : bool, default=False
        If True, also return a dict with keys ``mean`` and ``scale`` (only applies if fitting).
    mean : ndarray, optional
        Pre-calculated mean to use for transformation (avoids fitting).
    scale : ndarray, optional
        Pre-calculated scale (standard deviation) to use for transformation (avoids fitting).

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Standardized array.
    params : dict, optional
        Only if `return_params=True`. Contains: 'mean' and 'scale'.

    Raises
    ------
    ValueError
        If X is not 2D or is empty, or if mean/scale dimensions are mismatched.
    """
    X_arr = _ensure_2d_numeric(X, "X")

    # --- MODE 1: TRANSFORM ONLY (mean and scale are provided) ---
    if mean is not None and scale is not None:
        # Validate that provided parameters match the number of features
        if X_arr.shape[1] != mean.shape[0] or X_arr.shape[1] != scale.shape[0]:
            raise ValueError("Provided 'mean' or 'scale' dimensions do not match X features.")

        X_centered = X_arr - mean if with_mean else X_arr
        
        # Guard against zero scale if the user provided it (though scale should already be guarded)
        # Note: If scale was calculated correctly (guarded against zero), this step is safe.
        scale_guarded = scale.copy()
        scale_guarded[scale_guarded == 0.0] = 1.0 
        
        X_out = X_centered / scale_guarded if with_std else X_centered
        return X_out

    # --- MODE 2: FIT AND TRANSFORM (mean and scale are calculated from X) ---
    
    # 1. Calculate Mean (for centering)
    mean_calc = X_arr.mean(axis=0) if with_mean else np.zeros(X_arr.shape[1], dtype=float)
    X_centered = X_arr - mean_calc
    
    # 2. Calculate Scale (for dividing)
    if with_std:
        std_calc = X_centered.std(axis=0, ddof=ddof)
        scale_calc = std_calc.copy()
        # Guard against zero variance/std: set scale factor to 1.0 where std is 0
        scale_calc[scale_calc == 0.0] = 1.0
        X_out = X_centered / scale_calc
    else:
        scale_calc = np.ones(X_arr.shape[1], dtype=float)
        X_out = X_centered

    # 3. Return results
    if return_params:
        return X_out, {"mean": mean_calc, "scale": scale_calc}
    return X_out


def minmax_scale(
    X: ArrayLike,
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    r"""
    Scale features to a specified range (feature-wise). 

    The transformation is: `X_scaled = (X - min) / (max - min) * (max' - min') + min'`.
    Columns with zero range (`max - min = 0`) are mapped to the minimum value of 
    `feature_range`.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    feature_range : tuple(float, float), default=(0.0, 1.0)
        Desired value range for each feature (min', max').
    return_params : bool, default=False
        If True, also return a dict with keys ``min`` and ``scale``.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Scaled array.
    params : dict, optional
        Only if `return_params=True`. Contains:
        - 'min': ndarray of shape (n_features,) (original min)
        - 'scale': ndarray of shape (n_features,) (original max-min; 1.0 where range=0)
        - 'feature_range': tuple(float, float) (the requested range)

    Raises
    ------
    ValueError
        If X is not 2D, empty, or feature_range is invalid.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0., 10.], [5., 10.], [10., 10.]])
    >>> X2, params = minmax_scale(X, feature_range=(0, 1), return_params=True)
    >>> X2[:, 0].tolist()
    [0.0, 0.5, 1.0]
    >>> X2[:, 1].tolist()
    [0.0, 0.0, 0.0]
    >>> params["scale"].tolist()
    [10.0, 1.0]
    """
    X = _ensure_2d_numeric(X, "X")
    if not (
        isinstance(feature_range, tuple)
        and len(feature_range) == 2
        and all(isinstance(v, (int, float)) for v in feature_range)
    ):
        raise ValueError("feature_range must be a tuple of two numeric values (min, max).")
    fr_min, fr_max = float(feature_range[0]), float(feature_range[1])
    if fr_min >= fr_max:
        raise ValueError("feature_range must have min < max.")

    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    range_ = Xmax - Xmin
    scale = range_.copy()
    scale[scale == 0.0] = 1.0
    
    # Transformation steps
    X01 = (X - Xmin) / scale
    X_out = X01 * (fr_max - fr_min) + fr_min

    if return_params:
        return X_out, {"min": Xmin, "scale": scale, "feature_range": (fr_min, fr_max)}
    return X_out


def maxabs_scale(
    X: ArrayLike,
    *,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    r"""
    Scale features by their maximum absolute value (feature-wise).

    The transformation is: `X_scaled = X / max(|X|)`.
    Columns that contain all zeros are left unchanged (effectively divided by 1).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    return_params : bool, default=False
        If True, also return a dict with key ``scale``.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Scaled array.
    params : dict, optional
        Only if `return_params=True`. Contains:
        - 'scale': ndarray of shape (n_features,) (max abs per feature; 1.0 where max abs=0)

    Raises
    ------
    ValueError
        If X is not 2D, empty, or contains NaN/Inf.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-2., 0.], [1., 0.], [2., 0.]])
    >>> X2, params = maxabs_scale(X, return_params=True)
    >>> X2[:, 0].tolist()
    [-1.0, 0.5, 1.0]
    >>> X2[:, 1].tolist()
    [0.0, 0.0, 0.0]
    >>> params["scale"].tolist()
    [2.0, 1.0]
    """
    X = _ensure_2d_numeric(X, "X")
    maxabs = np.max(np.abs(X), axis=0)
    scale = maxabs.copy()
    scale[scale == 0.0] = 1.0
    X_out = X / scale
    if return_params:
        return X_out, {"scale": scale}
    return X_out


def l1_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    r"""
    Row-wise L1 normalization (sum of absolute values = 1).

    Each row x is replaced by x / max(||x||_1, eps). Rows containing all zeros 
    remain zero.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    eps : float, default=1e-12
        Floor value to avoid division by zero when the L1 norm is zero.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Row-wise L1-normalized array.

    Raises
    ------
    ValueError
        If X is not 2D, empty, or eps <= 0.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2., 3.], [0., 0., 0.]])
    >>> Xn = l1_normalize_rows(X)
    >>> Xn[0].sum().round(7)
    1.0
    >>> Xn[1].tolist()
    [0.0, 0.0, 0.0]
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    X = _ensure_2d_numeric(X, "X")
    # L1 norm (sum of absolute values)
    norms = np.sum(np.abs(X), axis=1)
    denom = np.maximum(norms, eps)[:, None]
    return X / denom

def l2_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    r"""
    Row-wise L2 normalization (Euclidean norm = 1). 

    Each row x is replaced by x / max(||x||_2, eps). Rows containing all zeros 
    remain zero.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    eps : float, default=1e-12
        Floor value to avoid division by zero when the L2 norm is zero.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Row-wise L2-normalized array.

    Raises
    ------
    ValueError
        If X is not 2D, empty, or eps <= 0.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[3., 4.], [0., 0.]])
    >>> Xn = l2_normalize_rows(X)
    >>> np.allclose(np.linalg.norm(Xn[0]), 1.0)
    True
    >>> Xn[1].tolist()
    [0.0, 0.0]
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    X = _ensure_2d_numeric(X, "X")
    # L2 norm (sqrt of sum of squares)
    norms = np.sqrt(np.sum(X ** 2, axis=1))
    denom = np.maximum(norms, eps)[:, None]
    return X / denom

# --- Splitting ---

def _stratified_indices(y: np.ndarray, split_size: float, rng: np.random.Generator):
    r"""
    Internal helper to return train/test indices with class-wise proportional sampling.
    """
    classes, y_indices = np.unique(y, return_inverse=True)
    split1_idx: List[int] = []
    split2_idx: List[int] = []
    
    for cls in range(len(classes)):
        cls_indices = np.flatnonzero(y_indices == cls)
        rng.shuffle(cls_indices)
        n_total = len(cls_indices)
        
        n_split2 = int(round(split_size * n_total))
        
        # Ensure splits are non-empty for classes with > 1 sample
        if n_total > 1:
            n_split2 = min(max(n_split2, 1), n_total - 1)
        elif n_total == 1:
            # Single sample class is assigned entirely to split1 (train)
            n_split2 = 0
            
        split2_idx.extend(cls_indices[:n_split2])
        split1_idx.extend(cls_indices[n_split2:])
        
    return np.array(split1_idx), np.array(split2_idx)


def train_test_split(X, y=None, test_size=0.25, random_state=None, shuffle=True, stratify=None):
    r"""
    Split arrays into random train and test subsets with validation and stratification.
    """
    X = np.asanyarray(X)
    
    # Validation for X dimensionality
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
        
    # Validation for test_size range
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float in (0, 1).")
        
    # Validation for random_state type
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError("random_state must be an integer or None.")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if y is not None:
        y = np.asanyarray(y)
        if y.shape[0] != n_samples:
            raise ValueError("X and y must have a compatible first dimension.")

    # Handle Stratification
    if stratify is not None:
        y_strat = np.asanyarray(stratify)
        if y_strat.shape[0] != n_samples:
            raise ValueError("Stratify array must have same length as X.")
        
        rng = np.random.default_rng(random_state)
        classes, y_indices = np.unique(y_strat, return_inverse=True)
        
        train_idx, test_idx = [], []
        for i in range(len(classes)):
            idx_class = indices[y_indices == i]
            rng.shuffle(idx_class)
            split = int(len(idx_class) * (1 - test_size))
            # Ensure at least one sample in test if possible
            if split == len(idx_class) and len(idx_class) > 0:
                split -= 1
            train_idx.extend(idx_class[:split])
            test_idx.extend(idx_class[split:])
            
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    
    elif shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        split_idx = int(n_samples * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    else:
        # Strictly sequential split for shuffle=False
        split_idx = int(n_samples * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    if y is not None:
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return X[train_idx], X[test_idx]

def train_val_test_split(X, y=None, val_size=0.2, test_size=0.2, random_state=None, stratify=None):
    """3-way split supporting stratification and size validation."""
    if (val_size + test_size) >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    if y is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        # Recalculate val_size relative to the remaining data
        rel_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=rel_val_size, random_state=random_state, 
            stratify=y_temp if stratify is not None else None
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_temp, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        rel_val_size = val_size / (1 - test_size)
        X_train, X_val = train_test_split(X_temp, test_size=rel_val_size, random_state=random_state)
        return X_train, X_val, X_test