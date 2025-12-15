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
    """Check for NaN and Inf values and raise ValueError."""
    if np.isnan(arr).any():
        raise ValueError(f"Input array {name} contains NaN values. Please handle missing data before preprocessing.")
    if np.isinf(arr).any():
        raise ValueError(f"Input array {name} contains Infinite values. Please handle extreme data before preprocessing.")

def _ensure_2d_numeric(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric NumPy array."""
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

# --- Scaling / Normalization Functions --- #
def standardize(
    X: ArrayLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Z-score standardization (feature-wise).

    Each feature column is transformed to `(X - mean) / std` when enabled.
    Columns with zero variance are centered (if `with_mean=True`) but not scaled, 
    and the corresponding scaling factor returned in `params` is 1.0.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    with_mean : bool, default=True
        Center features by subtracting the column mean.
    with_std : bool, default=True
        Scale features by dividing by the column standard deviation.
    ddof : int, default=0
        Delta degrees of freedom for variance/std calculation.
    return_params : bool, default=False
        If True, also return a dict with keys ``mean`` and ``scale``.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Standardized array.
    params : dict, optional
        Only if `return_params=True`. Contains:
        - 'mean': ndarray of shape (n_features,)
        - 'scale': ndarray of shape (n_features,) (std; 1.0 where std=0)

    Raises
    ------
    ValueError
        If X is not 2D or is empty, or contains NaN/Inf.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2.], [3., 2.], [5., 2.]])
    >>> Z, params = standardize(X, return_params=True)
    >>> Z.mean(axis=0).round(7).tolist()
    [0.0, 0.0]
    >>> params["scale"].round(7).tolist()
    [1.6329932, 1.0]
    """
    X = _ensure_2d_numeric(X, "X")
    mean = X.mean(axis=0) if with_mean else np.zeros(X.shape[1], dtype=float)
    Xc = X - mean if with_mean else X.copy()

    if with_std:
        std = Xc.std(axis=0, ddof=ddof)
        scale = std.copy()
        scale[scale == 0.0] = 1.0
        X_out = Xc / scale
    else:
        scale = np.ones(X.shape[1], dtype=float)
        X_out = Xc

    if return_params:
        return X_out, {"mean": mean, "scale": scale}
    return X_out


def minmax_scale(
    X: ArrayLike,
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
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
    """
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
    """
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
    """
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
    """
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


def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.
    y : array_like, shape (n_samples,), optional
        Target vector. If provided, is split in the same way as X.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0 < test_size < 1).
    shuffle : bool, default=True
        Whether to shuffle the data before splitting (ignored when stratify is provided).
    stratify : array_like, optional
        If provided, data is split in a stratified fashion using these labels.
        Must be 1D and have length n_samples.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test : ndarray
        Split feature matrices.
    y_train, y_test : ndarray, optional
        Returned only if `y` is provided.

    Raises
    ------
    ValueError
        If input shapes are invalid, or test_size is not in (0, 1).
    TypeError
        If random_state is not an int or None.
    """
    X = _ensure_2d_numeric(X, "X")
    y_arr = _ensure_1d_vector(y, "y")
    _check_Xy_shapes(X, y_arr)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float in (0, 1).")

    n = X.shape[0]
    rng = _rng_from_seed(random_state)

    if stratify is not None:
        strat = _ensure_1d_vector(stratify, "stratify")
        if len(strat) != n:
            raise ValueError("stratify must have the same length as X.")
        # stratified split
        train_idx, test_idx = _stratified_indices(strat, test_size, rng)
    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        n_test = int(round(test_size * n))
        n_test = min(max(n_test, 1), n - 1) if n > 1 else n_test
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    if y_arr is None:
        return X_train, X_test
    y_train, y_test = y_arr[train_idx], y_arr[test_idx]
    return X_train, X_test, y_train, y_test

def train_val_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Split arrays into train, validation, and test subsets.

    The split is performed in two stages:
    1. The data is split into a Remainder (Train + Validation) set and a Test set.
    2. The Remainder set is then split into the final Train and Validation sets.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.
    y : array_like, shape (n_samples,), optional
        Target vector. If provided, is split in the same way as X.
    val_size : float, default=0.1
        Proportion of the *total* dataset for the validation split (0 < val_size < 1).
    test_size : float, default=0.2
        Proportion of the *total* dataset for the test split (0 < test_size < 1).
    shuffle : bool, default=True
        Whether to shuffle before splitting (ignored when stratify is provided).
    stratify : array_like, optional
        If provided, data is split in a stratified fashion using these labels.
        Must be 1D and have length n_samples.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test : ndarray
        Split feature matrices.
    y_train, y_val, y_test : ndarray, optional
        Returned only if `y` is provided.

    Raises
    ------
    ValueError
        If sizes are invalid, shapes mismatch, or `val_size + test_size >= 1.0`.
    TypeError
        If random_state is not an int or None.
    """
    if not (0.0 < val_size < 1.0) or not (0.0 < test_size < 1.0):
        raise ValueError("val_size and test_size must be floats in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0.")

    # 1. Split into Remainder (Train + Validation) and Test
    split1_out = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=stratify,
        random_state=random_state,
    )
    
    if y is None:
        X_rem, X_test = split1_out
        y_rem = None
    else:
        X_rem, X_test, y_rem, y_test = split1_out

    # Handle edge case where remainder is empty
    if X_rem.shape[0] == 0:
        X_train, X_val = X_rem, X_rem
        y_train, y_val = y_rem, y_rem
    else:
        # 2. Recalculate validation size relative to the REMAINING data
        # val_prop_remaining = val_size / (1 - test_size)
        val_prop_remaining = val_size / (1.0 - test_size)
        
        # Prepare the stratification vector for the remaining data
        # If stratify was provided, the corresponding labels in y_rem are used for the second split's stratification
        stratify_rem = y_rem if stratify is not None else None
        
        # 3. Split Remainder into Train and Validation
        split2_out = train_test_split(
            X_rem,
            y_rem,
            test_size=val_prop_remaining,
            shuffle=shuffle,
            stratify=stratify_rem, 
            random_state=random_state,
        )
        
        if y is None:
            X_train, X_val = split2_out
            y_train, y_val = None, None
        else:
            X_train, X_val, y_train, y_val = split2_out

    if y is None:
        return X_train, X_val, X_test

    return X_train, X_val, X_test, y_train, y_val, y_test