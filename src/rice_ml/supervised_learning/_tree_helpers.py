import numpy as np
from typing import Union, Optional, Sequence, Any, Tuple
from ..processing.post_processing import mse

# --- Impurity/Variance Functions ---

def gini_impurity(y: np.ndarray) -> float:
    """
    Computes the Gini Impurity of a target vector for classification.

    $$Gini(y) = 1 - \sum_{k=1}^{K} p_k^2$$

    Parameters
    ----------
    y : np.ndarray
        The 1D array of class labels.

    Returns
    -------
    float
        The Gini Impurity score (0.0 for pure nodes).
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    
    if n_samples == 0:
        return 0.0

    probabilities = counts / n_samples
    gini = 1.0 - np.sum(probabilities ** 2)
    
    return float(gini)


def entropy(y: np.ndarray) -> float:
    """
    Computes the Entropy (Information Content) of a target vector for classification.

    $$Entropy(y) = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

    Parameters
    ----------
    y : np.ndarray
        The 1D array of class labels.

    Returns
    -------
    float
        The Entropy score (0.0 for pure nodes).
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    
    if n_samples == 0:
        return 0.0

    probabilities = counts / n_samples
    
    # Use log2 for information theory, handling log(0) case
    entropy_val = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    
    return float(entropy_val)


def variance(y: np.ndarray) -> float:
    """
    Computes the variance of a target vector for regression.

    This measures the impurity (or error) of a node for regression trees.

    Parameters
    ----------
    y : np.ndarray
        The 1D array of target values.

    Returns
    -------
    float
        The variance (MSE around the mean) of the target values.
    """
    if len(y) == 0:
        return 0.0
    
    # Variance is equivalent to the MSE of the target values around their mean prediction
    y_mean = np.mean(y)
    var = mse(y, np.full_like(y, y_mean))
    
    return float(var)


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, 
                     metric: str = 'gini') -> float:
    """
    Calculates the Information Gain (or Variance Reduction) from a split.
    

    Gain = Parent_Impurity - Weighted_Child_Impurity

    Parameters
    ----------
    y_parent : np.ndarray
        Target values of the parent node.
    y_left : np.ndarray
        Target values of the left child node.
    y_right : np.ndarray
        Target values of the right child node.
    metric : {'gini', 'entropy', 'variance'}, default='gini'
        The impurity metric to use. Use 'variance' for regression.

    Returns
    -------
    float
        The gain in purity/reduction in variance.
    """
    n_parent = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_parent == 0:
        return 0.0
        
    # 1. Determine the impurity function
    if metric == 'gini':
        impurity_func = gini_impurity
    elif metric == 'entropy':
        impurity_func = entropy
    elif metric == 'variance':
        impurity_func = variance
    else:
        raise ValueError(f"Unknown impurity metric: {metric}")
    
    # 2. Calculate impurity of the parent
    parent_impurity = impurity_func(y_parent)
    
    # 3. Calculate weighted impurity of the children
    weighted_child_impurity = (n_left / n_parent) * impurity_func(y_left) + \
                              (n_right / n_parent) * impurity_func(y_right)
    
    # 4. Gain = Parent Impurity - Weighted Child Impurity
    gain = parent_impurity - weighted_child_impurity
    
    return float(gain)


# --- Public API ---

__all__ = [
    'gini_impurity', 
    'entropy', 
    'variance', 
    'information_gain'
]