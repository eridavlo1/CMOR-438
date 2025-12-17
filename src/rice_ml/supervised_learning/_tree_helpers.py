import numpy as np
from typing import Union, Optional, Sequence, Any, Tuple
from ..processing.post_processing import mse

def gini_impurity(y: np.ndarray) -> float:
    r"""
    Computes the Gini Impurity.
    $$Gini(y) = 1 - \sum_{k=1}^{K} p_k^2$$
    """
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities ** 2))

def entropy(y: np.ndarray) -> float:
    r"""
    Computes the Entropy.
    $$Entropy(y) = - \sum_{k=1}^{K} p_k \log_2(p_k)$$
    """
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(-np.sum(probs * np.log2(probs, where=(probs > 0))))

def variance(y: np.ndarray) -> float:
    r"""Computes the variance for regression."""
    if len(y) == 0: return 0.0
    return float(np.var(y))

def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, 
                     metric: str = 'gini') -> float:
    r"""Calculates Gain = Parent_Impurity - Weighted_Child_Impurity"""
    n_p, n_l, n_r = len(y_parent), len(y_left), len(y_right)
    if n_p == 0: return 0.0
    
    m_func = {'gini': gini_impurity, 'entropy': entropy, 'variance': variance}
    if metric not in m_func: raise ValueError(f"Unknown metric: {metric}")
    
    impurity_func = m_func[metric]
    w_child_impurity = (n_l / n_p) * impurity_func(y_left) + (n_r / n_p) * impurity_func(y_right)
    return float(impurity_func(y_parent) - w_child_impurity)

__all__ = ['gini_impurity', 'entropy', 'variance', 'information_gain']