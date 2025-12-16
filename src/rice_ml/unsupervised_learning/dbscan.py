import numpy as np
from typing import Optional, Union, Sequence, Any, List, Tuple
import warnings
from ..utils.validation import ArrayLike, ensure_2d_numeric, ensure_1d_vector 

# --- Helper Definitions ---

def _get_neighbors(data: np.ndarray, query_idx: int, eps: float) -> np.ndarray:
    """
    Retrieves the indices of points within the epsilon-neighborhood of a given point.
    """
    diff = data - data[query_idx]
    distances_sq = np.sum(diff * diff, axis=1)
    return np.flatnonzero(distances_sq <= eps**2)


# --- DBSCAN Algorithm Implementation ---

class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.
    
    Identifies clusters based on density and can discover arbitrarily shaped 
    clusters while marking outliers (noise).

    Parameters
    ----------
    eps : float
        The maximum distance (epsilon) between two samples for one to be 
        considered as in the neighborhood of the other.
    min_samples : int
        The number of samples (or total weight) in a neighborhood for a point 
        to be considered as a core point.
    """
    _UNVISITED = 0
    _NOISE = -1

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if eps <= 0:
            raise ValueError("Epsilon (eps) must be greater than 0.")
        if min_samples < 1:
            raise ValueError("Minimum samples (min_samples) must be at least 1.")
            
        self.eps = eps
        self.min_samples = min_samples
        
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self._n_samples: int = 0

    def _expand_cluster(self, X: np.ndarray, labels: np.ndarray, core_idx: int, cluster_id: int) -> None:
        """
        Recursively adds all density-reachable points to the current cluster.
        """
        labels[core_idx] = cluster_id
        queue = list(_get_neighbors(X, core_idx, self.eps))
        
        head = 0
        while head < len(queue):
            point_idx = queue[head]
            head += 1
            
            if labels[point_idx] in (self._UNVISITED, self._NOISE):
                labels[point_idx] = cluster_id
                
                neighbors = _get_neighbors(X, point_idx, self.eps)
                
                if len(neighbors) >= self.min_samples:
                    for neighbor_idx in neighbors:
                        if labels[neighbor_idx] == self._UNVISITED:
                            queue.append(neighbor_idx)
        
    def fit(self, X: ArrayLike) -> "DBSCAN":
        """
        Performs DBSCAN clustering on the input data.
        """
        X_arr = ensure_2d_numeric(X)
        self._n_samples = X_arr.shape[0]
        
        labels = np.full(self._n_samples, self._UNVISITED, dtype=int)
        current_cluster_id = 0
        
        for i in range(self._n_samples):
            if labels[i] != self._UNVISITED:
                continue
            
            neighbors = _get_neighbors(X_arr, i, self.eps)
            
            if len(neighbors) >= self.min_samples:
                current_cluster_id += 1 
                self._expand_cluster(X_arr, labels, i, current_cluster_id)
            
            if labels[i] == self._UNVISITED:
                labels[i] = self._NOISE
        
        self.labels_ = labels
        self.n_clusters_ = current_cluster_id
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Convenience method to fit and return the labels."""
        self.fit(X)
        if self.labels_ is None:
             raise RuntimeError("Clustering failed.")
        return self.labels_