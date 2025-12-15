import numpy as np
from typing import Optional, Union, Sequence, Any, Tuple, List, Literal
import warnings
from rice_ml.utils import ArrayLike, ensure_2d_numeric 
from rice_ml.supervised_learning.distances_metrics import euclidean_distance

class KMeans:
    """
    K-Means Clustering Algorithm.

    K-Means is an unsupervised, iterative, centroid-based clustering algorithm 
    that partitions the dataset into K distinct, non-overlapping subsets (clusters) 
    such that the within-cluster variance is minimized. 

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters (K) to form.
    max_iter : int, default=300
        The maximum number of iterations for the algorithm.
    tol : float, default=1e-4
        Tolerance for convergence. If the change in centroids between iterations 
        is less than this value, the algorithm stops.
    init : {'random', 'k-means++'}, default='random'
        Method for centroid initialization.
        - 'random': Choose K random samples from the data as initial centroids.
        - 'k-means++': Employs a smarter seeding technique to accelerate convergence.
    random_state : Optional[int], default=None
        Seed for random number generation for initialization.

    Attributes
    ----------
    cluster_centers_ : np.ndarray
        Coordinates of the K cluster centers found by the algorithm. Shape (n_clusters, n_features).
    labels_ : np.ndarray
        Labels of each sample, i.e., the index of the cluster to which each sample belongs.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center (the objective value).
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-4, 
                 init: Literal['random', 'k-means++'] = 'random', random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self._rng = np.random.default_rng(random_state)

    # --- Initialization Methods ---
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes cluster centroids based on the specified method."""
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Select n_clusters unique indices randomly
            random_indices = self._rng.choice(n_samples, size=self.n_clusters, replace=False)
            centroids = X[random_indices]
            
        elif self.init == 'k-means++':
            # 1. Choose one center uniformly at random.
            centroids = [X[self._rng.choice(n_samples)]]
            
            for _ in range(1, self.n_clusters):
                # 2. Compute the distance D(x) from each data point x to the nearest center.
                sq_distances = []
                for x in X:
                    # Calculate distance from x to ALL current centroids
                    dist_to_centers = [np.sum((x - c) ** 2) for c in centroids]
                    # D(x) is the distance to the CLOSEST center
                    sq_distances.append(min(dist_to_centers))
                
                sq_distances = np.array(sq_distances)
                
                # 3. Choose the next center with probability proportional to D(x)^2 (or D(x))
                # Probability P(x) = D(x)^2 / sum(D(x)^2)
                probabilities = sq_distances / np.sum(sq_distances)
                
                # Choose new index based on probabilities (must handle sum=0 case)
                if np.sum(probabilities) == 0:
                    new_center_idx = self._rng.choice(n_samples) # Fallback to random
                else:
                    new_center_idx = self._rng.choice(n_samples, p=probabilities)
                
                centroids.append(X[new_center_idx])
            
            centroids = np.array(centroids)
            
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
            
        return centroids

    # --- Core K-Means Step ---

    def _assign_clusters(self, X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, float]:
        """Assigns each sample to the nearest centroid and calculates inertia."""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        inertia = 0.0

        for i in range(n_samples):
            x = X[i]
            # Calculate Euclidean distance from x to ALL centers (requires a vector operation or loop)
            # Use the imported or local helper:
            distances = [np.sum((x - center) ** 2) for center in centers]
            
            # Find the index of the closest center
            closest_center_index = np.argmin(distances)
            labels[i] = closest_center_index
            inertia += distances[closest_center_index] # Sum of squared distances (inertia)
            
        return labels, inertia
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recalculates centroids as the mean of all samples assigned to the cluster."""
        n_features = X.shape[1]
        new_centers = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Get all samples assigned to cluster k
            cluster_k_samples = X[labels == k]
            
            if len(cluster_k_samples) > 0:
                # Calculate the mean of these samples
                new_centers[k] = np.mean(cluster_k_samples, axis=0)
            else:
                # Handle empty cluster: re-initialize the centroid randomly or keep old one
                # Standard practice is often to keep the old centroid or re-initialize randomly
                warnings.warn(f"Cluster {k} is empty. Re-initializing centroid.")
                # For simplicity here, we re-initialize it as a random point from X
                random_idx = self._rng.choice(len(X))
                new_centers[k] = X[random_idx]
                
        return new_centers

    # --- Public API ---

    def fit(self, X: ArrayLike) -> "KMeans":
        """
        Computes the K-Means clustering on the input data.
        """
        X_arr = ensure_2d_numeric(X)
        n_samples, n_features = X_arr.shape
        
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be greater than n_clusters.")

        # 1. Initialize Centroids
        current_centers = self._initialize_centroids(X_arr)
        
        # 2. Main Iteration Loop
        for i in range(self.max_iter):
            old_centers = current_centers.copy()
            
            # E-step: Assign samples to the nearest cluster
            labels, inertia = self._assign_clusters(X_arr, current_centers)
            
            # M-step: Recalculate centroids
            current_centers = self._update_centroids(X_arr, labels)
            
            # Check for convergence (change in centroids)
            # Calculate the distance moved by all centroids
            center_shift = np.sum(np.sqrt(np.sum((current_centers - old_centers)**2, axis=1)))
            
            if center_shift <= self.tol:
                print(f"K-Means converged after {i + 1} iterations.")
                break
        else:
            print(f"K-Means failed to converge after {self.max_iter} iterations.")

        # Finalize attributes
        self.cluster_centers_ = current_centers
        self.labels_ = labels
        self.inertia_ = inertia
        
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Assigns new data points to the closest cluster center found during fitting.
        
        Returns
        -------
        np.ndarray
            The cluster label (index) for each sample in X.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")
            
        X_arr = ensure_2d_numeric(X)
        
        # Re-use the assignment logic
        labels, _ = self._assign_clusters(X_arr, self.cluster_centers_)
        
        return labels