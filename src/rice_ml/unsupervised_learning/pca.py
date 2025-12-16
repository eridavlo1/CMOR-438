import numpy as np
from typing import Optional, Union, Sequence, Any, Literal, Tuple
import warnings
from ..utils import ArrayLike, ensure_2d_numeric 

class PCA:
    """
    Principal Component Analysis (PCA) for Dimensionality Reduction.

    PCA is an unsupervised linear transformation technique that identifies the 
    directions (principal components) in the data that maximize variance. It projects 
    the data onto a new, lower-dimensional subspace while retaining as much 
    of the original data's variability as possible. 

    Parameters
    ----------
    n_components : Optional[int], default=None
        The number of principal components to retain (the target dimension). 
        If None, all components (min(n_samples, n_features)) are kept.
        If 0 < n_components < 1, it specifies the desired ratio of variance 
        to be explained (e.g., 0.95 for 95% variance).
    random_state : Optional[int], default=None
        Seed for reproducibility (e.g., for internal sorting/tie-breaking).

    Attributes
    ----------
    components_ : np.ndarray
        The principal axes (components) in feature space, representing the 
        directions of maximum variance. Shape (n_components, n_features).
    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each selected component.
    mean_ : np.ndarray
        Per-feature empirical mean, used for centering the data.
    n_components_ : int
        The actual number of components selected after fitting.
    """

    def __init__(self, n_components: Optional[Union[int, float]] = None, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    # --- Core PCA Steps ---

    def _select_components(self, explained_variance_ratio: np.ndarray, n_features: int) -> int:
        """Determines the final number of components based on n_components parameter."""
        
        if self.n_components is None:
            # If None, keep min(n_samples, n_features) - here we use n_features as an upper bound
            return n_features
        
        if isinstance(self.n_components, int):
            if self.n_components <= 0 or self.n_components > n_features:
                raise ValueError(f"n_components={self.n_components} must be > 0 and <= n_features ({n_features}).")
            return self.n_components
            
        if isinstance(self.n_components, float):
            if not (0.0 < self.n_components < 1.0):
                raise ValueError(f"n_components={self.n_components} must be between 0.0 and 1.0 when a float.")
            
            # Find the number of components needed to explain the desired variance ratio
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_comp = np.argmax(cumulative_variance >= self.n_components) + 1
            
            if n_comp > n_features: # Safety check
                n_comp = n_features
                
            return int(n_comp)
            
        raise TypeError(f"Invalid type for n_components: {type(self.n_components)}")


    # --- Public API ---

    def fit(self, X: ArrayLike) -> "PCA":
        """
        Computes the principal components from the input data.
        
        1. Centers the data.
        2. Computes the covariance matrix.
        3. Performs Eigenvalue Decomposition.
        4. Selects components based on `n_components`.
        """
        X_arr = ensure_2d_numeric(X)
        n_samples, n_features = X_arr.shape
        
        # 1. Centering the data
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_
        
        # 2. Compute Covariance Matrix (C = X_T * X / (N - 1))
        # Note: If n_samples < n_features, we can calculate X*X_T and use the property 
        # that eigenvalues are the same, but we stick to the standard C calculation here.
        covariance_matrix = np.cov(X_centered, rowvar=False) # rowvar=False means features are columns

        # 3. Perform Eigenvalue Decomposition
        # V = eigenvectors (columns), D = eigenvalues (vector)
        # Eigenvectors are the principal components (PCs)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by corresponding eigenvalues in descending order
        # We need to sort both eigenvalues and eigenvectors simultaneously
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices] # Sort columns of eigenvectors

        # 4. Calculate Explained Variance Ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance

        # 5. Select Components
        n_components_final = self._select_components(explained_variance_ratio, n_features)
        
        # The principal components are the first n_components eigenvectors
        # (transposed to have shape (n_components, n_features))
        self.components_ = eigenvectors[:, :n_components_final].T
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components_final]
        self.n_components_ = n_components_final

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Applies dimensionality reduction to X by projecting it onto the 
        principal components.
        
        Z = (X - mean_) @ components_.T
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X) first.")
            
        X_arr = ensure_2d_numeric(X)
        
        # Center the new data using the mean learned during fitting
        X_centered = X_arr - self.mean_
        
        # Project the centered data onto the principal components (PCs)
        # Components_ has shape (n_components, n_features)
        # Projection Z = X_centered @ W (where W is components_.T)
        X_transformed = X_centered @ self.components_.T 
        
        return X_transformed

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fits the model with X and immediately applies the dimensionality reduction.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """
        Transforms the reduced data Z back to the original feature space.
        
        X_reconstructed = Z @ components_ + mean_
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Model is not fitted.")
            
        Z_arr = ensure_2d_numeric(Z)
        
        # Reconstruct the data: X_reconstructed = Z @ W_T + mean
        # Components_ has shape (n_components, n_features)
        # W_T is components_
        X_reconstructed = Z_arr @ self.components_ + self.mean_
        
        return X_reconstructed