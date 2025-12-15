import numpy as np
from typing import Optional, Union, Sequence

# --- 1. Node Class for the Tree Structure ---

class Node:
    """A node in the regression tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        # Parameters for an internal (split) node
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node (True side of the split)
        self.right = right              # Right child node (False side of the split)

        # Parameter for a leaf node
        self.value = value              # The predicted mean value if it's a leaf

    def is_leaf_node(self):
        """Checks if the node is a terminal (leaf) node."""
        return self.value is not None

# --- 2. Mean Squared Error (MSE) Function ---

def _mean_squared_error(y: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (Variance) around the mean for a set of target values y.
    This serves as the impurity measure for regression trees.
    """
    if len(y) == 0:
        return 0.0
    
    # Cost = mean of (y - y_mean)^2
    return np.mean((y - np.mean(y))**2)

# --- 3. Regression Tree Regressor ---

class RegressionTree:
    """
    A simple CART-style Regression Tree implemented from scratch.
    Uses Mean Squared Error (MSE) / Variance Reduction for splitting.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If None, the tree is grown until 
        all leaves are pure or until min_samples_split is met.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.
        Default is 2.
    random_state : int, optional
        A seed for reproducibility.
    """
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root: Optional[Node] = None
        self.n_features: Optional[int] = None

    # --- Core Splitting Logic ---

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[Optional[int], Optional[float]]:
        """
        Find the best split based on the maximum Reduction in Variance (Information Gain).
        """
        best_variance_reduction = -1.0
        best_feature_idx = None
        best_threshold = None
        n_samples, _ = X.shape

        # Initial impurity (parent node cost)
        parent_mse = _mean_squared_error(y)

        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            # Consider unique values as potential thresholds
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # 1. Split the data
                left_indices = X_column <= threshold
                right_indices = X_column > threshold

                # Skip splits that result in empty children
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                if n_left < 1 or n_right < 1:
                    continue

                # Get targets for the children
                y_left = y[left_indices]
                y_right = y[right_indices]

                # 2. Calculate Weighted Child Cost (MSE)
                mse_left = _mean_squared_error(y_left)
                mse_right = _mean_squared_error(y_right)

                # Weighted MSE of the split
                weighted_child_mse = (n_left / n_samples) * mse_left + \
                                     (n_right / n_samples) * mse_right

                # 3. Calculate Variance Reduction (Information Gain)
                variance_reduction = parent_mse - weighted_child_mse

                # 4. Update the best split
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    # --- Recursive Tree Building ---

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds the regression tree.
        """
        n_samples, _ = X.shape

        # --- Check Stopping Criteria (Base Cases) ---
        
        # 1. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=np.mean(y))
        
        # 2. All target values are identical (pure node)
        # Note: Variance will be 0 if all y values are the same.
        if np.allclose(y, y[0]):
             return Node(value=np.mean(y))
            
        # 3. Minimum samples for a split not met
        if n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        # --- Find and Apply Best Split ---
        
        feature_idx, threshold = self._best_split(X, y)

        # If no split improves the variance reduction significantly
        if feature_idx is None or feature_idx < 0:
            return Node(value=np.mean(y))

        # Get indices for the split
        X_column = X[:, feature_idx]
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        
        # Handle case where the best split still yields an empty subset (should be caught by _best_split, but as safety)
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return Node(value=np.mean(y))

        # Recursive calls for children
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the internal node
        return Node(feature_idx, threshold, left_child, right_child)

    # --- Public API Methods ---

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]):
        """
        Builds the regression tree from the training data (X, y).
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("y must be a 1D array matching the number of rows in X.")
            
        self.n_features = X_arr.shape[1]
        self.root = self._build_tree(X_arr, y_arr)
        return self

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """
        Recursively traverse the tree for a single data point x.
        """
        # If it's a leaf, return the predicted mean value
        if node.is_leaf_node():
            return node.value

        # Get the feature value for the split
        feature_value = x[node.feature_idx]
        
        # Decide whether to go left or right
        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Predicts continuous target values for the input data X.

        Parameters
        ----------
        X : np.ndarray
            The input data to predict on.

        Returns
        -------
        np.ndarray
            The predicted continuous values.
        """
        if self.root is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
            
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            # Ensure input is 2D for consistent indexing
            X_arr = X_arr.reshape(1, -1)
            
        # Apply the traversal function to every row in X
        predictions = np.array([self._traverse_tree(x, self.root) for x in X_arr])
        return predictions