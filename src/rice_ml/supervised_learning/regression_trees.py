import numpy as np
from typing import Optional, Union, Sequence, Any, Tuple, Literal
from ..utils import ArrayLike, ensure_2d_numeric, ensure_1d_vector, check_Xy_shapes
from ._tree_helpers import variance, information_gain

# --- Internal Node Class ---
class Node:
    """A node in the regression tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Mean target value for leaf nodes
        
    def is_leaf_node(self):
        return self.value is not None

# --- Decision Tree Regressor ---

class DecisionTreeRegressor:
    """
    A Decision Tree Regressor using the CART algorithm with Variance Reduction 
    (based on Mean Squared Error) as the splitting criterion. 
    
    Parameters
    ----------
    max_depth : Optional[int], default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, 
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree_: Optional[Node] = None
        self.n_features_: Optional[int] = None
        
    def _get_leaf_value(self, y: np.ndarray) -> float:
        """Determines the prediction value for a leaf node (the mean of the target values)."""
        return float(np.mean(y))

    # --- Core Tree Building Logic ---

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Finds the split that maximizes variance reduction.
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        best_gain = -1.0
        best_feature_idx = None
        best_threshold = None
        y_parent = y

        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            
            for threshold in unique_values:
                left_mask = X[:, feature_idx] <= threshold
                y_left = y[left_mask]
                y_right = y[~left_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Use the centralized information_gain helper with the 'variance' metric
                gain = information_gain(y_parent, y_left, y_right, metric='variance')
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively builds the regression tree.
        """
        n_samples, _ = X.shape
        mean_y = np.mean(y)

        # Base Cases
        # Use variance helper for purity check (variance < 1e-6)
        if (variance(y) < 1e-6 or 
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split):
            
            return Node(value=mean_y)

        feature_idx, threshold, gain = self._find_best_split(X, y)

        if gain <= 0:
            return Node(value=mean_y)

        left_mask = X[:, feature_idx] <= threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature_idx, threshold, left_child, right_child)

    # --- Public API ---

    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Builds the Regression Tree from the training data.
        """
        X_arr = ensure_2d_numeric(X, name="X")
        y_arr = ensure_1d_vector(y, name="y")
        check_Xy_shapes(X_arr, y_arr)
        
        if not np.issubdtype(y_arr.dtype, np.number):
             y_arr = y_arr.astype(float, copy=False)

        self.n_features_ = X_arr.shape[1]
        self.tree_ = self._build_tree(X_arr, y_arr, depth=0)
        
        return self

    def _traverse_tree(self, x: np.ndarray, node: Optional[Node]) -> float:
        """
        Traverses the fitted tree to find the prediction for a single data point x.
        """
        if node is None:
            raise RuntimeError("Traversal attempted on empty tree node.")
            
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts continuous target values for the input data.
        """
        if self.tree_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
            
        X_arr = ensure_2d_numeric(X, name="X")
            
        predictions = np.array([self._traverse_tree(x, self.tree_) for x in X_arr])
        return predictions
    
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Returns the R^2 score of the prediction.
        """
        from rice_ml.post_processing import r2_score
        y_pred = self.predict(X)
        y_true = ensure_1d_vector(y)
        return float(r2_score(y_true, y_pred))