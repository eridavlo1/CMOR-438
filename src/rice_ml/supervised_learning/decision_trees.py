import numpy as np
from typing import Optional, Union, Sequence, Any, Tuple, Literal, List
import warnings

from rice_ml.utils import ArrayLike, ensure_2d_numeric, ensure_1d_vector, check_Xy_shapes
from rice_ml.supervised_learning._tree_helpers import gini_impurity, entropy, information_gain 

# --- Internal Node Class ---

class Node:
    """
    A node in a decision tree, representing either a split or a leaf.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # Index of the feature to split on (for split nodes)
        self.threshold = threshold   # Threshold value for the split (for split nodes)
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.value = value           # Class label or prediction (for leaf nodes)
        
    def is_leaf(self):
        """Checks if the node is a terminal leaf node."""
        return self.value is not None

# --- Decision Tree Classifier ---

class DecisionTreeClassifier:
    """
    A simple Decision Tree Classifier using the CART (Classification and Regression Tree) 
    algorithm with Gini Impurity or Entropy as the splitting criterion. 
    
    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_depth : Optional[int], default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    random_state : Optional[int], default=None
        Seed for reproducibility.
    """
    
    def __init__(self, criterion: Literal['gini', 'entropy'] = 'gini', max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, max_features: Optional[Union[str, float, int]] = None, random_state: Optional[int] = None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree_: Optional[Node] = None 
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        
        # Determine impurity function based on criterion
        if self.criterion == 'gini':
            self._impurity_func = gini_impurity
        elif self.criterion == 'entropy':
            self._impurity_func = entropy
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    # --- Core Tree Building Logic ---

    def _get_leaf_value(self, y: np.ndarray) -> Any:
        """Determines the prediction value for a leaf node (the mode of the labels)."""
        counts = np.unique(y, return_counts=True)
        return counts[0][np.argmax(counts[1])]

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Finds the split that maximizes information gain using the criterion.
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0
        
        # --- Random Feature Subset Logic ---
        rng = np.random.default_rng(self.random_state)
        
        if self.max_features is None:
            feature_indices = np.arrange(n_features)
        elif isinstance(self.max_features, str) and self.max_features == 'sqrt':
            k = int(np.sqrt(n_features))
            feature_indices = rng.choice(n_features, size=k, replace=False)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
            features_indices = rng.choice(n_features, size = k, replace = False)
        else: 
            raise ValueError(f"Invalid max_features value: {self.max_features}")

        best_gain = -1.0
        best_feature_idx = None
        best_threshold = None
        
        y_parent = y

        for feat_idx in range(n_features):
            unique_values = np.unique(X[:, feat_idx])
            
            for threshold in unique_values:
                left_mask = X[:, feat_idx] <= threshold
                y_left = y[left_mask]
                y_right = y[~left_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Use the centralized information_gain helper
                gain = information_gain(y_parent, y_left, y_right, metric=self.criterion)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feat_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively builds the decision tree (CART algorithm).
        """
        # Base Cases
        if (len(np.unique(y)) == 1 or 
            (self.max_depth is not None and depth >= self.max_depth) or
            len(y) < self.min_samples_split):
            
            return Node(value=self._get_leaf_value(y))

        feat_idx, threshold, gain = self._find_best_split(X, y)

        if gain <= 0:
            return Node(value=self._get_leaf_value(y))

        left_mask = X[:, feat_idx] <= threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature=feat_idx, threshold=threshold, left=left_child, right=right_child)
    
    def _traverse_tree_and_predict(self, x: np.ndarray, node: Node) -> Any:
        """
        Recursively traverses the trained decision tree for a single sample 'x'.
        """
        if node.is_leaf():
            return node.value
        
        # Check if the feature value is less than or equal to the split threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree_and_predict(x, node.left)
        else:
            return self._traverse_tree_and_predict(x, node.right)
    

    # --- Public API ---

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTreeClassifier":
        """
        Builds the Decision Tree from the training data.
        """
        X_arr = ensure_2d_numeric(X, name="X")
        y_arr = ensure_1d_vector(y, name="y")
        check_Xy_shapes(X_arr, y_arr)
        
        self.n_features_ = X_arr.shape[1]
        self.tree_ = self._build_tree(X_arr, y_arr, depth=0)
        self
        
    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            The predicted class labels.
        """
        X_arr = ensure_2d_numeric(X, name="X")

        if self.tree_ is None:
            raise RuntimeError("Model must be trained before calling predict.")
        
        # Iterate over all samples in X_arr and use the helper function 
        # to find the prediction for each one.
        y_pred = np.array([self._traverse_tree_and_predict(sample, self.tree_) for sample in X_arr])
        
        return y_pred