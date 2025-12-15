import numpy as np
class Node:
    """A node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node (true side of the split)
        self.right = right              # Right child node (false side of the split)
        # parameter for leaf node
        self.value = value              # Class label for leaf nodes
    def is_leaf(self):
        return self.value is not None
def __gini_impurity(y):
    """Calculate the Gini impurity for a list of class labels.
    Gini impurity: G = 1 - sum(p_i^2) for each class i
     """   
     # if no samples, impurity is 0
    if len(y) == 0:
        return 0
    # get unique class counts
    __, counts = np.unique(y, return_counts=True)
    
    # calculate probabilities
    probabilities = counts / len(y)
    
    # calculate gini impurity
    gini = 1 - np.sum(probabilities ** 2)
    return gini

class DecisionTree:
    """
    A simple CART-style decision tree  classifier implemented from scratch.
    Uses Gini impurity as the splitting criterion.
    """
    def __init__ (self, max_depth=None, min_samples_split=2):
        """
        Initializes the Decision Tree classifier.
        Parameters:
        -----------
        max_depth: int, option
            The maximum depth of the tree. If None, the tree is grown until all leaves are pure or until min_samples_split is reached.
        min_samples_split: int, optional
            The minimum number of samples required to split an internal node.
            Must be greater than or equal to 2. Default is 2.
        random_state: int, optional
            Seed for the random number generator for reproducibility.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = None
        
     # Core Splitting Logic   
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the dataset (X, y)
        based on the lowest weighted Gini impurity (highest Information Gain).
        """
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = X.shape

        # Initial impurity (parent node)
        parent_gini = __gini_impurity(y)

        # Iterate over all features
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            # Potential thresholds are the unique values of the feature
            # A common simplification is to use the midpoint between sorted unique values
            thresholds = np.unique(X_column)
            
            # Iterate over all possible thresholds for the current feature
            for threshold in thresholds:
                # 1. Split the data
                left_indices = X_column <= threshold
                right_indices = X_column > threshold

                # Skip splits that result in empty children
                if np.sum(left_indices) < 1 or np.sum(right_indices) < 1:
                    continue

                # Get labels for the children
                y_left = y[left_indices]
                y_right = y[right_indices]

                # 2. Calculate Weighted Gini Impurity (Child Impurity)
                n_left, n_right = len(y_left), len(y_right)
                n_total = n_samples

                gini_left = __gini_impurity(y_left)
                gini_right = __gini_impurity(y_right)

                # Weighted Gini of the split
                gini_child = (n_left / n_total) * gini_left + \
                             (n_right / n_total) * gini_right

                # 3. Calculate Information Gain
                information_gain = parent_gini - gini_child

                # 4. Update the best split
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold
    
    # ---- Recurvsive Tree Building ----
    def __build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # --- Check Stopping Criteria (Base Cases) ---
        # 1. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
        
        # 2. Only one class left (pure node)
        if n_labels == 1:
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
        
        # 3. Minimum samples to split not met
        if n_samples < self.min_samples_split:
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
        
        # --- Find and apply the best split ---
        feature_idx, threshold = self.__best_split(X, y)
        # If no split improves the gain (e.g., best gain = -1 if features are complex/bad)
        if feature_idx is None:
            leaf_value = self.__most_common_label(y)
            return Node(value=leaf_value)
        
        # Get indices for the split
        X_column = X[:, feature_idx]
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        
        # Recursive calls for children
        left_child = self.__build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self.__build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=feature_idx, threshold=threshold, left=left_child, right=right_child)
    
    ### Helper Method to find the most common label in a list
    def __most_common_label(self, y):
        """Return the most frequent class label in the array y"""
        if len(y) == 0:
            return None
        # np.unique returns unique labels and their counts
        values, counts = np.unique(y, return_counts=True)
        # argmax returns the index of the maximum count, and use that index
        # to get the corresponding value (label)
        return values[np.argmax(counts)]
    
    ### --- Public API Methods ---
    
    def fit(self, X, y):
        """
        Builds the decision tree classifier from the training data (X, y).
        """
        self.n_features = X.shape[1]
        self.root = self.__build_tree(X, y)
        return self # Allows for chaining (e.g, tree.fit(X, y).predict(X_test))
    
    def __traverse_tree(self, x, node):
        """
        Recursively traverse the tree for a single data point x.
        """
        # Base case: if leaf node, return the predicted class value
        if node.is_leaf():
            return node.value
        # Get the feature value for the split
        feature_value = x[node.feature_idx]
        
        # Decide to go left or right based on the feature and threshold
        if feature_value <= node.threshold:
            return self.__traverse_tree(x, node.left)
        else:
            return self.__traverse_tree(x, node.right)
    def predict(self, X):
        """
        Predicts class labels for the input data X.
        
        Parameters:
        ----------
        X: np.ndarray
            The input data to predict on.
            
        Returns:
        -------
        np.ndarray
            Predicted class labels for each input sample.
        """
        ### Apply the transversal function to every row in X.
        predictions = np.array([self.__traverse_tree(x, self.root) for x in X])
        return predictions
    
