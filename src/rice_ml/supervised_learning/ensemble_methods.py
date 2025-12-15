import numpy as np
from typing import List, Any, Optional, Union, Sequence
from collections import Counter
from .decision_tree import DecisionTree
import copy



ArrayLike = Union[np.ndarray, Sequence[Any]]

# --- Internal Helper for Validation and Bootstrap ---

def _ensure_2d_numeric(X: ArrayLike) -> np.ndarray:
    """Ensures X is a 2D numeric NumPy array (simplified)."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError("Input must be a 1D or 2D array.")
    return X_arr

def _get_bootstrap_indices(n_samples: int, random_state: Optional[int]) -> np.ndarray:
    """Generates indices for a bootstrap sample (sampling with replacement)."""
    rng = np.random.default_rng(random_state)
    return rng.choice(n_samples, size=n_samples, replace=True)

# ----- 1. Hard Voting Classifier --------

class HardVotingClassifier:
    """
    Implements a Hard Voting classifier (majority vote).

    Parameters
    ----------
    estimators : List[Any]
        A list of fitted or unfitted estimator objects. They must support the `predict` method.
    """
    def __init__(self, estimators: List[Any]):
        self.estimators = estimators
        self.classes_ = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "HardVotingClassifier":
        """
        Fits all base estimators on the provided data.
        """
        X_arr = _ensure_2d_numeric(X)
        y_arr = np.asarray(y)
        
        self.classes_ = np.unique(y_arr)

        for estimator in self.estimators:
            estimator.fit(X_arr, y_arr)
        
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class label based on the majority vote from all estimators.
        """
        X_arr = _ensure_2d_numeric(X)
        
        if not self.estimators:
            raise RuntimeError("Estimators list is empty.")
            
        # Collect predictions from all estimators
        all_preds = []
        for estimator in self.estimators:
            all_preds.append(estimator.predict(X_arr))
            
        # Stack predictions: shape (n_estimators, n_samples)
        pred_matrix = np.array(all_preds).T 

        # Perform majority vote (mode) for each sample
        final_predictions = np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], 
            axis=1, 
            arr=pred_matrix
        )
        return final_predictions


# ----------------------------- 2. Bagging Classifier -----------------------------

class BaggingClassifier:
    """
    Implements a Bagging (Bootstrap Aggregating) classifier.

    Trains multiple base estimators on random subsets of the training data
    (sampled with replacement) and aggregates their predictions via majority vote.

    Parameters
    ----------
    base_estimator : Any
        The estimator object to be cloned and trained (e.g., DecisionTree).
    n_estimators : int
        The number of base estimators (trees) in the ensemble. Default is 10.
    random_state : Optional[int]
        Controls the randomness of the bootstrapping.
    """
    def __init__(self, base_estimator: Any = DecisionTree(max_depth=None), 
                 n_estimators: int = 10, random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: List[Any] = []
        self.classes_ = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BaggingClassifier":
        """
        Fits the ensemble by training each estimator on a bootstrap sample.
        """
        X_arr = _ensure_2d_numeric(X)
        y_arr = np.asarray(y)
        n_samples = X_arr.shape[0]
        self.classes_ = np.unique(y_arr)
        
        self.estimators_ = []
        
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 10000, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            # Create a deep copy of the base estimator
            estimator = copy.deepcopy(self.base_estimator)
            
            # 1. Generate bootstrap indices
            # Use a unique seed for each estimator based on the master random_state
            bootstrap_indices = _get_bootstrap_indices(n_samples, int(seeds[i]))
            
            # 2. Sample data
            X_sample, y_sample = X_arr[bootstrap_indices], y_arr[bootstrap_indices]
            
            # 3. Train the estimator
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
            
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class label using the majority vote from all estimators.
        """
        X_arr = _ensure_2d_numeric(X)
        
        # Collect predictions from all estimators
        all_preds = np.array([est.predict(X_arr) for est in self.estimators_]).T
        
        # Perform majority vote (mode) for each sample
        final_predictions = np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], 
            axis=1, 
            arr=all_preds
        )
        return final_predictions


# ----------------------------- 3. Random Forest Classifier -----------------------------

class RandomForestClassifier(BaggingClassifier):
    """
    Implements a Random Forest classifier (specialized Bagging).

    Random Forests introduce randomness at the feature level: each tree is 
    trained on a bootstrap sample, and at each split, it only considers a 
    random subset of features (max_features) for finding the best split.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest. Default is 100.
    max_depth : Optional[int]
        The maximum depth of each tree.
    max_features : Union[str, float, int]
        The number of features to consider when looking for the best split. 
        - If 'sqrt' (default): max_features = sqrt(n_features)
        - If 'log2': max_features = log2(n_features)
        - If int: max_features is the absolute number.
        - If float (0.0 to 1.0): max_features is the fraction.
    random_state : Optional[int]
        Controls the randomness of the bootstrapping and feature selection.
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 max_features: Union[str, float, int] = 'sqrt', random_state: Optional[int] = None):
        
        # Random Forests use the DecisionTree as the base estimator.
        # We assume DecisionTree supports the 'max_features' parameter from our earlier implementation
        # (even if the internal tree logic isn't fully implemented in the placeholder).
        base_estimator = DecisionTree(max_depth=max_depth, random_state=random_state, max_features=max_features)
        
        # BaggingClassifier is the superclass, handling the bootstrapping and aggregation.
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )
        # Store attributes for transparency
        self.max_features = max_features
        self.max_depth = max_depth

    # The fit and predict methods are inherited directly from BaggingClassifier.
    # The feature randomness is handled internally by the DecisionTree object 
    # during its own fit process (as defined by max_features). 

# --- Example Usage and Testing ---

if __name__ == '__main__':
    print("--- Testing Ensemble Methods ---")
    
    # Simple XOR-like data (often hard for single linear models)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                  [0.5, 0.5], [0.6, 0.4], [0.4, 0.6], [1.5, 1.5],
                  [0, 2], [2, 0]])
    y = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0]) # Non-linear labels

    # 1. Hard Voting Test
    print("\n--- 1. Hard Voting Classifier ---")
    
    # Placeholder: Assuming we have three different types of classifiers
    class DummyClf1(DecisionTree): 
        def predict(self, X): return np.where(X[:, 0] > 0.5, 1, 0) # Split on X1
    class DummyClf2(DecisionTree): 
        def predict(self, X): return np.where(X[:, 1] > 0.5, 1, 0) # Split on X2
    class DummyClf3(DecisionTree): 
        def predict(self, X): return np.where(X[:, 0] + X[:, 1] > 1.5, 0, 1) # Diagonal split

    voter = HardVotingClassifier(estimators=[DummyClf1(), DummyClf2(), DummyClf3()])
    voter.fit(X, y)
    y_pred_vote = voter.predict(X)
    
    accuracy_vote = np.mean(y_pred_vote == y)
    print(f"Hard Voting Accuracy: {accuracy_vote:.2f}")

    # 2. Bagging Classifier Test
    print("\n--- 2. Bagging Classifier (Decision Trees) ---")
    # Bagging helps reduce variance of unstable models like Decision Trees
    bagging_clf = BaggingClassifier(
        base_estimator=DecisionTree(max_depth=2), 
        n_estimators=5, 
        random_state=42
    )
    # NOTE: Actual accuracy will depend entirely on the DecisionTree placeholder implementation.
    bagging_clf.fit(X, y)
    y_pred_bag = bagging_clf.predict(X)
    
    accuracy_bag = np.mean(y_pred_bag == y)
    print(f"Bagging (5 Trees) Accuracy: {accuracy_bag:.2f}")

    # 3. Random Forest Classifier Test
    print("\n--- 3. Random Forest Classifier ---")
    # max_features='sqrt' (standard RF setting) is used by the base tree
    rf_clf = RandomForestClassifier(n_estimators=10, max_depth=3, max_features='sqrt', random_state=42)
    rf_clf.fit(X, y)
    y_pred_rf = rf_clf.predict(X)

    accuracy_rf = np.mean(y_pred_rf == y)
    print(f"Random Forest (10 Trees) Accuracy: {accuracy_rf:.2f}")