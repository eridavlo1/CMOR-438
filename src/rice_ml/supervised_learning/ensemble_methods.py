import numpy as np
from typing import List, Any, Optional, Union, Sequence, Callable, Dict, Tuple, Literal
from collections import Counter
import warnings
import copy
from rice_ml.utils import ArrayLike, ensure_2d_numeric, ensure_1d_vector, check_Xy_shapes
from .decision_trees import DecisionTreeClassifier 
from .regression_trees import DecisionTreeRegressor 

# --- Internal Helper for Aggregation  ---


def _aggregate_predictions(y_preds: np.ndarray, mode: str) -> np.ndarray:
    """Aggregates predictions (rows) from multiple models (columns)."""
    
    if mode == 'hard_vote':
        # Used by Classifier: find the mode (most frequent) label for each sample
        
        # Use Counter to find the mode and handle ties by choosing the smallest label
        def get_mode(row):
            counts = Counter(row)
            return min([key for key, value in counts.items() if value == max(counts.values())])
        
        return np.array([get_mode(y_preds[i, :]) for i in range(y_preds.shape[0])])
    
    elif mode == 'average':
        # Used by Regressor: calculate the mean prediction for each sample
        return np.mean(y_preds, axis=1)
    
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

# --- Internal Helper for Bootstrap Indices ---
def _get_bootstrap_indices(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generates indices for a bootstrap sample (sampling with replacement)."""
    return rng.choice(n_samples, size=n_samples, replace=True)


# ----- 1. Hard Voting Classifier -----

class HardVotingClassifier:
    """
    Implements a Hard Voting classifier (majority vote). 

    Parameters
    ----------
    estimators : List[Any]
        A list of fitted or unfitted estimator objects.
    """
    def __init__(self, estimators: List[Any]):
        self.estimators = estimators
        self.classes_ = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "HardVotingClassifier":
        """
        Fits all base estimators on the provided data.
        """
        X_arr = ensure_2d_numeric(X)
        y_arr = ensure_1d_vector(y)
        check_Xy_shapes(X_arr, y_arr)
        
        self.classes_ = np.unique(y_arr)

        for estimator in self.estimators:
            estimator.fit(X_arr, y_arr)
        
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class label based on the majority vote from all estimators.
        """
        X_arr = ensure_2d_numeric(X)
        
        if not self.estimators:
            raise RuntimeError("Estimators list is empty.")
            
        # Collect predictions and stack: shape (n_samples, n_estimators)
        all_preds = np.array([est.predict(X_arr) for est in self.estimators]).T
        
        # Use the centralized aggregation function
        return _aggregate_predictions(all_preds, mode='hard_vote')


# ----- 2. Bagging Classifier ------

class BaggingClassifier:
    """
    Implements a Bagging (Bootstrap Aggregating) classifier. 
    
    Parameters
    ----------
    base_estimator : Any
        The base model to be cloned (default: DecisionTreeClassifier).
    n_estimators : int, default=10
        The number of base estimators.
    random_state : Optional[int], default=None
        Controls the randomness of the bootstrapping.
    """
    def __init__(self, base_estimator: Any = DecisionTreeClassifier(), 
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
        X_arr = ensure_2d_numeric(X)
        y_arr = ensure_1d_vector(y)
        check_Xy_shapes(X_arr, y_arr)
        n_samples = X_arr.shape[0]
        self.classes_ = np.unique(y_arr)
        
        self.estimators_ = []
        rng = np.random.default_rng(self.random_state)
        
        for i in range(self.n_estimators):
            # Create a deep copy of the base estimator
            estimator = copy.deepcopy(self.base_estimator)
            
            # 1. Generate bootstrap indices
            bootstrap_indices = _get_bootstrap_indices(n_samples, rng)
            
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
        X_arr = ensure_2d_numeric(X)
        
        # Collect predictions from all estimators and stack: shape (n_samples, n_estimators)
        all_preds = np.array([est.predict(X_arr) for est in self.estimators_]).T
        
        # Use the centralized aggregation function
        return _aggregate_predictions(all_preds, mode='hard_vote')


# ----- 3. Random Forest Classifier -----

class RandomForestClassifier(BaggingClassifier):
    """
    Implements a Random Forest classifier (specialized Bagging). 

    Random Forests combine bootstrapping (from Bagging) with random feature selection
    at each tree split (handled by the base estimator).
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : Optional[int], default=None
        The maximum depth of each tree.
    max_features : Union[str, float, int], default='sqrt'
        The number of features to consider when looking for the best split. 
    random_state : Optional[int], default=None
        Controls the randomness.
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 max_features: Union[str, float, int] = 'sqrt', random_state: Optional[int] = None):
        
        # 1. Initialize the base estimator (Decision Tree Classifier)
        base_estimator = DecisionTreeClassifier(
            max_depth=max_depth, 
            random_state=random_state, 
            max_features=max_features # Assumes DecisionTreeClassifier supports this
        )
        
        # 2. Call the BaggingClassifier constructor to handle the ensemble logic
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.max_features = max_features
        self.max_depth = max_depth

    # fit and predict methods are inherited from BaggingClassifier.
    # The feature randomness is applied internally by DecisionTreeClassifier during fit.