# The Source Code for Rice ML# rice_ml Source Code Structure

This directory (`src`) contains the core Python implementation for the entire `rice_ml` machine learning library. The code is organized by machine learning paradigm and core utility functions.

## High-Level Organization

The `rice_ml` library is structured to keep model implementations separate from utility and data processing functions.

| Directory | Content Type | Primary Focus |
| :--- | :--- | :--- |
| `supervised_learning` | Model Implementations | Classifiers (Perceptron, MLP, etc.) and Regressors (Linear Regression, Trees). |
| `unsupervised_learning` | Model Implementations | Clustering (K-Means, DBSCAN) and Dimensionality Reduction (PCA). |
| `processing` | Utility Functions | Data preprocessing (scaling, splitting) and post-training evaluation (metrics). |
| `utils` | Internal Helpers | Validation, distance metrics, and core mathematical functions used by models. |

---

## Core Modules

### 1. processing

Houses all functions related to data preparation and evaluation, ensuring consistency and preventing data leakage across all models.

* `pre_processing.py`: Contains feature scaling functions (`standardize`, `minmax_scale`) and data splitting logic (`train_test_split`, `train_val_test_split`).
  * **Note:** The `standardize` function is implemented to support both **fitting** (learning parameters from the training set) and **transforming** (applying those parameters to the test set), adhering to best practices.
* `post_processing.py`: Provides all post-training evaluation metrics for both classification (`accuracy_score`, `f1_score`, `confusion_matrix`) and regression (`r2_score`, `mse`, `mae`).

### 2. utils

Internal functions designed to support the models and ensure data integrity.

* `validation.py`: Contains helpers like `_ensure_2d_numeric` and checks for NaN/Inf values, ensuring models receive clean inputs.
* `distances_metrics.py`: Contains common distance calculations (`euclidean_distance`, `manhattan_distance`) used by distance-based algorithms like KNN and DBSCAN.
* `_tree_helpers.py`: Contains specific impurity measures (`variance`, `information_gain`) used by the Decision Tree implementations.

### 3. supervised_learning

Implements all models that require labeled data for training.

* `linear_regression.py`: Implements OLS, Ridge, and Gradient Descent fitting methods.
* `logistic_regression.py`: Implements binary logistic regression using Gradient Descent.
* `perceptron.py`: Implements the classic single-layer Perceptron model.
* `multilayer_perceptron.py`: Implements the MLP (Neural Network) for binary classification using backpropagation.
* `k_nearest_neighbors.py`: Implements both `KNNClassifier` and `KNNRegressor` (Lazy Learning models).
* `decision_trees.py`: Implements `DecisionTreeRegressor` (and implicitly, the classifier counterpart using Gini/Entropy).

### 4. unsupervised_learning

Implements all models that learn structure from unlabeled data.

* `k_means_clustering.py`: Implements K-Means with support for 'random' and 'k-means++' initialization.
* `dbscan.py`: Implements DBSCAN for density-based clustering and outlier detection.
* `pca.py`: Implements Principal Component Analysis for dimensionality reduction based on eigenvalue decomposition.
* `community_detection.py`: Implements graph-based algorithms like Label Propagation for community detection.

---

## Usage Guidelines

When developing within the `src` folder:

1. **Dependency:** Models in `supervised_learning` and `unsupervised_learning` should only import validation and helper functions from `utils`, and should not directly import other models.
2. **API Consistency:** All external methods (`fit`, `predict`, `transform`, `score`) should maintain consistency with the established **scikit-learn API pattern**
