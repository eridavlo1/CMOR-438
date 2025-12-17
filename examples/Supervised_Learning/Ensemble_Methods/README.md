# Ensemble Methods (Bagging and Random Forests)

This package contains implementations of ensemble learning techniques, which combine multiple base models (estimators) to improve overall performance, robustness, and generalization capability.

## Core Algorithms

We focus on **Bootstrap Aggregating (Bagging)** and its most famous application, the **Random Forest**.

### 1. Bootstrap Aggregating (Bagging)

Bagging trains multiple instances of the same base estimator (often a decision tree) on different subsets of the training data, where each subset is created by **sampling with replacement** (bootstrapping).

* **Objective:** Reduce the variance of high-variance models (like deep decision trees) by averaging out their predictions.
* **Prediction:** Predictions are aggregated by **Hard Voting** (for classification) or **Averaging** (for regression).

### 2. Random Forest

The Random Forest is a specialized form of Bagging that uses Decision Trees as base estimators and adds an extra layer of randomness.

* **Added Randomness:** At each node split, the algorithm only considers a random subset of features (`max_features`) rather than all features.
* **Objective:** Decouple the individual trees, forcing them to be less correlated and thus further improving the robustness of the ensemble.

## Key Hyperparameters

The following critical parameters are shared across `Bagging*` and `RandomForest*` estimators:

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `n_estimators` | The number of base estimators (trees) in the ensemble. | 50 to 200 |
| `max_depth` | The maximum depth of the individual decision trees. | 10 to None (no limit) |
| `max_features` | The number of random features to consider at each split. | `'sqrt'` (for classification), `'log2'` (for regression) |
| `random_state` | Seed for reproducibility of the bootstrapping process. | Any integer |

---

# Data Requirements

The ensemble methods rely entirely on the data requirements of their base estimator, the Decision Tree.

### Input Features ($\mathbf{X}$)

* **Format:** Requires a 2D array or similar structure, where rows are samples and columns are feature\
    shape: $(N_{samples}, N_{features})$.
* **Type:** Features must be entirely **numeric** (float or integer) as ensemble methods use thresholding for splitting. Categorical features must be one-hot encoded or label encoded prior to use.

### Labels ($\mathbf{Y}$)

* **Classification:** Labels must be discrete integers or strings (e.g., $[0, 1, 2]$).
* **Regression:** Labels must be continuous floating-point numbers.

### Preprocessing

The primary preprocessing required for these algorithms is standard data validation and splitting:

1. **Splitting:** Use `train_test_split` to separate the data into training and testing sets.
2. **Validation:** Input arrays must be validated using `ensure_2d_numeric`, `ensure_1d_vector`, and `check_Xy_shapes` (handled internally by the `fit` method).
