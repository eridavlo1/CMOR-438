# Principal Component Analysis (PCA)

This package implements the **Principal Component Analysis (PCA)** algorithm, a powerful **Unsupervised Linear Dimensionality Reduction** technique. PCA is used to transform a high-dimensional dataset into a new set of dimensions (principal components) that are uncorrelated and ordered by the amount of variance they explain.

## Algorithm Overview

The primary goal of PCA is to project the data onto a lower-dimensional subspace while preserving the maximum amount of information (variance). .

### Core Steps

1. **Centering:** The data is centered by subtracting the mean of each feature (column).
2. **Covariance:** The covariance matrix is calculated to understand the relationships between the features.
3. **Eigenvalue Decomposition:** Eigenvalue decomposition is performed on the covariance matrix to obtain:
    * **Eigenvectors:** These are the principal components (the new axes).
    * **Eigenvalues:** These represent the amount of variance explained by each corresponding component.
4. **Selection:** Components are selected based on the user-defined criteria (`n_components`).

## Key Hyperparameters

| Parameter | Type | Description | Importance |
| :--- | :--- | :--- | :--- |
| `n_components` | `int`, `float`, or `None` | **Target Dimension.** Controls how many principal components are kept: | Critical for controlling the trade-off between dimensionality reduction and information loss. |
| | `int > 0` | Keeps this exact number of components. | |
| | `0 < float < 1` | Keeps enough components to explain this ratio of the total variance (e.g., 0.95). | |
| | `None` | Keeps all components ($\min(N, D)$). | |

## Data Requirements and Preprocessing

PCA is a distance-based method, highly sensitive to feature scales.

* **Features ($\mathbf{X}$):** Must be a 2D numeric array.
* **Scaling (Mandatory):** **Feature Scaling (Standardization)** is mandatory *before* running PCA. Without standardization, features with larger initial variance will artificially dominate the first principal components, leading to skewed results.
