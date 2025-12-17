# K-Means Clustering

This package implements the **K-Means Clustering** algorithm, a classic method in **Unsupervised Learning** used to partition $N$ observations into $K$ clusters. It is a centroid-based algorithm that aims to minimize the within-cluster variance.

## Algorithm Overview

K-Means is an iterative process that alternates between two key steps (E-step and M-step) to optimize the cluster assignments:

### Core Mechanism (Expectation-Maximization)

1. **Initialization:** Select $K$ initial centroids. This implementation supports:
    * **'random':** Choosing $K$ random data points.
    * **'k-means++':** A smarter method that selects initial centroids far away from each other, leading to faster and more reliable convergence.
2. **E-step (Expectation):** Each data point is assigned to the cluster whose centroid is nearest (based on Euclidean distance).
3. **M-step (Maximization):** The centroid of each cluster is recalculated as the **mean** of all data points assigned to that cluster.
4. **Convergence:** The process repeats until the centroids no longer move significantly (change is less than `tol`) or the maximum number of iterations (`max_iter`) is reached.

### Objective Function (Inertia)

The algorithm minimizes the **Inertia** (or within-cluster sum of squares, WCSS): the sum of squared distances between each point and its assigned centroid.

$$
\text{Inertia} = \sum_{i=0}^{N} \min_{k=0}^{K} ||\mathbf{x}_i - \mathbf{\mu}_k||^2
$$

## Key Hyperparameters

| Parameter | Type | Description | Importance |
| :--- | :--- | :--- | :--- |
| `n_clusters` ($K$) | `int` | **The number of clusters to find.** Must be specified manually (a core limitation of K-Means). | Directly controls the number of groups discovered. |
| `init` | `str` | **Centroid Initialization Method.** Determines the starting points of the centroids. | Crucial for avoiding poor local minima; `'k-means++'` is generally preferred. |
| `max_iter` | `int` | Maximum number of algorithm iterations. | Controls execution time and convergence tolerance. |
| `tol` | `float` | Tolerance threshold. Stops the algorithm if the centroid movement is below this value. | Defines the level of convergence required. |

---

## Data Requirements

K-Means relies on distance calculation, making feature preparation essential.

* **Features ($\mathbf{X}$):** Must be a 2D numeric array.
* **Scaling:** **Feature scaling (Standardization or Normalization)** is **mandatory**. Without scaling, features with larger numerical ranges will dominate the Euclidean distance calculation, leading to biased clustering.
