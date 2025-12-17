# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This package implements the **DBSCAN** algorithm, a powerful method for **Unsupervised Clustering** that identifies groups based on the density of data points. Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand and can discover arbitrarily shaped clusters while explicitly identifying outliers (noise).

## Algorithm Overview

DBSCAN classifies every point in the dataset into one of three roles:

1. **Core Point:** A point that has at least `min_samples` within its $\epsilon$ (epsilon) radius.
2. **Border Point:** A point that is within the $\epsilon$ neighborhood of a Core Point, but does not itself satisfy the `min_samples` criterion.
3. **Noise Point (Outlier):** A point that is neither a Core Point nor a Border Point. It is assigned a special label (typically -1).

### Clustering Mechanism

A cluster is formed by starting at a random, unvisited Core Point and recursively adding all points that are **density-reachable** from it.

* **Directly Density-Reachable:** Point $p$ is directly reachable from point $q$ if $p$ is in $q$'s $\epsilon$-neighborhood and $q$ is a Core Point.
* **Density-Reachable:** A chain of directly density-reachable points exists between two points.

## Key Hyperparameters

DBSCAN's performance is highly sensitive to the correct tuning of its two core parameters:

| Parameter | Type | Description | Effect on Clustering |
| :--- | :--- | :--- | :--- |
| `eps` ($\epsilon$) | `float` | **Neighborhood Radius.** The maximum distance to look for neighboring samples. | Determines the reach of the local density measure. |
| `min_samples` | `int` | **Density Threshold.** The minimum number of points required to form a dense region (i.e., to define a Core Point). | Controls the sensitivity to noise and the minimum size of a cluster. |

---

## Data Requirements

DBSCAN is a distance-based algorithm, making it sensitive to the scale of the input features.

* **Features ($\mathbf{X}$):** Must be a 2D numeric array.
* **Scaling:** **Feature scaling (Standardization or Normalization)** is highly recommended to ensure that all dimensions contribute equally to the distance calculation.
