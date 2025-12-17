# K-Nearest Neighbors

The K-Nearest Neighbors (KNN) algorithm is non-parametric, lazy learning method used for both classification and regression. It is one of the simplest supervised learning algorithm.

## Algorithm Overview

The core idea of KNN is to classify or predict the values of a new data point based on the labels or values of its *k* closest neighbors in the feature space.

### Classification

1. **Objective:** To predict the class label of a new data point.
2. **Process**
    * Calculate the distance (usually Euclidean) betwween the new point and all points in the training set.
    * Identify the *k* sample with the smallest distances (the "nearest neighbor").
    * The predicted class for the new point is the **mode** (most frequent class) among those *k* neighbors.

### Regression

1. **Objective:** To predict a continuous target value for a new data point.
2. **Process:**
    * The steps are identical to classification, up to finding the *k* nearest neighbors.
    * The predicted value for the new point is the **mean** (average) of the target values of those *k* neighbors.

## Key Hyperparameters

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `k` | **The number of neighbors** to consider when making a prediction. This is the most crucial parameter. | Odd numbers like 3, 5, 7. |
| `distance_metric` | The function used to measure distance between points (e.g., `'euclidean'`, `'manhattan'`). | `'euclidean'` |

---

## Data Requirements

### Input Features ($\mathbf{X}$)

* **Format:** Requires a 2D array (shape: $(N_{samples}, N_{features})$).
* **Type:** Features must be entirely **numeric**. KNN is highly sensitive to the scale of features, so **standardization (scaling)** of the input data is strongly recommended before training.

### Labels ($\mathbf{Y}$)

* **Classification:** Discrete integers or strings.
* **Regression:** Continuous floating-point numbers.
