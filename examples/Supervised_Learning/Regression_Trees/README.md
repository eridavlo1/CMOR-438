# Decision Tree Regressor (CART)

This package implements the **DecisionTreeRegressor**, a supervised learning model used to predict continuous target variables. It is based on the **CART (Classification and Regression Tree)** algorithm.

## Algorithm Overview

A Regression Tree recursively partitions the input feature space into smaller regions. For any new data point, the prediction is simply the **mean** target value of all training samples that fell into the final region (leaf node) containing that point.

### Splitting Criterion: Variance Reduction

Unlike classification trees which use measures like Gini Impurity or Information Gain, the Regression Tree uses a measure of variance to decide the best split.

The algorithm chooses the split (feature and threshold) that results in the **greatest reduction in variance** across the resulting child nodes. This is mathematically equivalent to maximizing the reduction in the **Mean Squared Error (MSE)**.

$$
\text{Variance Reduction} = \text{Variance}(Y_{\text{parent}}) - \sum_{i \in \{\text{L, R}\}} \frac{N_i}{N_{\text{total}}} \times \text{Variance}(Y_i)
$$

### Prediction Mechanism

The prediction for a data point $x$ is the constant value $y$ assigned to the leaf node where $x$ lands:

$$
\hat{y}(x) = \text{Mean}(Y) \quad \text{for all } y \in \text{Leaf Region}
$$

## Key Hyperparameters

| Parameter | Type | Description | Effect on Model |
| :--- | :--- | :--- | :--- |
| `max_depth` | `Optional[int]` | The maximum allowed depth of the tree. | Controls complexity and helps prevent **overfitting**. |
| `min_samples_split` | `int` | The minimum number of samples a node must contain to be considered for splitting. | Prevents splitting nodes with too little data, acting as a regularization technique. |
| `max_features` | `str, float, int` | The number of random features to consider at each split. | Crucial for **Random Forests**; introduces randomness and reduces correlation between trees. |
| `criterion` | `{'mse'}` | The function used to measure the quality of a split (currently, Mean Squared Error). | Defines the "impurity" metric to be minimized. |

## Data Requirements and Preprocessing

* **Features ($\mathbf{X}$):** Must be a 2D numeric array.
* **Target ($\mathbf{y}$):** Must be a 1D numeric array (continuous values).
* **Scaling:** **Feature scaling is generally NOT required for tree-based models** because they rely on thresholds (`x[i] <= threshold`) rather than distance metrics. However, scaling may be helpful for interpretation or use in ensemble methods like Gradient Boosting.
