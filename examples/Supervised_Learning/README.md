# Decision Tree Regressor

This package provides an implementation of the Decision Tree Regressor using the CART (Classification and Regression Tree) algorithm. Unlike classification trees that predict a category, regression trees predict a continuous numerical value.

## Algorithm Overview

A regression tree works by recursively splitting the data into subsets such that the samples within each resulting node become increasingly homogeneous with respect to the target value ($y$).

### Splitting Criterion: Variance Reduction

The core mechanism for finding the best split is minimizing the error (or disorder) in the child nodes. For regression, this is achieved by minimizing the **Variance** within each node.

* **Metric:** The model uses **Variance Reduction** (based on Mean Squared Error, or MSE) as the impurity measure. 
* **Process:** At every node, the model evaluates potential splits across all features and selects the split that results in the largest drop in the total weighted variance of the resulting child nodes.
* **Prediction:** When a sample reaches a **leaf node**, the final predicted value is the **mean** of all target values of the training samples contained in that leaf.

## Key Hyperparameters

| Parameter | Description | Relevance |
| :--- | :--- | :--- |
| `max_depth` | The maximum depth the tree is allowed to grow. Controls overfitting; smaller values lead to simpler models. | Pruning |
| `min_samples_split` | The minimum number of samples a node must contain to attempt a split. Prevents splits on tiny subsets. | Pruning |
| `max_features` | The number of random features to consider at each split. Primarily used to enable **Random Forest** algorithms (via ensemble methods). | Randomness |
| `criterion` | The function to measure the quality of a split. Currently only `'mse'` (Variance) is implemented. | Splitting |
| `random_state` | Seed for reproducibility, especially when `max_features` is used. | Reproducibility |

---

## Data Requirements

The Decision Tree Regressor accepts numeric data but does not require feature scaling, as its splits are based on thresholds rather than distance metrics.

* **Features ($\mathbf{X}$):** Must be a 2D numeric array.
* **Targets ($\mathbf{Y}$):** Must be a 1D array of continuous floating-point numbers.