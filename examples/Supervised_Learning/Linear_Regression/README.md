# Linear Regression

This package implements a comprehensive Linear Regression model capable of solving the optimization problem using three distinct methods: Ordinary Least Squares (OLS), Ridge Regression, and Gradient Descent (GD).

## Algorithm Overview

Linear Regression models the relationship between a scalar dependent variable ($y$) and one or more explanatory variables ($X$). The prediction is a linear combination of the input features:

$$
\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

The goal of the training process is to find the optimal set of weights ($\mathbf{w}$) and bias ($b$) that minimize the cost function, typically the Mean Squared Error (MSE).

### Optimization Methods

| Method | Core Idea | When to Use |
| :--- | :--- | :--- |
| **OLS (Ordinary Least Squares)** | **Closed-form Solution** (Analytical). Computes the exact minimum using linear algebra (the Normal Equation). | For small to medium datasets ($N < 100,000$) where speed and accuracy are crucial. |
| **Ridge Regression** | **Closed-form Solution with L2 Regularization**. Adds a penalty ($\alpha$) to the cost function to shrink feature weights, preventing overfitting and managing collinearity. | When dealing with correlated features or high-dimensional data prone to overfitting. |
| **GD (Gradient Descent)** | **Iterative Solution**. Starts with random weights and repeatedly takes steps in the direction opposite to the gradient of the cost function until convergence. | For very large datasets that cannot be handled by matrix inversion (OLS/Ridge) due to memory or computational constraints.  |

## Key Hyperparameters

| Parameter | Used By | Description | Default |
| :--- | :--- | :--- | :--- |
| `method` | All | Specifies the optimization technique. | `'ols'` |
| `alpha` ($\lambda$) | Ridge | **Regularization strength.** Larger values penalize weights more heavily. | `0.0` |
| `eta` ($\eta$) | GD | **Learning Rate.** Controls the size of the step taken during each iteration. Must be tuned carefully. | `0.01` |
| `epochs` | GD | **Number of Iterations.** The maximum number of times the model updates the weights. | `1000` |

---

## Data Requirements

### Scaling Pre-requisite

If you use the **Gradient Descent (`gd`)** method, **feature scaling (Standardization or Normalization)** of the input data ($\mathbf{X}$) is **mandatory**. Gradient Descent converges much faster and more reliably when features are on a similar scale.

### Input/Output

* **Features ($\mathbf{X}$):** Must be a 2D numeric array (shape: $(N_{samples}, N_{features})$).
* **Targets ($\mathbf{Y}$):** Must be a 1D numeric array of continuous values.