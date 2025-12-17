# Logistic Regression

This package implements the Logistic Regression model, which is a fundamental algorithm for **binary classification**. Despite its name, Logistic Regression is a classification algorithm, not a regression one.

## Algorithm Overview

Logistic Regression models the **probability** that a given input sample belongs to a particular class (Class 1 or Class 0).

1. **Net Input (Linear Model):** The process starts with a linear combination of features, similar to Linear Regression: $Z = \mathbf{X} \mathbf{W} + b$.
2. **Sigmoid Function:** The linear output $Z$ is then passed through the **Sigmoid function** ($\sigma$), which squashes the result into the range $(0, 1)$, producing the probability estimate $\hat{p}$:

    $\hat{p} = \sigma(Z) = \frac{1}{1 + e^{-Z}}$
3. **Cost Function:** The weights ($\mathbf{W}$) are optimized by minimizing the **Binary Cross-Entropy Loss** (or Log Loss) using **Batch Gradient Descent (GD)**.

### Optimization

The model uses **Batch Gradient Descent** to iteratively update the weights and bias. This process involves calculating the average gradient across the entire training batch ($\mathbf{X}$) at each step.

* **Learning Rate (`eta`):** Determines the step size in the direction of steepest descent. A proper learning rate is crucial for convergence.
* **Epochs:** The total number of iterations over the training set.

## Key Hyperparameters

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `eta` ($\eta$) | `float` | **Learning Rate** for Gradient Descent. | `0.01` |
| `epochs` | `int` | **Maximum iterations** (passes over the full dataset). | `100` |
| `random_state` | `int` | Seed for initializing weights (reproducibility). | `None` |

---

## Data Requirements and Preprocessing

Logistic Regression relies on Gradient Descent for fitting, which imposes strict requirements on data preparation:

### 1. Target Variables ($\mathbf{Y}$)

* **Format:** Must be strictly **binary**, encoded as **integers 0 and 1**. The model will raise an error if any other labels are found.

### 2. Feature Scaling ($\mathbf{X}$)

* **Requirement:** **Feature scaling (Standardization or Normalization)** is **mandatory** for Gradient Descent. Scaling the features prevents large input ranges from causing the gradient to explode and ensures the model converges efficiently to the optimal solution.
