# Supervised Learning Examples

This directory contains executable Jupyter Notebooks (`.ipynb`) demonstrating the implementation and usage of every supervised machine learning algorithm available in the `rice_ml` library.

## Goal

The notebooks in this folder serve as functional documentation, showcasing:

1. How to correctly initialize and configure each model.
2. The necessary data preprocessing steps (e.g., scaling for Gradient Descent, target encoding for Perceptron).
3. The core mechanisms (e.g., Variance Reduction in Regression Trees, Backpropagation in MLP).
4. Evaluation of performance using standard metrics (e.g., Accuracy, $R^2$).

---

## Available Model Examples

### 1. Decision Trees

| Directory | Model Type | Description |
| :--- | :--- | :--- |
| `Decision_Trees` | **Decision Tree Classifier** | Demonstrates classification using the CART algorithm with impurity criteria (Gini or Entropy). The notebook showcases how the model finds discrete decision boundaries. |
| `Regression_Trees` | **Decision Tree Regressor** | Demonstrates regression using the CART algorithm with **Variance Reduction** as the splitting criterion. Focuses on predicting continuous values. |

### 2. Linear Models

| Directory | Model Type | Description |
| :--- | :--- | :--- |
| `Linear_Regression` | **Linear Regression** | Demonstrates the core linear regression methods: the closed-form **Ordinary Least Squares (OLS)**, **Ridge Regression** (regularized OLS), and the iterative **Gradient Descent (GD)** approach. |
| `Logistic_Regression` | **Logistic Regression** | Demonstrates binary classification using the **Sigmoid** function and optimized via **Batch Gradient Descent**. Highlights the necessity of feature scaling and $\mathbf{\{0, 1\}}$ target encoding. |
| `Perceptron` | **Perceptron** | Demonstrates the simplest linear classifier, showcasing its **error-correction learning rule**. The notebook emphasizes the strict requirement for $\mathbf{\{-1, 1\}}$ target encoding and linear separability. |
| `Multilayer_Perceptron` | **MLP Binary Classifier** | Demonstrates a foundational neural network, including multiple hidden layers, **ReLU** activation, and the use of **Backpropagation** and Binary Cross-Entropy Loss to solve non-linear classification problems. |

### 3. Instance-Based and Ensemble Methods

| Directory | Model Type | Description |
| :--- | :--- | :--- |
| `K_Nearest_Neighbors` | **KNN Classifier/Regressor** | Demonstrates the **lazy learning** method. The notebook shows how predictions are based on the distances to the *k* nearest neighbors, comparing the different distance metrics (Euclidean, Manhattan). |
| `Ensemble_Methods` | **Bagging Regressor/Classifier** | Demonstrates ensemble techniques, specifically **Bagging (Bootstrap Aggregating)**. This example showcases how to combine multiple base estimators (like Decision Trees) to create a more robust and accurate model. |

---

## Running the Examples

1. **Navigate** to the specific algorithm folder you wish to explore (e.g., `examples/Supervised_Learning/Decision_Trees`).
2. **Open** the corresponding `.ipynb` file (e.g., `decision_trees_example.ipynb`).
3. **Ensure** your `rice_ml` package is installed and accessible in your environment.
4. **Run** the cells sequentially to observe the training, evaluation, and convergence processes.

***
**Note on Data:** Most notebooks utilize standard datasets from Scikit-learn (e.g., Iris, synthetic data) for quick, reproducible demonstrations
***
