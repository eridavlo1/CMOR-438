# rice_ml Example Notebooks

This directory contains executable Jupyter Notebooks (`.ipynb`) and supporting documentation that demonstrate the usage, implementation, and core concepts of the machine learning models and utilities available in the `rice_ml` library.

## Goal

The primary goals of these examples are:

1. **Demonstration:** Show how to initialize, train, predict, and evaluate every core model in the library.
2. **Best Practices:** Emphasize critical preprocessing steps (like standardization) and proper evaluation to ensure correct model usage and prevent common errors like data leakage.
3. **Visualization:** Provide clear visual output (convergence curves, decision boundaries, cluster plots) to illustrate model behavior.

---

## Directory Structure Overview

The examples are organized into two major categories corresponding to the fundamental paradigms of machine learning:

### 1. Supervised_Learning

Contains examples for models that learn from labeled data (X, y) to predict targets (classification or regression).

| Model Directory | Core Model(s) | Focus |
| :--- | :--- | :--- |
| `Decision_Trees` | Decision Tree Classifier | Classification using Gini Impurity and Information Gain. |
| `Regression_Trees` | Decision Tree Regressor | Regression using Variance Reduction (MSE). |
| `Ensemble_Methods`| Random Forest, AdaBoost | Combining multiple models for improved performance. |
| `K_Nearest_Neighbors`| KNN Classifier/Regressor | Lazy, distance-based learning for both tasks. |
| `Linear_Regression` | OLS, Ridge, Gradient Descent | Basic linear modeling and optimization techniques. |
| `Logistic_Regression`| Logistic Regression | Binary classification using the Sigmoid function and Gradient Descent. |
| `Perceptron` | Perceptron | The fundamental single-layer neural network for linearly separable data. |
| `Multilayer_Perceptron`| MLP Binary Classifier | Non-linear classification using hidden layers and backpropagation. |

### 2. Unsupervised_Learning

Contains examples for models that find hidden patterns or structures in unlabeled data (X).

| Model Directory | Core Model(s) | Focus |
| :--- | :--- | :--- |
| `PCA` | Principal Component Analysis | Dimensionality reduction by maximizing variance. |
| `DBSCAN` | DBSCAN | Density-based clustering, finding arbitrary shapes and noise. |
| `K_Means_Clustering` | K-Means | Centroid-based clustering for globular data. |
| `Community_Detection`| Label Propagation | Graph-based algorithm for finding natural groupings in networks. |

---

## Essential Preprocessing Note

The robust design of the `rice_ml` library emphasizes proper data handling. In nearly **all** examples, especially those involving distance calculations or Gradient Descent optimization (e.g., K-Means, Perceptron, Logistic Regression, PCA), feature scaling is mandatory.

* **Rule:** Features must be standardized (Z-score scaling) **before** training the model.
* **Method:** All notebooks use the powerful `standardize` function from `rice_ml.preprocessing` and adhere to the principle: **Fit only on the training data; transform both training and test data.** This ensures reliable results and prevents data leakage.

## Getting Started

1. **Install `rice_ml`:** Ensure the library and its dependencies (NumPy, Matplotlib, scikit-learn for data loading) are installed in your environment.
2. **Navigate:** Choose an algorithm of interest (e.g., `Supervised_Learning/Logistic_Regression`).
3. **Execute:** Run the cells in the corresponding `.ipynb` notebook sequentially to replicate the training, evaluation, and visualization steps.
