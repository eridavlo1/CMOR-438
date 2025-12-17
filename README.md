# rice_ml: A Custom Machine Learning Library from Scratch

This repository hosts a custom-built machine learning package, **`rice_ml`**, developed as a core project for **CMOR 438 (Data Science and Machine Learning)**. The project implements a foundational set of **supervised and unsupervised learning algorithms entirely from scratch** using Python and NumPy.

The primary goal is to shift focus from black-box framework usage to **algorithmic transparency**, emphasizing the mathematical mechanics, proper data handling, and interpretability of each model.

---

## Project Highlights

This repository serves as a complete, educational codebase showcasing:

* **Custom Core Implementations:** Complete, from-scratch source code for all major algorithms, focusing on vectorized NumPy operations.
* **Modular Package Design:** A cleanly organized, installable Python package (`rice_ml`).
* **Rigorous Preprocessing:** Dedicated modules for standardized feature scaling and data splitting, enforcing the "fit-on-train, transform-on-test" rule.
* **Educational Examples:** Structured directories (`examples/`) that visually and analytically demonstrate model behavior.
* **Quality Assurance:** A comprehensive **`pytest` test suite** covering correctness and stability.

---

## Capabilities: The `rice_ml` Feature Set

### Supervised Learning

Models designed for predictive tasks, implemented in `src/rice_ml/supervised_learning`:

| Algorithm | Focus | Examples Available |
| :--- | :--- | :--- |
| **Linear Models** | Linear Regression (OLS), Logistic Regression | `examples/Supervised_Learning/Linear_Regression`, `examples/Supervised_Learning/Logistic_Regression` |
| **Nearest Neighbors** | K-Nearest Neighbors Classifier | `examples/Supervised_Learning/K_Nearest_Neighbors` |
| **Neural Networks** | Perceptron, Multilayer Perceptron (MLP) | `examples/Supervised_Learning/Perceptron`, `examples/Supervised_Learning/Multilayer_Perceptron` |
| **Tree Methods** | Decision Trees, Regression Trees | `examples/Supervised_Learning/Decision_Trees`, `examples/Supervised_Learning/Regression_Trees` |
| **Ensemble Methods** | Ensemble techniques (e.g., Bagging/Boosting variants) | `examples/Supervised_Learning/Ensemble_Methods` |

### Unsupervised Learning

Models designed for pattern discovery and data transformation, implemented in `src/rice_ml/unsupervised_learning`:

| Algorithm | Focus | Examples Available |
| :--- | :--- | :--- |
| **Clustering** | K-Means (Distance-based), DBSCAN (Density-based) | `examples/Unsupervised_Learning/K_Means_Clustering`, `examples/Unsupervised_Learning/DBSCAN` |
| **Dimensionality Reduction** | Principal Component Analysis (PCA) | `examples/Unsupervised_Learning/PCA` |
| **Graph Analysis** | Community Detection | `examples/Unsupervised_Learning/Community_Detection` |

### Data Processing Utilities

Implemented in `src/rice_ml/processing` and `src/rice_ml/utils`:

| Utility | Description |
| :--- | :--- |
| **`processing`** | Includes core transformations like feature scaling (`StandardScaler`) and data splitting (`train_test_split`). |
| **`utils`** | General helper functions and evaluation metrics (e.g., `accuracy_score`, `r2_score`, `mse`). |

---

## Repository Structure

The project follows a standard, clean layout to separate source code, executable examples, and testing infrastructure.

| Directory/File | Purpose |
| :--- | :--- |
| **`examples/`** | Contains structured demonstration scripts/notebooks for every implemented algorithm. |
| **`src/`** | The core source code for the installable `rice_ml` package. |
| `src/rice_ml/supervised_learning` | All supervised learning models (e.g., Linear Regression, MLP). |
| `src/rice_ml/unsupervised_learning` | All unsupervised learning models (e.g., K-Means, PCA). |
| `src/rice_ml/processing` | Data preparation utilities (e.g., standardization). |
| **`tests/`** | Comprehensive unit tests for all models and utilities. |
| **`.github/`** | Configuration for GitHub features (e.g., issue templates). |

---

## Exploring Examples

To run the examples and see the models in action:

1. Navigate to the specific example directory (e.g., `examples/Supervised_Learning/Logistic_Regression`).
2. Follow the instructions in the example file (usually a Jupyter Notebook or Python script) to load data, process it, train the model, and evaluate performance.

This is the fastest way to understand the API and usage of the implemented models.

---

## Testing

The project maintains a strong emphasis on correctness through unit testing.

To run the full test suite, ensure you have `pytest` installed and run the following command from the repository root:

```bash
pytest

```

## Installation

Since this is a project-based library, it is not published to PyPI. To use it in your environment:
    1. Clone the Repository:

```bash
git clone [https://github.com/eridavlo1/rice_ml.git](https://github.com/eridavlo1/CMOR-438.git)
cd rice_ml
```

    2. Install Locally (Editable Mode):

``` bash
pip install -e .
```

This command installs the package in "editable" mode, allowing you to import `rice_ml` in your scripts while any changes you make to the `src/` directory are immediately reflected.

### Author and License

**Author:** Erika Vasquez (CMOR 438, Fall 2025)
**License:** This project is licensed under the MIT License. See the LICENSE file for details.
