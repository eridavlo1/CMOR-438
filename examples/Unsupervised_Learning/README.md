# Unsupervised Learning Examples

This directory contains executable Jupyter Notebooks (`.ipynb`) demonstrating the implementation and usage of the core unsupervised machine learning algorithms available in the `rice_ml` library.

## Goal

Unsupervised learning is used to find inherent structure, patterns, or groupings within data without relying on pre-existing labels. The notebooks here demonstrate how to:

1. Initialize and configure models that find patterns (like clusters or component directions).
2. Properly preprocess data for distance-based algorithms (mandatory scaling).
3. Execute the fitting process and interpret the resulting structure (e.g., cluster centers, explained variance).

---

## Available Model Examples

### 1. Clustering and Grouping

| Directory | Model Type | Description |
| :--- | :--- | :--- |
| `K_Means_Clustering` | **K-Means Clustering** | Demonstrates the iterative, centroid-based algorithm for partitioning data into $K$ distinct, globular clusters. Highlights the importance of the **`n_clusters`** parameter and the use of the **K-Means++** initialization method for faster convergence. |
| `DBSCAN` | **DBSCAN (Density-Based Clustering)** | Demonstrates the non-centroid-based approach for finding clusters of arbitrary shape and identifying **outliers (noise)**. Emphasizes the crucial role of the **`eps`** (radius) and **`min_samples`** (density) parameters. |
| `Community_Detection` | **Label Propagation Algorithm (LPA)** | Demonstrates a graph-based technique for finding natural groupings (communities) within networks represented by an adjacency matrix. LPA is a fast, iterative method that does **not require a predefined number of communities** ($K$). |

### 2. Dimensionality Reduction

| Directory | Model Type | Description |
| :--- | :--- | :--- |
| `PCA` | **Principal Component Analysis** | Demonstrates the linear transformation technique used to project high-dimensional data onto a lower-dimensional subspace. Shows how to select components based on a fixed count or a **desired level of explained variance**. **Standardization is mandatory** before applying PCA. |

---

## Running the Examples

1. **Navigate** to the specific algorithm folder you wish to explore (e.g., `examples/Unsupervised_Learning/PCA`).
2. **Open** the corresponding `.ipynb` file.
3. **Ensure** your `rice_ml` package is installed and accessible.
4. **Note on Preprocessing:** Unlike some supervised examples, scaling (using the `standardize` function from `rice_ml.preprocessing`) is **required** for all notebooks in this directory, as they rely on accurate distance metrics.
