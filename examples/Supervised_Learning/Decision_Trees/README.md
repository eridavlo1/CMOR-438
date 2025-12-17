# Decision Tree

## Algorithm

Decisions Trees are a non-parametric supervised learning algorithm used for both classification and regression task. They work by partitioning the feature space into a set of rectangular regions. The prediction for any sample that falls into a specific region (or leaf node) is the majority class (for classification) or the average target value (for regression) of the training samples in that region.

## 1. Core Idea: Learning by Splitting

The tree is built top-down by recursively selecting the best feature and threshold to split the data. The goal at each step is to choose a split that results in the purest possible child nodes, meaning the samples within each child node belong predominantly to one class (in classification).

## 2. Objective Function: Impurity and Information Gain

The construction of a Decision Tree is a greedy process where, at every node, the algorithm seeks the optimal split point (feature and threshold) that maximizes the resulting homogeneity of the child nodes. This optimization is achieved by maximizing the **Information Gain (IG)**, which is the reduction in impurity achieved by the split.

### Impurity Metrics

The Decision Tree algorithm (specifically, the CART algorithm) uses different impurity measures depending on the task:

* **Classification (Implemented in this code):** The goal is to maximize the homogeneity of class labels within each node. The implemented `DecisionTreeClassifier` supports:
    1. **Gini Impurity ($G$):** Measures the probability of misclassifying a randomly chosen element in the subset if it were randomly labeled according to the distribution of labels.
        $$G = 1 - \sum_{k=1}^{K} (p_k)^2$$
    2. **Entropy ($H$):** Measures the degree of randomness or uncertainty (disorder) in the distribution of class labels.
        $$H = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

* **Regression:** For continuous target variables, the objective is to minimize the variance within each node. Common measures include Mean Squared Error (MSE) or Mean Absolute Error (MAE).

### Information Gain Calculation

**Information Gain ($IG$)** quantifies the value of a split by measuring the drop in the impurity measure from the parent node to the weighted average of the child nodes' impurities:

$$IG = \text{Impurity}(\text{Parent}) - \sum_{j=1}^{m} \frac{N_j}{N} \text{Impurity}(\text{Child}_j)$$

---

## 3. Controlling Tree Complexity (Hyperparameters)

Decision trees are inherently prone to **overfitting** (creating a very deep, complex tree that perfectly memorizes the training data but generalizes poorly to unseen data). The primary hyperparameters are used to control the tree's complexity and prevent this by providing early stopping conditions.

| Hyperparameter | Description | Effect on Complexity |
| :--- | :--- | :--- |
| `max_depth` | The maximum number of levels (splits) allowed in the tree. Once this depth is reached, no further splitting occurs. | Limits complexity (Primary method to prevent deep overfitting). |
| `min_samples_split` | The minimum number of samples a node must contain to be considered for splitting. If fewer samples are present, the node becomes a leaf. | Limits complexity (Prevents splitting on small, potentially noisy subsets). |
| `max_features` | The number of features to consider when looking for the best split (e.g., `'sqrt'` for $\sqrt{N}$ features). This is crucial for ensemble methods like Random Forests. | Introduces randomness and decorrelates trees; reduces time complexity. |
| `criterion` | The function used to measure the quality of a split (e.g., 'gini', 'entropy'). | Affects the split selection process, but not the structural complexity. |

(Note: `min_sample_leaf` is a common hyperparameter but is not explicitly implemented in the provided `DecisionTreeClassifier`.)

## Data

Decision Trees are versatile and handle various data types well, but certain considerations are needed for optimal performance.

### 1. Input Features ($X$)

* Numeric Features: Trees handle numerical data naturally. They identify optimal split thresholds (e.g., "Is Feature A $<= 5.5$?). No specific scaling (like Standardization or Normalization) is required because the splitting logic is based on absolute values and is invariant to monotonic transformations.
* Categorical Features: Categorical Features must be encoded into a numerical format.
  * One-Hot Encoding: Creates a binary columns for each category (e.g., Red = 1, Blue = 2). THis should be used cautiously as it implies an implies an ordinal relationship which the tree might mistakenly interpret. One-Hot Encoding is generally safer.

### 2. Labels ($y$)

* Classification: Labels must be discrete, integer-encoded class values (e.g., $0,1,2$).
* Regression: Labels are continuous, real-valued numbers.

### 3. Data Loading and Preprocessing

The input data $X$ and labels $y$ are expected to be NumPy arrays.

1. Load Data: Load a dataset (e.g., from a CSV file using Pandas, then convert to NumPy).
2. Handle Missing Values: Decision Trees cannot natively handle missing data (NaNs).
3. Encode Categoricals: Apply appropriate encoding (as described above).
    Missing values must be imputed (e.g., with the mean, median, or mode of the feature) or the rows must be dropped.
4. Split Data: Separate the dataset into training and testing sets to evaluate generalization performance.
