# Decision Tree
## Algorithm
Decisions Trees are a non-parametric supervised learning algorithm used for both classification and regression task. They work by partitioning the feature space into a set of rectangular regions. The prediction for any sample that falls into a specific region (or leaf node) is the majority class (for classification) or the average target value (for regression) of the training samples in that region.
## 1. Core Idea: Learning by Splitting
The tree is built top-down by recursively selecting the best feature and threshold to split the data. The goal at each step is to choose a split that results in the purest possible child nodes, meaning the samples within each child node belong predominantly to one class (in classification).
## 2. Objective Function: Impurity and Information Given
The "best" split is determined by minimizing an impurity measure or, equivalently, maximizing the Information Gain (IG).
- Classification (CART Algorithm): The common impurity measure is Gini Impurity or Entropy.
    - Gini Impurity (G): Measures the probability of misclassifying a randomly chosen element in the subset.
    - Information Gain (IG): The reduction in impurity acheived by the split
- Regression (CART Algorithm): Uses measures like Mean Squared Error (MSE) or Mean Absolute Error (MAE) to determine the best split
## 3. Key Hyperparameters
Decisions trees are prone to overfitting (creating a very deep tree that performs perfectly on training data but poorly on unseen data). The primary hyperparameters are used to control the tree's complexity and prevent this:

|Hyperparameter | Description | Effect on Complexity|
|-------------- | ----------- | --------------------|
| max_depth | The maximum number of levels (splits) alllowed in the tree.| Limits complexity (prevents overfitting)|
|min_samples_split | The minimum number of samples a node must contain to be considered for splitting | Limits complexity (prevents overfitting)|
|min_sample_leaf | The minimum number of samples required to be in a leaf node. | Ensures leaves are representative|
|criterion | The function to measure the quality of a split (e.g., 'gini', 'entropy', 'mse'). | Affects the split selection process. |
## Data
Decision Trees are versatile and handle various data types well, but certain considerations are needed for optimal performance.
### 1. Input Features ($X$)
- Numeric Features: Trees handle numerical data naturally. They identify optimal split thresholds (e.g., "Is Feature A $<= 5.5$?). No specific scaling (like Standardization or Normalization) is required because the splitting logic is based on absolute values and is invariant to monotonic transformations.
- Categorical Features: Categorical Features must be encoded into a numerical format.
    - One-Hot Encoding: Creates a binary columns for each category (e.g., Red = 1, Blue = 2). THis should be used cautiously as it implies an implies an ordinal relationship which the tree might mistakenly interpret. One-Hot Encoding is generally safer.
### 2. Labels ($y$)
- Classification: Labels must be discrete, integer-encoded class values (e.g., $0,1,2$).
- Regression: Labels are continuous, real-valued numbers.
### 3. Data Loading and Preprocessing
The input data $X$ and labels $y$ are expected to be NumPy arrays.
1. Load Data: Load a dataset (e.g., from a CSV file using Pandas, then convert to NumPy).
2. Handle Missing Values: Decision Trees cannot natively handle missing data (NaNs).
3. Encode Categoricals: Apply appropriate encoding (as described above).
    Missing values must be imputed (e.g., with the mean, median, or mode of the feature) or the rows must be dropped.
4. Split Data: Separate the dataset into training and testing sets to evaluate generalization performance. 