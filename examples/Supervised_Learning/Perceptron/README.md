# Perceptron

This package implements the **Perceptron**, the single-layer neural network model introduced by Frank Rosenblatt. It is a fundamental **binary classification** algorithm that finds a linear decision boundary separating two classes.

## Algorithm Overview

The Perceptron is an online, iterative, error-correcting algorithm. It updates the model weights only when a misclassification occurs.

### Core Mechanism

1. **Net Input ($z$):** For a given sample $\mathbf{x}$, the model calculates the weighted sum of its features plus the bias:
    $$
    z = \mathbf{w} \cdot \mathbf{x} + w_0
    $$
2. **Activation (Step Function):** The net input $z$ determines the output $\hat{y}$. If $z \ge 0$, the output is $+1$; otherwise, it is $-1$. [Image of the Heaviside Step Function]
3. **Update Rule:** If the prediction $\hat{y}$ does not match the true target $y$, the weights are updated based on the Perceptron rule:
    $$
    \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \Delta \mathbf{w}
    $$
    where the update term $\Delta \mathbf{w}$ is calculated as:
    $$
    \Delta \mathbf{w} = \eta \cdot (y - \hat{y}) \cdot \mathbf{x}
    $$
    The model continues iterating over the training data until all samples are correctly classified (convergence) or the maximum number of epochs is reached.

### Guarantee

The Perceptron learning algorithm is guaranteed to converge to a perfect solution if, and only if, the training data is **linearly separable**.

## Key Hyperparameters

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `eta` ($\eta$) | **Learning Rate.** Controls the magnitude of the weight adjustment during each misclassification. | `0.01` to `0.1` |
| `epochs` | **Maximum Iterations.** The total number of passes over the training dataset before stopping. | `50` |
| `random_state` | Seed for the initial random generation of the weights. | Any integer |

---

## Data Requirements and Preprocessing

The Perceptron has a strict requirement for the target variable format:

### 1. Target Variables ($\mathbf{Y}$)

* **Format:** Must be strictly **binary**, encoded as **$-1$ and $+1$**. Other encodings (like $\{0, 1\}$) will lead to incorrect updates.

### 2. Feature Scaling ($\mathbf{X}$)

* **Recommendation:** While not mandatory, **feature scaling (Standardization)** is strongly recommended. Scaling helps ensure all features contribute equally to the net input and can lead to faster convergence.
