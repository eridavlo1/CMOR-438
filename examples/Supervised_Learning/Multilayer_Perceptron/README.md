# Multilayer Perceptron (MLP) Binary Classifier

This package provides a from-scratch implementation of a **Multilayer Perceptron (MLP)**, a feedforward artificial neural network used for complex **binary classification** tasks.

## Architecture and Mechanism

The MLP overcomes the linear separability limitations of the simple Perceptron by introducing one or more hidden layers, allowing it to model non-linear relationships. [Image of Multilayer Perceptron architecture]

### 1. Structure

The network is composed of interconnected layers:

* **Input Layer:** Receives the feature data.
* **Hidden Layers:** One or more layers that transform the data non-linearly.
* **Output Layer:** Produces the final prediction (probability).

### 2. Activation and Optimization

| Component | Function | Purpose |
| :--- | :--- | :--- |
| **Hidden Layer Activation** | **Rectified Linear Unit (ReLU):** $A = \max(0, Z)$ | Introduces non-linearity to allow the network to learn complex patterns. |
| **Output Layer Activation** | **Sigmoid Function:** $\hat{p} = 1 / (1 + e^{-Z})$ | Squashes the output into the range $(0, 1)$, representing the probability of Class 1. |
| **Loss Function** | **Binary Cross-Entropy (Log Loss)** | Measures the difference between the predicted probability ($\hat{p}$) and the true label ($y$). |
| **Learning Algorithm** | **Backpropagation** with **Batch Gradient Descent (GD)** | An efficient method for calculating and applying weight updates based on the error gradient across all layers. |

## Key Hyperparameters

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `hidden_layer_sizes` | `Tuple[int, ...]` | Defines the number of hidden layers and the number of neurons in each layer (e.g., `(10, 5)` means two hidden layers with 10 and 5 neurons, respectively). | `(100,)` |
| `eta` ($\eta$) | `float` | **Learning Rate** for Gradient Descent. Controls the size of the weight update step. | `0.01` |
| `epochs` | `int` | **Maximum Iterations.** The total number of passes over the training dataset. | `100` |
| `random_state` | `int` | Seed used for the initialization of the network weights. | `None` |

---

## Data Requirements and Preprocessing

As with all deep learning models optimized by Gradient Descent, proper data scaling is essential.

### 1. Target Variables ($\mathbf{Y}$)

* **Format:** Must be strictly **binary**, encoded as **integers 0 and 1**.

### 2. Feature Scaling ($\mathbf{X}$)

* **Requirement:** **Feature scaling (Standardization or Normalization)** is **mandatory**. This ensures numerical stability, prevents gradients from exploding or vanishing, and significantly improves convergence speed.
