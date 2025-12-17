import numpy as np
from typing import List, Optional, Tuple, Union, Sequence, Any
import warnings
from ..utils.validation import ArrayLike, ensure_2d_numeric, ensure_1d_vector, check_Xy_shapes
from ._linear_helpers import sigmoid, relu, relu_derivative
from .gradient_descent import GradientDescent

class MLPBinaryClassifier:
    r"""
    Multilayer Perceptron (MLP) for Binary Classification using Batch Gradient Descent.

    Architecture: Input -> (Hidden Layers with ReLU) -> Output (Sigmoid)
    Loss: Binary Cross-Entropy (Log Loss)
    Optimization: Batch Gradient Descent with Backpropagation
    
    Parameters
    ----------
    # ... (parameters remain the same) ...
    """
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100,), eta: float = 0.01, 
                 epochs: int = 100, random_state: Optional[int] = None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        self.weights_: List[np.ndarray] = []
        self.cost_history_: List[float] = []
        self.n_layers: int = 0
        self.classes_: Optional[np.ndarray] = None

    # --- Activation Functions and Derivatives (No changes) ---

    def _relu(self, Z: np.ndarray) -> np.ndarray:
        r"""Rectified Linear Unit (ReLU) activation: max(0, Z)."""
        return np.maximum(0, Z)

    def _relu_derivative(self, A: np.ndarray) -> np.ndarray:
        r"""Derivative of ReLU: 1 if A > 0, else 0."""
        return (A > 0).astype(float)

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        r"""Sigmoid activation: 1 / (1 + exp(-Z))."""
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    def _add_bias_unit(self, X: np.ndarray, how: str = 'col') -> np.ndarray:
        r"""Adds a bias unit (a column of 1s) to the input matrix X."""
        if how == 'col':
            return np.hstack((X, np.ones((X.shape[0], 1))))
        elif how == 'row':
            return np.vstack((X, np.ones((1, X.shape[1]))))
        raise ValueError("Bias unit addition must be 'col' or 'row'.")

    def _initialize_weights(self, n_features: int, n_output: int):
        r"""Initializes weights using a scaled random distribution (He/Xavier-like)."""
        
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_output]
        self.n_layers = len(layer_sizes) - 1
        
        self.weights_ = []
        
        for L in range(self.n_layers):
            n_in = layer_sizes[L]
            n_out = layer_sizes[L+1]
            scale = np.sqrt(2.0 / n_in)
            W = rng.standard_normal(size=(n_in + 1, n_out)) * scale 
            self.weights_.append(W)


    # --- Forward Propagation (No changes) ---

    def _forward_propagate(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r"""
        Performs the forward pass through the network.
        """
        A = [X]
        Z = [np.empty(0)]
        
        for L in range(self.n_layers - 1):
            X_biased = self._add_bias_unit(A[-1], how='col')
            Z_lplus1 = X_biased @ self.weights_[L]
            A_lplus1 = self._relu(Z_lplus1)
            
            A.append(A_lplus1)
            Z.append(Z_lplus1)
            
        # Output layer
        X_biased = self._add_bias_unit(A[-1], how='col')
        Z_out = X_biased @ self.weights_[self.n_layers - 1]
        A_out = self._sigmoid(Z_out)
        
        A.append(A_out)
        Z.append(Z_out)
        
        return A, Z
    
    # --- Cost Function (No changes) ---
    
    def _cost(self, A_out: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
        r"""Calculates the Binary Cross-Entropy Loss (Log Loss)."""
        n_samples = y.shape[0]
        y_pred = np.clip(A_out, eps, 1 - eps) 
        
        loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return float(np.sum(loss) / n_samples)

    # --- Backpropagation (No changes) ---
    
    def _backpropagate(self, A: List[np.ndarray], Z: List[np.ndarray], y: np.ndarray) -> List[np.ndarray]:
        r"""
        Performs the backward pass to calculate gradients for all weights.
        """
        n_samples = y.shape[0]
        dW: List[np.ndarray] = [np.empty(0)] * self.n_layers 
        deltas: List[np.ndarray] = [np.empty(0)] * (self.n_layers + 1)
        
        # 1. Output Layer
        A_out = A[-1]
        deltas[-1] = A_out - y.reshape(-1, 1)
        A_last_hidden_biased = self._add_bias_unit(A[-2], how='col')
        dW[self.n_layers - 1] = A_last_hidden_biased.T @ deltas[-1] / n_samples
        
        # 2. Hidden Layers
        for L in range(self.n_layers - 1, 0, -1):
            W_no_bias = self.weights_[L][:-1, :]
            error_prop = deltas[L+1] @ W_no_bias.T
            deltas[L] = error_prop * self._relu_derivative(Z[L])
            A_prev_biased = self._add_bias_unit(A[L-1], how='col')
            dW[L-1] = A_prev_biased.T @ deltas[L] / n_samples
            
        return dW

    # --- Public API Methods ---

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPBinaryClassifier":
        r"""
        Trains the MLP using batch gradient descent and backpropagation.
        """
        # --- UPDATE 1: Use imported validation function ---
        X_arr = ensure_2d_numeric(X)
        y_arr = np.asarray(y).astype(float)
        
        if len(np.unique(y_arr)) != 2:
            raise ValueError("Target y must contain exactly two classes {0, 1}.")
        if np.any((y_arr != 0) & (y_arr != 1)):
             raise ValueError("Target y must be binary and encoded as {0, 1}.")
            
        self.classes_ = np.unique(y_arr)
        
        self._initialize_weights(X_arr.shape[1], 1)

        self.cost_history_ = []
        for epoch in range(self.epochs):
            A, Z = self._forward_propagate(X_arr)
            A_out = A[-1]
            cost = self._cost(A_out, y_arr)
            self.cost_history_.append(cost)
            dW = self._backpropagate(A, Z, y_arr)
            
            for L in range(self.n_layers):
                self.weights_[L] -= self.eta * dW[L]
                
            if (epoch + 1) % 100 == 0:
                warnings.filterwarnings("ignore", message="The accuracy of the best fit is 1.0", category=UserWarning)
                print(f"Epoch {epoch + 1}/{self.epochs}, Cost: {cost:.4f}")
            
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        r"""
        Predicts the probability of belonging to class 1.
        """
        # --- UPDATE 2: Use imported validation function ---
        X_arr = ensure_2d_numeric(X)
        
        A, _ = self._forward_propagate(X_arr)
        
        return A[-1].flatten()

    def predict(self, X: ArrayLike) -> np.ndarray:
        r"""
        Predicts the class label (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)