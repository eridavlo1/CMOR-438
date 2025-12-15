import numpy as np
from typing import List, Optional, Tuple, Union, Sequence, Any
import warnings

# --- Missing Definitions Added for standalone execution ---

ArrayLike = Union[np.ndarray, Sequence[Any]]

def _check_for_nan_inf(arr: np.ndarray, name: str) -> None:
    """Check for NaN and Inf values (simplified helper)."""
    if np.isnan(arr).any():
        raise ValueError(f"Input array {name} contains NaN values. Please handle missing data.")
    if np.isinf(arr).any():
        raise ValueError(f"Input array {name} contains Infinite values. Please handle extreme data.")

def _ensure_2d_numeric(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Ensure X is a 2D numeric NumPy array. 
    (Simplified version from pre_processing.py for dependency resolution)
    """
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1) # Handle 1D input by reshaping to (n, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    
    _check_for_nan_inf(arr, name)

    return arr

# --- MLPBinaryClassifier Class (Original Code Follows) ---

class MLPBinaryClassifier:
    """
    Multilayer Perceptron (MLP) for Binary Classification using Batch Gradient Descent.
    
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

    # --- Activation Functions and Derivatives ---

    def _relu(self, Z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU) activation: max(0, Z)."""
        return np.maximum(0, Z)

    def _relu_derivative(self, A: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if A > 0, else 0."""
        return (A > 0).astype(float)

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + exp(-Z))."""
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    def _add_bias_unit(self, X: np.ndarray, how: str = 'col') -> np.ndarray:
        """Adds a bias unit (a column of 1s) to the input matrix X."""
        if how == 'col':
            return np.hstack((X, np.ones((X.shape[0], 1))))
        elif how == 'row':
            return np.vstack((X, np.ones((1, X.shape[1]))))
        raise ValueError("Bias unit addition must be 'col' or 'row'.")

    def _initialize_weights(self, n_features: int, n_output: int):
        """Initializes weights using a scaled random distribution (He/Xavier-like)."""
        
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_output]
        self.n_layers = len(layer_sizes) - 1
        
        self.weights_ = []
        
        for L in range(self.n_layers):
            n_in = layer_sizes[L]
            n_out = layer_sizes[L+1]
            
            # He-like initialization for ReLU layers
            scale = np.sqrt(2.0 / n_in)
            
            # W includes the bias unit W[l][n_in, :]
            W = rng.standard_normal(size=(n_in + 1, n_out)) * scale 
            self.weights_.append(W)


    # --- Forward Propagation ---

    def _forward_propagate(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
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
    
    # --- Cost Function ---
    
    def _cost(self, A_out: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
        """Calculates the Binary Cross-Entropy Loss (Log Loss)."""
        n_samples = y.shape[0]
        y_pred = np.clip(A_out, eps, 1 - eps) 
        
        loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return float(np.sum(loss) / n_samples)

    # --- Backpropagation (Core Training Logic) ---
    
    def _backpropagate(self, A: List[np.ndarray], Z: List[np.ndarray], y: np.ndarray) -> List[np.ndarray]:
        """
        Performs the backward pass to calculate gradients for all weights.
        """
        n_samples = y.shape[0]
        # Initialize gradient list (dW[L] stores the gradient for W[L])
        dW: List[np.ndarray] = [np.empty(0)] * self.n_layers 
        # Deltas/errors for each layer (Delta[L] is error for layer L)
        deltas: List[np.ndarray] = [np.empty(0)] * (self.n_layers + 1)
        
        # 1. Output Layer (L = n_layers)
        A_out = A[-1]
        deltas[-1] = A_out - y.reshape(-1, 1) # Delta[out] = A[out] - y
        
        # Gradient for output weights W[n_layers-1]: dW[out] = A[last_hidden]_biased.T @ Delta[out]
        A_last_hidden_biased = self._add_bias_unit(A[-2], how='col')
        dW[self.n_layers - 1] = A_last_hidden_biased.T @ deltas[-1] / n_samples
        
        # 2. Hidden Layers (L = n_layers - 1 down to 1)
        for L in range(self.n_layers - 1, 0, -1):
            # Step 1: Compute the error signal propagating back
            # W_no_bias excludes the bias row (W[-1, :]) which is not used for backpropagation
            W_no_bias = self.weights_[L][:-1, :]
            
            # error_prop = Delta[L+1] @ W[L]_no_bias.T (weighted sum of errors)
            error_prop = deltas[L+1] @ W_no_bias.T

            # Step 2: Apply the derivative of the activation function (ReLU)
            deltas[L] = error_prop * self._relu_derivative(Z[L])
            
            # Step 3: Compute the gradient for W[L-1]
            A_prev_biased = self._add_bias_unit(A[L-1], how='col')
            dW[L-1] = A_prev_biased.T @ deltas[L] / n_samples
            
        return dW

    # --- Public API Methods ---

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPBinaryClassifier":
        """
        Trains the MLP using batch gradient descent and backpropagation.
        """
        # Uses the newly defined helper function to validate/prepare input
        X_arr = _ensure_2d_numeric(X)
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
        """
        Predicts the probability of belonging to class 1.
        """
        X_arr = _ensure_2d_numeric(X)
        
        A, _ = self._forward_propagate(X_arr)
        
        return A[-1].flatten()

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the class label (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)
