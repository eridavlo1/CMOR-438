import numpy as np

class Perceptron:
    """
    Perceptron classifier implemented from scratch using NumPy.
    
    This model is designed for binary classification problems and uses the 
    Perceptron learning rule, iterating over the training data until 
    convergence or maximum epochs are reached.
    
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0). Default is 0.01 (modified from 0.5 
        for better stability in a general case).
    epochs : int
        Maximum number of passes over the training data (epochs). Default is 50.
    random_state : int, optional
        The seed used by the random number generator for initial weight initialization. 
        Default is None.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting, where the last element is the bias unit (w_0).
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, epochs=50, random_state=None):
        # Hyperparameters
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        # Attributes initialized during training
        self.w_ = None
        self.errors_ = []

    def fit(self, X, y):
        """
        Trains the Perceptron model on the given training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples 
            and n_features is the number of features.
        y : {array-like}, shape = [n_samples]
            Target values, must be binary and encoded as {-1, 1}.
            
        Returns
        -------
        self : object
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # 1. Initialize weights: 
        # w_ is initialized with small random values. 
        # The size is n_features + 1 (for the bias unit w_0).
        self.w_ = np.random.rand(1 + X.shape[1])
        
        self.errors_ = []
        
        for epoch in range(self.epochs):
            errors = 0
            
            # Iterate through each sample in the training data
            for xi, target in zip(X, y):
                # Calculate the prediction/target difference (error)
                # target is the expected output (-1 or 1)
                # self.predict(xi) is the actual output (-1 or 1)
                
                prediction = self._predict_single(xi)
                
                # The core Perceptron update rule: w = w + eta * (target - prediction) * xi
                # Here, we calculate 'update' as: eta * (target - prediction)
                # Note: target - prediction will be:
                # 0   if correct
                # 2   if target=1, pred=-1 (needs +ve update)
                # -2  if target=-1, pred=1 (needs -ve update)
                
                update = self.eta * (target - prediction) 
                
                # Check for misclassification (update != 0 means target != prediction)
                if update != 0.0:
                    errors += 1

                # Update weights for features (w_1 to w_n)
                # Note: the update term already includes the sign and learning rate (eta)
                self.w_[:-1] += update * xi
                
                # Update the bias unit (w_0, which is self.w_[-1])
                self.w_[-1] += update
            
            # Record the number of errors for this epoch
            self.errors_.append(errors)
            
            # Check for convergence: if no errors, stop training early
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return self
                
        return self

    def _net_input(self, X):
        """
        Calculates the net input, z = w * x + w_0.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features] or [n_features]
            Input vector(s).
            
        Returns
        -------
        1d-array
            The net input (z) before the activation function.
        """
        # np.dot(X, self.w_[:-1]) performs the weighted sum for all features
        # self.w_[-1] is the bias (w_0)
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def _predict_single(self, xi):
        """
        Predicts the class label for a single input vector.
        """
        # Step function activation: if net_input >= 0, classify as 1, else -1
        # np.where(condition, value_if_true, value_if_false)
        return np.where(self._net_input(xi) >= 0.0, 1, -1)

    def predict(self, X):
        """
        Predicts the class labels for the input data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Input vectors.
            
        Returns
        -------
        1d-array
            Predicted class label (1 or -1).
        """
        # Apply the prediction rule to all samples at once
        return np.where(self._net_input(X) >= 0.0, 1, -1)
