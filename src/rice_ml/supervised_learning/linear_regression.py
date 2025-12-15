import numpy as np

class LinearRegression:
    """
    Linear Regression model implemented using Gradient Descent (GD) for optimization.

    This model finds the best-fit line (or hyperplane) for a dataset by minimizing
    the Mean Squared Error (MSE) cost function iteratively.

    Parameters
    ----------
    eta : float
        The learning rate (step size) for gradient descent (between 0.0 and 1.0).
        A small eta ensures stability, but a very small eta slows convergence.
        Default is 0.01.
    epochs : int
        The number of training iterations (passes over the entire training set).
        Default is 1000.
    random_state : int, optional
        Seed for the random number generator used for initial weight initialization.
        Default is None.

    Attributes
    ----------
    w_ : 1d-array
        Optimized weights (coefficients) after fitting, excluding the bias.
    b_ : float
        Optimized bias term (intercept) after fitting.
    cost_history_ : list
        The value of the Mean Squared Error cost function after each epoch.
    """
    def __init__(self, eta=0.01, epochs=1000, random_state=None):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        # Attributes initialized during fitting
        self.w_ = None
        self.b_ = None
        self.cost_history_ = []

    def fit(self, X, y):
        """
        Trains the Linear Regression model using Gradient Descent.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors (features).
        y : {array-like}, shape = [n_samples]
            Target values (true outcomes).

        Returns
        -------
        self : object
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # 1. Initialize weights (w) and bias (b)
        # Initialize w_ with small random values
        self.w_ = np.random.rand(n_features) 
        # Initialize bias b_ to a small random value (or 0)
        self.b_ = np.random.rand(1)[0]
        
        self.cost_history_ = []
        
        # 2. Start Gradient Descent Loop
        for _ in range(self.epochs):
            
            # 2a. Calculate the Hypothesis (Prediction)
            # h(X) = X * w + b. This predicts y for ALL samples at once.
            y_pred = self._net_input(X)
            
            # 2b. Calculate the Error Vector
            # Error = (h(X) - y)
            errors = y_pred - y
            
            # 2c. Calculate the Gradients
            # Gradient of Cost w.r.t. Weights (w_j): 
            # (1/n) * sum((h(xi) - yi) * xij)
            # np.dot(X.T, errors) efficiently computes the sum across all samples
            # for all features, matching the derivative formula.
            dw = (1 / n_samples) * np.dot(X.T, errors)
            
            # Gradient of Cost w.r.t. Bias (b): 
            # (1/n) * sum(h(xi) - yi)
            db = (1 / n_samples) * np.sum(errors)
            
            # 2d. Update Weights and Bias 
            # w = w - eta * dw
            # b = b - eta * db
            self.w_ -= self.eta * dw
            self.b_ -= self.eta * db
            
            # 2e. Calculate and Store Cost (MSE)
            # Cost = (1/2n) * sum((h(xi) - yi)^2)
            cost = (1 / (2 * n_samples)) * np.sum(errors**2)
            self.cost_history_.append(cost)
            
        return self

    def _net_input(self, X):
        """
        Calculates the linear hypothesis, h(X) = X * w + b.
        """
        # Linear equation: weighted sum of features plus the bias
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        Predicts the continuous target value for the input data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Input vectors.

        Returns
        -------
        1d-array
            Predicted target values.
        """
        return self._net_input(X)
