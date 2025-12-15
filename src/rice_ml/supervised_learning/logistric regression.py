import numpy as np

class LogisticRegression:
    """
    Binary Logistic Regression classifier implemented using Gradient Descent (GD).

    This model estimates the probability of a binary outcome (0 or 1) by fitting 
    a linear function to the data and passing the result through the sigmoid function.

    Parameters
    ----------
    eta : float
        The learning rate (step size) for gradient descent (between 0.0 and 1.0).
        Default is 0.01.
    epochs : int
        The number of training iterations (passes over the entire training set).
        Default is 1000.
    random_state : int, optional
        Seed for the random number generator used for initial weight and bias initialization.
        Default is None.
    
    Attributes
    ----------
    w_ : 1d-array
        Optimized weight vector (coefficients) after fitting.
    b_ : float
        Optimized bias term (intercept) after fitting.
    cost_history_ : list
        The value of the Cost function (Binary Cross-Entropy) after each epoch.
    """
    def __init__(self, eta=0.01, epochs=1000, random_state=None):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        # Attributes initialized during fitting
        self.w_ = None
        self.b_ = None
        self.cost_history_ = []

    def _sigmoid(self, z):
        """
        The Sigmoid (or Logistic) activation function.
        f(z) = 1 / (1 + exp(-z))
        """
        # Clips the input to prevent overflow in exp(-z) for very large negative z
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _net_input(self, X):
        """
        Calculates the net input, z = X * w + b.
        """
        if X.ndim == 1:
             # Handle single sample input
            return np.dot(X, self.w_) + self.b_
        return np.dot(X, self.w_) + self.b_

    def fit(self, X, y):
        """
        Trains the Logistic Regression model using Gradient Descent.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors (features).
        y : {array-like}, shape = [n_samples]
            Target values, must be binary and encoded as {0, 1}.

        Returns
        -------
        self : object
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # 1. Initialize weights (w) and bias (b)
        self.w_ = np.random.rand(n_features) * 0.01
        self.b_ = np.random.rand(1)[0] * 0.01 # Initializing close to zero often helps
        
        self.cost_history_ = []
        
        # 2. Start Gradient Descent Loop
        for _ in range(self.epochs):
            
            # 2a. Calculate the Hypothesis (Predicted Probability)
            # h(X) is P(y=1 | X)
            z = self._net_input(X)
            y_prob = self._sigmoid(z) # [Image of Sigmoid function]
            
            # 2b. Calculate the Error Vector
            # Error = (y_prob - y)
            # This calculation (h(X) - y) is the term needed for the cross-entropy gradient.
            errors = y_prob - y
            
            # 2c. Calculate the Gradients
            # Gradient w.r.t. Weights (w): (1/n) * sum((y_prob - y) * x)
            dw = (1 / n_samples) * np.dot(X.T, errors)
            
            # Gradient w.r.t. Bias (b): (1/n) * sum(y_prob - y)
            db = (1 / n_samples) * np.sum(errors)
            
            # 2d. Update Weights and Bias [Image of Gradient Descent for Logistic Regression Cost function]
            self.w_ -= self.eta * dw
            self.b_ -= self.eta * db
            
            # 2e. Calculate and Store Cost (Binary Cross-Entropy/Log Loss)
            # J = - (1/n) * sum( y*log(y_prob) + (1-y)*log(1-y_prob) )
            # Add a small value (1e-15) to log inputs to prevent log(0)
            cost = (-1 / n_samples) * np.sum(y * np.log(y_prob + 1e-15) + (1 - y) * np.log(1 - y_prob + 1e-15))
            self.cost_history_.append(cost)
            
        return self

    def predict_proba(self, X):
        """
        Predicts the probability of belonging to class 1.
        """
        return self._sigmoid(self._net_input(X))

    def predict(self, X):
        """
        Predicts the class label (0 or 1).
        
        Uses a threshold of 0.5 on the predicted probability.
        """
        probabilities = self.predict_proba(X)
        # Class 1 if probability >= 0.5, else Class 0
        return np.where(probabilities >= 0.5, 1, 0)
