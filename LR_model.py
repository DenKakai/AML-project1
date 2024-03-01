import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class LR:
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

        return y_pred
    
    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        
        return [0 if y < 0.5 else 1 for y in y_pred]
