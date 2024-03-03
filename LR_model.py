import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batch_data(data, batch_size): 
    result = []
    for i in range(0, len(data), batch_size):  
        result.append(data[i:i + batch_size])
    return result

def add_data_interaction(data):
    result = []
    for row in np.array(data):
        new_row = row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                new_row = np.append(new_row, row[i] * row[j])
        result.append(new_row)
    return np.array(result)


class LR:
    def __init__(self, learning_rate=0.01, n_iterations=10000, interaction_model=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.interaction_model = interaction_model

    def fit(self, X, y, optimization_algorithm='SGD', batch_size=1):
        if self.interaction_model:
            X = add_data_interaction(X)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        if optimization_algorithm == 'SGD':
            X_batched = batch_data(X, batch_size)
            y_batched = batch_data(y, batch_size)

            # this min and // shenanigans is to not make smaller batch_size
            # have much more iterations compared to higher values
            for i in range(self.n_iterations * min(batch_size, n_samples) // n_samples):
                for X_batch, y_batch in zip(X_batched, y_batched):
                    batch_n_samples = X_batch.shape[0]
                    y_pred = sigmoid(np.dot(X_batch, self.weights) + self.bias)

                    dw = (1 / batch_n_samples) * np.dot(X_batch.T, (y_pred - y_batch))
                    db = (1 / batch_n_samples) * np.sum(y_pred - y_batch)

                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db

        if optimization_algorithm == 'IWLS':
            pass

        if optimization_algorithm == 'ADAM':
            pass

        # code for whole dataset gradient
        # for i in range(self.n_iterations):
        #     y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

        #     dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        #     db = (1 / n_samples) * np.sum(y_pred - y)

        #     self.weights -= self.learning_rate * dw
        #     self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        if self.interaction_model:
            X = add_data_interaction(X)
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

        return y_pred
    
    def predict(self, X):
        if self.interaction_model:
            X = add_data_interaction(X)
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        
        return [0 if y < 0.5 else 1 for y in y_pred]
