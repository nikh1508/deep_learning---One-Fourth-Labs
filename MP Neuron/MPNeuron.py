import numpy as np
from sklearn.metrics import accuracy_score


class MPNeuron:

    def __init__(self):
        self.b = None

    def model(self, x):
        return (sum(x) >= self.b)

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}
        for b in range(X.shape[1] + 1):
            self.b = b
            y_pred = self.predict(X)
            accuracy[b] = accuracy_score(y_pred, Y)
        self.b = max(accuracy, key=accuracy.get)
        print("Model Fitting Complete.\nThe Optimal Parameter is",
              self.b, "with accuracy of", accuracy[self.b] * 100, "%.")

    def test_accuracy(self, X, Y):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, Y)
