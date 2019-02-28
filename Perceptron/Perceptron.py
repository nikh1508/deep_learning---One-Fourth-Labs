import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class Perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):
        return (np.dot(x, self.w) >= self.b)

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return np.array(Y)

    def fit(self, X, Y, epochs=1, learning_rate=1.0, return_weight_matrix=False):
        self.w = np.ones(X.shape[1])
        self.b = 1
        accuracy = {}
        max_accuracy = 0.0
        wt_matrix = []
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y_pred == 0 and y == 1:
                    self.w += learning_rate * x
                    self.b += learning_rate * 1
                elif y_pred == 1 and y == 0:
                    self.w -= learning_rate * x
                    self.b -= learning_rate * 1
            wt_matrix.append(list(self.w))
            # print("i = ", i, wt_matrix)
            accuracy[i] = self.test_accuracy(X, Y)
            if accuracy[i] > max_accuracy:
                max_accuracy = accuracy[i]
                # Checkpointing ensures that at the end of Algorithm the hightest accuracy retains.
                checkpoint_w = self.w
                checkpoint_b = self.b
        self.w = checkpoint_w
        self.b = checkpoint_b
        print("Perceptron Model Fitting Complete.")
        print("The Maximum Accuracy obtained is equal to",
              max_accuracy * 100, "%.")
        print("The Maximum Accuracy was found at iteration :",
              max(accuracy, key=accuracy.get))
        plt.plot(accuracy.values())
        plt.ylim([0, 1])
        plt.show()
        if return_weight_matrix:
            return np.array(wt_matrix)

    def test_accuracy(self, X, Y):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, Y)
