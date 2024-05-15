import numpy as np
import matplotlib.pyplot as plt


class Perceptron_3:

        def __init__(self, learning_rate=0.01, n_iters=1000):
            self.learning_rate = learning_rate
            self.n_iters = n_iters
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0

            y_ = np.where(y <= 0, -1, 1)

            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    linear_output = np.dot(x_i, self.weights) + self.bias
                    y_predicted = np.where(linear_output > 0, 1, -1)

                    update = self.learning_rate * (y_[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update

        def predict(self, X):
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = np.where(linear_output > 0, 1, -1)
            return y_predicted

        def plot_decision_boundary(self, X, y):
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                                   np.arange(x2_min, x2_max, 0.01))
            grid = np.c_[xx1.ravel(), xx2.ravel()]
            Z = self.predict(grid)
            Z = Z.reshape(xx1.shape)

            plt.contourf(xx1, xx2, Z, alpha=0.3)
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
            plt.xlim(x1_min, x1_max)
            plt.ylim(x2_min, x2_max)
            plt.show()
