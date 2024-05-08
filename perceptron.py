import time
import numpy as np
import matplotlib.pyplot as plt



class Perceptron:

    def plot_data(self, x, y, i, fig):
        # Dibujamos los datos
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.title(f"Iteración {i + 1}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        fig.canvas.draw_idle()


    def fit(self, x, y, num_iter=100):
        plt.ion()
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.draw()
        n_samples = x.shape[0]
        n_features = x.shape[1]
        self.weights = np.zeros((n_features + 1,))
        x = np.concatenate([x, np.ones((n_samples, 1))], axis=1)  # Columna de unos
        for i in range(num_iter):  # Recorremos las iteraciones
            self.plot_data(x[:, :-1], y, i, fig)
            plt.pause(0.25)
            for j in range(n_samples):  # Recorremos cada valor de n_samples
                  if y[j]*np.dot(self.weights, x[j, :]) <= 0:  # Evaluamos el producto escalar
                    # Actualizamos los pesos
                    self.weights += y[j]*x[j, :]
        plt.waitforbuttonpress()


    def predict(self, x):
        if not hasattr(self, "weights"):
            print("The model is not trained")
            return
        n_samples = x.shape[0]
        x = np.concatenate([x, np.ones((n_samples, 1))], axis=1)  # Columna de unos
        y = np.matmul(x, self.weights)  # Multiplicamos matrizes
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)  # Evaluamos si el resultado es mayor o menor que 0
        return y

    def score(self, x, y):
        pred_y = self.predict(x)
        return np.mean(y==pred_y)  # Comparamos la pred_y con la y real y analuzamos la media


