import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def plot_data(self, x, y, i, fig):
        # Dibujamos los datos
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.title(f"Iteración {i + 1}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        x_plots = np.array([x[:, 0].min(), x[:, 0].max()])
        y_plots = []
        for counts in range (0, len(x_plots)):
            y_line = -(self.weights[2] / self.weights[1]) / (self.weights[2] / self.weights[0]) * x_plots[counts] + (-self.weights[2] / self.weights[1])
            y_plots.append(y_line)
        print("Iteracion: ", i)
        print("X_plots: ", x_plots)
        print("Y_plots: ", y_plots)
        print("Weights", self.weights)
        plt.plot(x_plots, y_plots)
        fig.canvas.draw_idle()

    def fit(self, x, y, num_iter=100):
        plt.ion()
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.draw()
        n_samples = x.shape[0]
        n_features = x.shape[1]
        self.weights = np.zeros((n_features + 1,))
        x = np.concatenate([x, np.ones((n_samples, 1))], axis=1)  # Columna de unos / el bias viene dentro del vector
        for i in range(num_iter):  # Recorremos las iteraciones
            plt.clf()
            self.plot_data(x[:, :-1], y, i, fig)
            plt.pause(0.25)

            for j in range(n_samples):  # Recorremos cada valor de n_samples
                producto_escalar = np.dot(self.weights, x[j, :])
                print("Producto escalar:", producto_escalar)
                y_predicted = np.where(producto_escalar > 0, 1, -1)
                print("Antes de actualizar pesos:", self.weights)
                update = self.learning_rate * (y[j] - y_predicted)
                self.weights += update * x[j, :]
                print("Después de actualizar pesos:", self.weights)

        print(self.weights)
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
        return np.mean(y == pred_y)  # Comparamos la pred_y con la y real y analuzamos la media
