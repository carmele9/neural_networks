import matplotlib.pyplot as plt
import numpy as np


class Adaline:
    def __init__(self, num_iter=100, learning_rate=0.01, random_state=1):
        self.weights = None
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, x, y):
        plt.ion()
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.draw()
        r_gen = np.random.RandomState(self.random_state)  # Generando los pesos con valores aleatorios
        self.weights = r_gen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])  # Generamos los pesos de la x mas el bias
        for i in range(self.num_iter):
            plt.clf()
            self.plot_data(x[:, :], y, i, fig)
            plt.pause(0.25)
            # Funcion de activacion: momento que tengamos una entrada
            y_predicted = np.dot(x, self.weights[1:]) + self.weights[0]
            # Error: recta entre la y producida y la y_predicted
            error = y - y_predicted
            # Actualizamos los pesos: x.T sirve para transponer el vector y poder multiplicarlo
            self.weights[1:] += self.learning_rate * x.T.dot(error)
            # Calculamos el bias
            self.weights[0] += self.learning_rate * error.sum()
        plt.waitforbuttonpress()

    def predict(self, x):
        sum_escalar = np.dot(x, self.weights[1:]) + self.weights[0]
        return np.where(sum_escalar >= 0.0, 1, 0)

    def score(self, x, y):
        error_data_count = 0
        for z, target in zip(x, y):
            output = self.predict(z)
            if target != output:
                error_data_count += 1
        total_data_count = len(x)
        self.score = (total_data_count-error_data_count) / total_data_count
        return self.score

    def plot_data(self, x, y, i, fig):
        # Dibujamos los datos
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis")
        plt.title(f"Iteración {i + 1}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        x_plots = np.array([x[:, 0].min(), x[:, 0].max()])
        y_plots = []
        for counts in range(0, len(x_plots)):
            y_line = -(self.weights[2] / self.weights[1]) / (self.weights[2] / self.weights[0]) * x_plots[counts] + (-self.weights[2] / self.weights[1])
            y_plots.append(y_line)
        print("Iteracion: ", i)
        print("X_plots: ", x_plots)
        print("Y_plots: ", y_plots)
        print("Weights", self.weights)
        plt.plot(x_plots, y_plots)
        fig.canvas.draw_idle()
