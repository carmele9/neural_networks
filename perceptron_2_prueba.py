import numpy as np
import matplotlib.pyplot as plt


class Perceptron_2:
    def __init__(self):
        self.weights = None

    def plot_data(self, x, y, i, fig):
        if self.weights is None:
            raise ValueError("El perceptron no ha sido entrenado.")
            # Dibujamos los datos
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
            plt.title(f"Iteración {i + 1}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            x_plots = np.array([x[:, 0].min(), x[:, 0].max()])
            y_plots = []
            for counts in range(0, len(x_plots)):
                y_line = -(self.weights[2] / self.weights[1]) / (self.weights[2] / self.weights[0]) * x_plots[
                    counts] + (-self.weights[2] / self.weights[1])
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
        n_samples = x.shape[0]
        n_features = x.shape[1]
        self.weights = np.zeros(n_features + 1)  # Inicializar con sesgo
        x = np.concatenate([x, np.ones((n_samples, 1))], axis=1)  # Agregar columna de unos para el sesgo
        for i in range(num_iter):  # Recorremos las iteraciones
            self.plot_data(x[:, :-1], y, i, fig)
            plt.pause(0.25)
            for j in range(n_samples):
                print("Tipos de x[j, :]:", x[j, :].dtype)
                print("Tipo de y[j]:", type(y[j]))
                dot_product = np.dot(self.weights, x[j, :])
                print("Producto Escalar ", dot_product)
                print("Pesos antes de actualizar: ", self.weights)
                if y[j] * dot_product <= 0:
                    self.weights += np.float64(y[j]) * x[j, :]
                    print( "Pesos despues de actualizar", self.weights)
    plt.waitforbuttonpress()

    def predict(self, x):
        if self.weights is None:
            print("El modelo no está entrenado.")
            return None
        n_samples = x.shape[0]
        x = np.concatenate([x, np.ones((n_samples, 1))], axis=1)  # Columna de unos para el sesgo
        y = np.dot(x, self.weights)  # Calcular el producto escalar
        return np.where(y > 0, 1, -1)  # Devuelve 1 o -1 según el signo

    def score(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == y)  # Calcular la precisión

