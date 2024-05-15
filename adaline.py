import matplotlib.pyplot as plt
import numpy as np


class Adaline:
    def __init__(self, num_iter=100, learning_rate=0.01, random_state=1):
        self.weights = None
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, x, y):
        r_gen = np.random.RandomState(self.random_state)  # Generando los pesos con valores aleatorios
        self.weights = r_gen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])  # Generamos los pesos de la x mas el bias
        for i in range(self.num_iter):
            # Funcion de activacion: momento que tengamos una entrada
            y_predicted = np.dot(x, self.weights[1:]) + self.weights[0]
            # Error: recta entre la y producida y la y_predicted
            error = y - y_predicted
            # Actualizamos los pesos: x.T sirve para transponer el vector y poder multiplicarlo
            self.weights[1:] += self.learning_rate * x.T.dot(error)
            # Calculamos el bias
            self.weights[0] += self.learning_rate * error.sum()