from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from perceptron_3_prueba import Perceptron_3
import numpy as np

# Creamos los datos para trabajar con ellos
x, y = make_classification(
    n_features=2,
    n_classes=2,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)


# Usamos el perceptron para hacer el fit
#perceptron = Perceptron()
#perceptron.fit(x, y, num_iter=10)

perceptron = Perceptron_3(learning_rate=0.01, n_iters=1000)
perceptron.fit(x, y)

# Predecir y evaluar
predictions = perceptron.predict(x)
accuracy = np.mean(predictions == y)
print(f'Accuracy: {accuracy:.2f}')

# Visualizar la frontera de decisión
perceptron.plot_decision_boundary(x, y)


