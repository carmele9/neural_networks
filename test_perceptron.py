from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Creamos los datos para trabajar con ellos
x, y = make_classification(
    n_features=2,
    n_classes=2,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)

# Representamos los datos
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y)
plt.show()

# Usamos el perceptron para hacer el fit
perceptron = Perceptron()
perceptron.fit(x, y, num_iter=10)
