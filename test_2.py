from sklearn.datasets import make_classification
from perceptron import Perceptron
import numpy as np

# Creamos los datos para trabajar con ellos
x, y = make_classification(
    n_features=2,
    n_classes=2,
    n_samples=200,
    n_redundant=0,
    n_clusters_per_class=1
)


#Usamos el perceptron para hacer el fit
perceptron = Perceptron()
perceptron.fit(x, y, num_iter=50)
