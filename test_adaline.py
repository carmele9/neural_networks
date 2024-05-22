from sklearn.datasets import make_classification

from adaline import Adaline
from sklearn import datasets
from sklearn.model_selection import train_test_split

x, y = make_classification(
    n_features=2,
    n_classes=2,
    n_samples=1000,
    n_redundant=0,
    n_clusters_per_class=1,
    shuffle=False,
    random_state=0
)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

ada = Adaline(20)
ada.fit(x_train, y_train)
print("Score: ", ada.score(x_test, y_test))
