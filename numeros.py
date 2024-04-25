import tensorflow as tf
import matplotlib.pyplot as plt

# Dataset: load data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images shape: " + str(train_images.shape))
print("train_labels shape: " + str(train_labels.shape))
print("test_images shape: " + str(test_images.shape))
print("test_labels shape: " + str(test_labels.shape))

# Representar los nueve primeros datos del dataset
fig = plt.figure(figsize=(10, 10))
n_rows = 3
n_cols = 3
for i in range(9):
    fig.add_subplot(n_rows, n_cols, i+1)
    plt.imshow(train_images[i])
    plt.title("Digit: {}".format(train_labels[i]))
    plt.axis(False)
plt.show()

# Preprocesamiento de datos
train_images = train_images / 255  # Valor entre 0-1
test_images = test_images / 255
print("Primer label antes de la conversion onehot: " + str(train_labels[0]))
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
print("Primer label despues de la conversion onehot " + str(train_labels[0]))

# Definir la arquitectura del modelo
# Modelo 2D layer -> Flatten 28x28 = 784
# Datos en una misma fila
# --------------------------------------
# Hidden Layer -> Dense con 512 neuronas
# La f(x) de activacion es relu
# Los pesos se ajustan despues a medida que se realiza el entrenamiento
# ----------------------------------------
# Output layer -> Dense con 10 neuronas correspondientes a las categorias
# La f(x) de activacion es softmax
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 512, activation = "relu"),
    tf.keras.layers.Dense(units = 10, activation = "softmax")
]) # Capas detras de una la otra

# Optimizar el modelo: funcion de perdida
model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)

# Entrenamiento del modelo
history = model.fit(
    x = train_images,
    y = train_labels,
    epochs = 10
)
# Representacion grafica de iteraciones cuando la funcion de perdida se va a acercando cada vez
# mas a 0 y el accuracy aumenta de valor
plt.plot(history.history["loss"])
plt.xlabel("epochs")
plt.legend(["loss"])
plt.show()

plt.plot(history.history["accuracy"], color = "orange")
plt.xlabel("epochs")
plt.legend(["accuracy"])
plt.show()

# Evaluando el modelo usando los datos de testeo
test_loss, test_accuraccy = model.evaluate(
    x = test_images,
    y = test_labels
)
print("Test Loss: " + str(test_loss))
print("Test Accuracy: " + str(test_accuraccy))
