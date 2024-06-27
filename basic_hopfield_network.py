import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)

    def asynchronous_update(self, input_pattern, max_iterations = 100):
        for _ in range(max_iterations):
            neuron_idx = np.random.randint(self.num_neurons)
            activation = np.dot(self.weights[neuron_idx], input_pattern)
            input_pattern[neuron_idx] = 1 if activation > 0 else -1

        return input_pattern

    def synchronous_update(self, input_pattern, max_iterations=100):
        for _ in range(max_iterations):
            activations = np.dot(self.weights, input_pattern)
            input_pattern = np.sign(activations)

        return input_pattern
