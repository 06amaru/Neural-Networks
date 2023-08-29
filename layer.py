import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation_func, activation_func_prime):
        self.dBiases = None
        self.dWeights = None
        self.dz = None
        self.a = None
        self.z = None
        self.input = None
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation_func
        self.activation_prime = activation_func_prime

    def forward(self, matrix_input):
        self.input = matrix_input
        self.z = self.input @ self.weights + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backward(self, dz_next, w_next):
        if w_next is None:
            self.dz = dz_next
        else:
            self.dz = (dz_next @ w_next.T) * self.activation_prime(self.z)

        self.dWeights = self.input.T @ self.dz
        self.dBiases = np.sum(self.dz, axis=0, keepdims=True)
        return self.dz
