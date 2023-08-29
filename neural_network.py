class NeuralNetwork:
    def __init__(self, loss_func, loss_func_prime, learning_rate):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.loss_func_prime = loss_func_prime

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, matrix_input):
        for layer in self.layers:
            matrix_input = layer.forward(matrix_input)

    def compute_loss(self, y_train):
        return self.loss_func(self.layers[-1].a, y_train)

    def backward(self, y_train):
        # this is output layer operation
        dz_next = self.loss_func_prime(self.layers[-1].a, y_train)

        for i in reversed(range(len(self.layers))):
            if i == len(self.layers)-1:
                w_next = None
            else:
                w_next = self.layers[i+1].weights
            dz_next = self.layers[i].backward(dz_next, w_next)
            # update weights and bias
            self.layers[i].weights -= self.learning_rate * self.layers[i].dWeights
            self.layers[i].biases -= self.learning_rate * self.layers[i].dBiases
