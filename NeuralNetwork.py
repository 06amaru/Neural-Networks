import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE = 0.01
EPOCHS = 20
np.random.seed(7) #agregamos una semilla

df = pd.read_csv('data/mnist_train.csv')

imgs =df.iloc[1:,1:] #separamos las imagenes de los labels
imgs=imgs.to_numpy()
print(imgs.shape)
y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy()
unique, counts = np.unique(y, return_counts=True)

y_train = np.zeros((len(y), 10))
for i, y in enumerate(y):
    y_train[i][y] = 1

imgs=imgs/255

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        print(inputs[:5])
        self.output = np.dot(inputs, self.weights) + self.biases
        print(self.output[:5])

layer1 = Layer_Dense(784, 10)
layer1.forward(imgs)

#print(layer1.output[:5])

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

activation1 = Activation_ReLu()
z1=activation1.forward(layer1.output)
