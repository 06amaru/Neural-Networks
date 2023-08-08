import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE = 0.01
EPOCHS = 20
np.random.seed(7) #agregamos una semilla

df = pd.read_csv('data/mnist_train.csv')

imgs =df.iloc[1:,1:] #separamos las imagenes de los labels
imgs=imgs.to_numpy()
#print(imgs.shape)
y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy()
unique, counts = np.unique(y, return_counts=True)

y_train = np.zeros((len(y), 10))
for i, y in enumerate(y):
    y_train[i][y] = 1

imgs=imgs/255

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

layer1 = Layer_Dense(784, 10)
#layer1.forward(imgs)

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

activation1 = Activation_ReLu()
#z1=activation1.forward(layer1.output)

class CategoricalCrossentropy:
    def forward(self, y_true, y_pred):
        # Asegurarse de que las predicciones no sean exactamente 0 o 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calcular la entropía cruzada para cada muestra
        cross_entropy = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

        # Devolver el promedio de la entropía cruzada para todas las muestras
        return np.mean(cross_entropy)
    
loss_func = CategoricalCrossentropy()

loss_history = []

for epoch in range(EPOCHS):
    layer1.forward(imgs)
    Z = layer1.output
    activation1.forward(Z)
    A = activation1.output

    loss = loss_func.forward(y_train, A)
    loss_history.append(loss)

    dL_dA = -y_train/A
    dA_dZ = np.where(Z > 0, 1, 0)

    dZ_dW = np.dot(imgs.T, dL_dA*dA_dZ)
    ## dZ_db = np.sum(dL_dA * dA_dZ, axis=0, keepdims=True) 

    layer1.weights -= dZ_dW * LEARNING_RATE
