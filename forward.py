import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE = 0.00001
EPOCHS = 600
np.random.seed(7) #agregamos una semilla

df = pd.read_csv('data/mnist_train.csv')

imgs =df.iloc[0:,1:] #separamos las imagenes de los labels
X=imgs.to_numpy()
y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy().squeeze()
unique, counts = np.unique(y, return_counts=True)

y_train = np.zeros((len(y), 10))
for i, y_it in enumerate(y):
    y_train[i][y_it] = 1
#Normalizing data to prevent overflow
X=X/255

#Activation functions

def softmax(x):
    shifted_matrix = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted_matrix)
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    return exps/sum_exps

#Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs) #W
        self.biases = np.zeros((1, n_neurons))

#Creating model
#2 hidden layers
layer = Layer_Dense(n_inputs=784,n_neurons=10) 
print("training in process...")
#Training #MODIFY!
for j in range(EPOCHS):
    #Forward Propagation
    z1=X @ layer.weights + layer.biases
    a1=softmax(z1)
    #Computation of error
    loss = -np.sum(y_train * np.log(a1 + 1e-10)) / len(y_train)  # Adding epsilon to avoid log(0)
        
    #Back Propagation
    gradient_z1 = a1 - y_train
    gradient_w1 = X.T @ gradient_z1 # dL/dW = dL/dZ * dZ/dW 
    gradient_b1 = np.sum(gradient_z1, axis=0, keepdims=True) # dL/dB dimension = 1,number of neurons

    layer.weights -= LEARNING_RATE * gradient_w1
    layer.biases -= LEARNING_RATE * gradient_b1
    
    
    print(f"epoch: {j+1} , loss: {loss} , process: "+"*"*(int((j/EPOCHS)*20)))
print("Neural network trained successfully!")