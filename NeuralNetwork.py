import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE = 0.000001
EPOCHS = 1200
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

def relu(x):
    return np.maximum(0, x)

#Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs) #W
        self.biases = np.zeros((1, n_neurons))

#Creating model
#2 hidden layers
hidden_layer = Layer_Dense(n_inputs=784,n_neurons=32) 
output_layer=Layer_Dense(n_inputs=32,n_neurons=10)
print("training in process...")
#Training #MODIFY!
for j in range(EPOCHS):
    #Forward Propagation
    z1=X @ hidden_layer.weights + hidden_layer.biases
    a1=relu(z1)
    z2=a1 @ output_layer.weights + output_layer.biases
    a2=softmax(z2)
    #Computation of error
    loss = -np.sum(y_train * np.log(a2 + 1e-10)) / len(y_train)  # Adding epsilon to avoid log(0)
        
    #Back Propagation
    gradient_z2 = a2-y_train # dL/dZ  = dL/dA * dA/dZ
    gradient_w2 = a1.T @ gradient_z2 # dL/dW = dL/dZ * dZ/dW 
    gradient_b2 = np.sum(gradient_z2, axis=0, keepdims=True) # dL/dB dimension = 1,number of neurons
    #Updating parameters 
    output_layer.weights -= LEARNING_RATE * gradient_w2
    output_layer.biases -= LEARNING_RATE * gradient_b2
    
    relu_derivative = (z1 > 0).astype(float)
    gradient_z1 = (gradient_z2 @ output_layer.weights.T) * relu_derivative # dot product (@) and hadamard product (*)
    gradient_w1 = X.T @ gradient_z1
    gradient_b1 = np.sum(gradient_z1, axis=0, keepdims=True)

    hidden_layer.weights -= LEARNING_RATE * gradient_w1
    hidden_layer.biases -= LEARNING_RATE * gradient_b1
    
    
    print(f"epoch: {j+1} , loss: {loss} , process: "+"*"*(int((j/EPOCHS)*20)))
print("Neural network trained sucessfully!")
data_test=pd.read_csv("data/mnist_test.csv")
x_test=data_test.iloc[:,1:].to_numpy()/255
y_test=data_test.iloc[:,:1].to_numpy().squeeze()
#print(x_test.shape)
#print(y_test.shape)
#MOMENT OF TRUTH
def prediction(input_,y_true,n_cases):
    hits=0
    for i in range(n_cases):
        x= input_[i:i+1,:]
        z1=x @ hidden_layer.weights + hidden_layer.biases
        a1=relu(z1)
        z2=a1 @ output_layer.weights + output_layer.biases
        a2=softmax(z2)
        pred=np.argmax(a2)
        print(f"Prediction: {pred} vs Real: {y_true[i]}",end=" ")
        if pred==y_true[i]:
            hits+=1
            print("CORRECT")
        else:
            print("INCORRECT")
    return "Precission: "+str(hits/n_cases)
print(prediction(x_test,y_test,100))