import numpy as np
import pandas as pd

# Semilla para reproducibilidad
np.random.seed(7)

df = pd.read_csv('data/mnist_train.csv')
imgs =df.iloc[0:,1:] #separamos las imagenes de los labels
X=imgs.to_numpy()
y_train = df.iloc[:,0:1] #capturamos los labels
y_train = y_train.to_numpy().squeeze()

# Tomar una muestra de entrada y su etiqueta del conjunto de datos MNIST
inputSample = X[0]  # Tomamos la primera imagen del conjunto de datos
labelSample = y_train[0]  # Tomamos la etiqueta correspondiente en formato one-hot
y_train = np.zeros(10)
y_train[labelSample] = 1

# Generar valores iniciales aleatorios para los pesos y sesgos
initialW1 = np.random.randn(784, 32) * np.sqrt(2. / 784)
initialB1 = np.zeros((1, 32))
initialW2 = np.random.randn(32, 10) * np.sqrt(2. / 32)
initialB2 = np.zeros((1, 10))

# Imprimir los datos para verificar
#print("inputSample:", inputSample)
#print("labelSample:", labelSample)
#print("initialW1:", initialW1)
#print("initialB1:", initialB1)
#print("initialW2:", initialW2)
#print("initialB2:", initialB2)
# Guardar los datos en archivos .csv
np.savetxt("inputSample.csv", inputSample.reshape(1, -1), delimiter=",")
np.savetxt("labelSample.csv", labelSample.reshape(1, -1), delimiter=",")
np.savetxt("initialW1.csv", initialW1, delimiter=",")
np.savetxt("initialB1.csv", initialB1, delimiter=",")
np.savetxt("initialW2.csv", initialW2, delimiter=",")
np.savetxt("initialB2.csv", initialB2, delimiter=",")

print("Datos guardados en archivos .csv")
def softmax(x):
    shifted_matrix = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted_matrix)
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    return exps/sum_exps
z1 = inputSample@initialW1 + initialB1
a1 = np.maximum(0, z1)
z2 = a1@initialW2 + initialB2
a2 = softmax(z2)

gradient_z2 = a2 - y_train
print(gradient_z2)
gradient_w2 = a1.T@gradient_z2
print(np.trace(gradient_w2))
gradient_b2 = np.sum(gradient_z2, axis=0, keepdims=True)
#print(gradient_b2)

relu_derivative = (z1 > 0).astype(float)
gradient_z1 = (gradient_z2 @ initialW2.T)*relu_derivative
#print(gradient_z1)
transpose = inputSample.reshape(1, -1)
gradient_w1 = transpose.T@gradient_z1
print(np.sum(gradient_w1))
