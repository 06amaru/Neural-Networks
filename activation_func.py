import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0).astype(float)


def softmax(x):
    shifted_matrix = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted_matrix)
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    return exps/sum_exps


def cross_entropy_loss(y_train, output):
    return -np.sum(y_train * np.log(output + 1e-10)) / len(y_train)


def cross_entropy_loss_prime(x, y):
    return x - y
