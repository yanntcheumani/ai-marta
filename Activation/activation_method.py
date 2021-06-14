import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return z * (z > 0)


def relu_prime(z):
    return z <= 0
