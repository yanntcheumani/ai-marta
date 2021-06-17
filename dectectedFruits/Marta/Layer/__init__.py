from Activation.activation_method import *

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.all_activation = ((tanh, tanh_prime), (relu, relu_prime), (sigmoid, sigmoid_prime))

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
