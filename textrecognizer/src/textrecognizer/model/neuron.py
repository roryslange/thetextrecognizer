import numpy as np

def sigmoid(val):
    return 1 / 1 + np.exp(-val)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    """
    Basically what we want to do here is (x1 * i1) + ... + (xN * iN) + b
    where x is a weight, i is an input, and b is the bias
    we then want to use sigmoid on it so it stays between 0 and 1
    """
    def feedForward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)