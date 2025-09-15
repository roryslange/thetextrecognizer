import numpy as np

def sigmoid(val):
    return 1 / 1 + np.exp(-val)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
