import numpy as np

def sigmoid(val):
    return 1 / 1 + np.exp(-val)

def softmax(val):
    val_shift = val - np.max(val)
    return np.exp(val_shift) / np.sum(val_shift)

def relu(val):
    return np.max(0, val)

def forwardProp(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = relu(Z1)
    Z2 = W2.dot(X) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
