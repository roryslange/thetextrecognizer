import numpy as np

def sigmoid(val):
    return 1 / 1 + np.exp(-val)

def softmax(val):
    val_shift = val - np.max(val)
    return np.exp(val_shift) / np.sum(val_shift)

def relu(val):
    return np.max(0, val)

def relu_derivative(val):
    return (val > 0).astype(int)

def forwardProp(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = relu(Z1)
    Z2 = W2.dot(X) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backProp(X, Y, W1, B1, W2, B2, forwards):
    pass
