import numpy as np

def sigmoid(val):
    return 1 / 1 + np.exp(-val)

def softmax(val):
    val_shift = val - np.max(val)
    return np.exp(val_shift) / np.sum(val_shift)

def relu(val):
    return np.maximum(0, val)

def relu_derivative(val):
    return (val > 0).astype(int)

def forwardProp(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backProp(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1] # get the y axis
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, a):
    W1 = W1 - a * dW1
    B1 = B1 - a * dB1
    W2 = W2 - a * dW2
    B2 = B2 - a * dB2
    return W1, B1, W2, B2

def init_params():
    W1 = np.random.randn(10, 784) - 0.5
    B1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    B2 = np.random.randn(10, 1) - 0.5
    return W1, B1, W2, B2

def gradient_dissent(X, Y, iterations, alpha):
    W1, B1, W2, B2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backProp(X, Y, Z1, A1, Z2, A2, W2)
        W1, B1, W2, B2 = update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)
        if (i % 15 == 0):
            print(f'iteration: {i}')
            print(f'loss: {accuracy(predictions(A2), Y)}')
    return W1, B1, W2, B2

def accuracy(predictions, Y):
    print (predictions, Y)
    return np.sum(predictions == Y) / Y.size

def predictions(A2):
    return np.argmax(A2, 0)