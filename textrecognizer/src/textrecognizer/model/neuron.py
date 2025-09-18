import numpy as np

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def softmax(val):
    exp_vals = np.exp(val - np.max(val, axis=0, keepdims=True))  # numerical stability
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

def relu(val):
    return np.maximum(val, 0)

def relu_derivative(val):
    return val > 0

def forwardProp(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backProp(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1] # get the y axis
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, dB1, dW2, dB2

def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, a):
    W1 = W1 - a * dW1
    B1 = B1 - a * dB1
    W2 = W2 - a * dW2
    B2 = B2 - a * dB2
    return W1, B1, W2, B2

def init_params():
    W1 = np.random.randn(128, 784) * 0.01
    B1 = np.zeros((128, 1))
    W2 = np.random.randn(10, 128) * 0.01
    B2 = np.zeros((10, 1))
    return W1, B1, W2, B2

def gradient_dissent(X, Y, iterations, alpha):
    W1, B1, W2, B2 = init_params()
    # print(W1.shape, B1.shape, W2.shape, B2.shape)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, B1, W2, B2, X)
        # print(Z1.shape, A1.shape, Z2.shape, A2.shape)
        dW1, dB1, dW2, dB2 = backProp(X, Y, Z1, A1, Z2, A2, W2)
        # print(dW1.shape, dB1.shape, dW2.shape, dB2.shape)
        W1, B1, W2, B2 = update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)
        if (i % 15 == 0):
            print(f'iteration: {i}')
            print(f'accuracy: {accuracy(predictions(A2), Y)}')
    return W1, B1, W2, B2

def accuracy(predictions, Y):
    print (predictions, Y)
    print(np.min(predictions), np.max(predictions))
    return np.sum(predictions == Y) / Y.size

def predictions(A2):
    return np.argmax(A2, 0)