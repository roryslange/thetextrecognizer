from .dataBuilder import dataframeFromCsv
from .neuron import one_hot, gradient_dissent
import numpy as np

def train():
    data = dataframeFromCsv()
    print(data.head())

    # the following code is based on multiple sources on how to partition the data for this example
    # I specifically got the code from this video: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=691s by Samson Zhang
    # main idea is we want to transpose data so its easier to index, and split into dev and training
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255

    W1, B1, W2, B2 = gradient_dissent(X_train, Y_train, 100, 0.1)








