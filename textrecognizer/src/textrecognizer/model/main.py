from .dataBuilder import dataframeFromCsv
from .neuron import one_hot
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

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]

    W1 = np.random.standard_normal((10, 784)) - 0.5
    B1 = np.random.standard_normal((10, 1)) - 0.5

    W2 = np.random.standard_normal((10, 784)) - 0.5
    B2 = np.random.standard_normal((10, 1)) - 0.5

    arr = np.array([1,4,5,2,3,4,2])
    arr = arr.T
    print(one_hot(arr))





