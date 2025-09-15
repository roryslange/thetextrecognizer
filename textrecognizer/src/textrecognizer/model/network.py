import numpy as np

from textrecognizer.model.neuron import Neuron

"""
We want this network to have one or two hidden layers
(probably 2)
so im going to start with the hidden layer just being an empty array.
this may eventually be an array of neurons

maybe combine layers into an array then can have lots of them
"""
class Network:
    def __init__(self, hiddenlayer0):
        self.hidden0 = hiddenlayer0
        # self.hidden1 = hiddenlayer1
        self.output = Neuron([],0)

    def feedForward(self, x):
        h0_out = np.array([])
        for neuron in self.hidden0:
            np.append(h0_out, neuron.feedForward(x))

        return self.output.feedForward(h0_out)


def networkTest():
    pass