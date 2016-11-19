from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np

class BaseOptimizer():

    __metaclass__ = ABCMeta

    def __init__(self,learning_rate=0.01, decay=0.01,momentum=0.9, ):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum


    @abstractmethod
    def backprop(self, nn, x, y):
        pass

    def _forwardprop(self,nn,x, trace=True):
        if trace == True:
            outputs = []
        input = x

        for i in range(len(nn._layersObject)):

            layer = nn._layersObject[i]
            Output = layer.activation_function(np.add(np.dot(input, layer.weights), layer.bias))
            if trace == True:
                if i == len(nn._layersObject)-1:
                    outputs.append((input, Output))
                else:
                    outputs.append((input))
            input = Output

        if trace == True:
            return outputs

        return Output




