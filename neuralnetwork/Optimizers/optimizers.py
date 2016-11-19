from base_optimizer import BaseOptimizer
from neuralnetwork.utils import generate_batches
import numpy as np


class SGD(BaseOptimizer):
    '''
    Stochastic gradient descent optimizer.
    '''

    def __init__(self, learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)


    def update_weights(self, delta, layer_input, curr_weights,Lambda):
        # print ('delta shape'), delta.shape
        # print('layer input'), layer_input.shape
        # raw_input()
        gradient = np.dot(layer_input.T, delta) - Lambda * curr_weights
        new_weights = curr_weights - self.learning_rate * gradient
        return new_weights


    def update_bias(self, delta, curr_bias):
        gradient = self.learning_rate * np.sum(delta, axis=0, keepdims=True)

        new_bias = curr_bias - self.learning_rate * gradient
        return new_bias


class RMSProp(BaseOptimizer):

    def __init__(self, learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)
