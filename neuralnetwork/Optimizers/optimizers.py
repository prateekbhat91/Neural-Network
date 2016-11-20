from __future__ import division
from base_optimizer import BaseOptimizer
import numpy as np
import copy


class SGD(BaseOptimizer):
    '''
    Stochastic gradient descent optimizer.
    '''

    def __init__(self, learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)


    def update_weights(self, delta, layer_input,layer_index, curr_weights,Lambda):

        gradient = np.dot(layer_input.T, delta) - Lambda * curr_weights

        self._gradients[layer_index] = self.momentum * self._gradients[layer_index] - self.learning_rate * gradient

        return curr_weights + self._gradients[layer_index]

    def decay_learning_rate(self, run):
        '''
        This function decays the learning rate depending
        on the number of runs and decay rate.
        :param run:
        :return:
        '''
        self.learning_rate *= 1 / (1 + self.decay * run)




class RMSProp(BaseOptimizer):

    def __init__(self, learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)

    def update_weights(self, delta, layer_input, layer_index, curr_weights, Lambda):
        gradient = np.dot(layer_input.T, delta) - Lambda * curr_weights

        eps = 1e-8

        self._gradients[layer_index] = self.decay * self._gradients[layer_index] + (1-self.decay) * np.power(gradient,2)

        return curr_weights - self.learning_rate * gradient / (np.sqrt(self._gradients[layer_index]) + eps)


class Adam(BaseOptimizer):

    def __init__(self,learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self._smooth_gradients = {}

    def set_smooth_gradients(self,nn):
        for i in range(len(nn._layersObject)):
            self._smooth_gradients[i] = 0


    def update_weights(self, delta, layer_input, layer_index, curr_weights, Lambda):
        gradient = np.dot(layer_input.T, delta) - Lambda * curr_weights

        eps = 1e-8

        self._smooth_gradients[layer_index] = self.beta1 * self._smooth_gradients[layer_index] + (1-self.beta1) * gradient

        self._gradients[layer_index] = self.beta2 * self._gradients[layer_index] + (1- self.beta2) * np.power(gradient,2)

        return curr_weights - self.learning_rate * self._smooth_gradients[layer_index] / (np.sqrt(self._gradients[layer_index]) + eps)



class Nesterov(BaseOptimizer):

    def __init__(self,learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)


    def update_weights(self, delta, layer_input,layer_index, curr_weights,Lambda):

        gradient = np.dot(layer_input.T, delta) - Lambda * curr_weights

        prev_velocity = copy.copy(self._gradients[layer_index])

        new_velocity = self.momentum * self._gradients[layer_index] - self.learning_rate * gradient

        return curr_weights - self.momentum * prev_velocity + (1 + self.momentum) * new_velocity


    def decay_learning_rate(self, run):
        '''
        This function decays the learning rate depending
        on the number of runs and decay rate.
        :param run:
        :return:
        '''
        self.learning_rate *= 1 / (1 + self.decay * run)



class Adagrad(BaseOptimizer):

    def __init__(self,learning_rate, decay, momentum):
        BaseOptimizer.__init__(self, learning_rate, decay, momentum)

    def update_weights(self, delta, layer_input,layer_index, curr_weights, Lambda):

        eps = 1e-8

        gradient = np.dot(layer_input.T, delta) - Lambda * curr_weights

        self._gradients[layer_index] =  np.power(gradient,2)

        return curr_weights - self.learning_rate * gradient / (np.sqrt(self._gradients[layer_index]) + eps)