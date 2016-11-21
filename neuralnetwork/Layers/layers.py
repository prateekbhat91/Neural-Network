from base_layer import BaseLayer
from neuralnetwork.activation_functions import *

class Sigmoid_Layer(BaseLayer):

    def __init__(self,input_dim,output_dim):
        '''
        :param input_dim : input dimension of layer
        :param output_dim: output dimension of layer
        '''
        BaseLayer.__init__(self,input_dim,output_dim, activation_function = sigmoid)

    def derivative(self, x):
        '''
        :param x: sigmoid activated input.
        :return: derivative of sigmoid
        '''
        return np.multiply(x, 1 - x)


class Tanh_Layer(BaseLayer):

    def __init__(self,input_dim,output_dim):
        '''
        :param input_dim : input dimension of layer
        :param output_dim: output dimension of layer
        '''
        BaseLayer.__init__(self,input_dim,output_dim, activation_function = tanh)

    def derivative(self, x):
        '''
        :param x: tanh activated input.
        :return: derivative of sigmoid
        '''
        return 1 - np.power(x, 2)

class ReLU_Layer(BaseLayer):

    def __init__(self,input_dim,output_dim):
        '''
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        '''
        BaseLayer.__init__(self, input_dim, output_dim, activation_function=ReLU)

    def derivative(self, x):
        '''
        :param x:
        :return: derivate of tanh function
        '''
        return np.clip(x > 0, 0.0, 1.0)


class LeakyReLU_Layer(BaseLayer):

    def __init__(self,input_dim,output_dim):
        '''
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        '''
        BaseLayer.__init__(self, input_dim, output_dim, activation_function=Leaky_ReLU)


    def derivative(self, x):
        '''
        :param x: Leaky ReLU activated input.
        :return: derivate of Leaky ReLU with leakage of 0.01.
        '''
        return np.clip(x > 0, 0.01, 1.0)
