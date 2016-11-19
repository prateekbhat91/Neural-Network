from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np
from neuralnetwork.activation_functions import *
np.random.seed(2488)


class BaseLayer():
    '''
    Create Layer object for Neural Network and store all the informations related to it.
    '''
    __metaclass__ = ABCMeta

    def __init__(self,input_dim,output_dim,activation_function):
        '''
        :param input_dim: Input Dimension for this Layer
        :param output_dim: Output Dimension for this Layer
        :param transformFunction: Non-Linear transform Function to be applied on the output result
        '''

        assert (input_dim > 0 and isinstance(input_dim, int)  and output_dim >0 and isinstance(output_dim, int)), \
            'Input and Output dimensions should be integers greater than zero, ' \
            'instead received input_dim = {0} and output_dim = {1}'.format(input_dim, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.weights = self.__set_weights(self.input_dim, self.output_dim)
        self.bias = np.ones((1,self.output_dim))


    def __set_weights(self,input_dim, output_dim):
        return np.random.normal(0,1,output_dim*input_dim).reshape(input_dim,output_dim)

    @abstractmethod
    def derivative(self,x):
        pass

    def __str__(self):
        '''
        Overrides print function of python.
        :return: layer details.
        '''
        return '{:>10s}{:>18d}\n'.format('Input dimension', self.input_dim) + \
               '{:>10s}{:>17d}\n'.format('Output dimension', self.output_dim) + \
               '{:>s}{:>18s}\n'.format('Activation Function', self.activation_function.__name__)+ \
               '{:>10s}{:>23s}\n'.format('Weights Shape', self.weights.shape)
