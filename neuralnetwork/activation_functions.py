from __future__ import division
import numpy as np

'''
This file contains implementation of various activation functions.
'''

#Random seed for consistency in results.
np.random.seed(32)

def sigmoid(tensor):
    '''
    Sigmoid transfer.
    :param tensor: numpy array
    :return: numpy array of same shape, where each input element has been modified.
    '''
    return 1/(1+np.exp(-tensor))


def tanh(tensor):
    '''
    Hyperbolic tangent.
    :param tensor: numpy array.
    :return: numpy array of same shape, where each input element has been modified.
    '''
    return np.tanh(tensor)


def ReLU(tensor):
    '''
    Rectified Linear Unit
    :param tensor:numpy array
    :return: numpy array of same shape, where each input element has been rectified with noise.
    '''
    return np.maximum(0,tensor)

def Noisy_ReLU(tensor):
    '''
    Nosiy Rectified Linear Unit. The noise is chosen from a
    gaussian distribution of zero mean and unit variance.
    :param tensor: numpy array
    :return: numpy array of same shape, where each input element has been rectified.
    '''
    return np.maximum(0,tensor+np.random.normal(0,1))

def Leaky_ReLU(tensor):
    '''
    Leaky Rectified Linear Unit.
    :param tensor: numpy array
    :return: numpy array of same shape.
    '''
    tensor[tensor<0] *= 0.01
    return tensor


def softmax(tensor):
    '''
    Softmax function(normalized exponential): changes 1D array of arbitrary real
    values into 1D array of real values in the range (0,1) that add upto 1.
    :param tensor: 1D numpy array.
    :return: softmax transferred numpy array of same dimension.
    '''
    'check the dimension of tensor'
    if tensor.ndim == 1:
        exp = np.exp(tensor)
        return exp / np.sum(exp)
    else:
        raise ValueError('Input tensor is not a 1D array')
