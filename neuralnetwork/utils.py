'''
This file includes util functions required while building neural network.
'''

import numpy as np


def convert_to_1D(tensor):
    '''
    Converts a numpy array into an array of 1 dimension.
    :param tensor: input numpy array
    :return: 1D array
    '''
    return np.ravel(tensor)

