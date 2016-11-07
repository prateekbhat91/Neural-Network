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

class label_encoder():
    '''
    Encodes the elements of an array from range 0 to num classes -1.
    This function is very similar to scikit-learn LabelEncoder.
    '''
    def __init__(self):
        self._classes = None
        self.transformation = None

    def fit(self,array):
        '''
        :param array: 1D input numpy array.
        :return: None
        '''
        if array.ndim == 1:
            self._classes, self._transformation = np.unique(array, return_inverse=True)
        else:
            raise ValueError("1D array required")

    def transform(self,array):
        '''
        :param array: 1D numpy array to be transformed
        :return: label encoded array.
        '''
        if array.ndim == 1:
            return np.searchsorted(self._classes, array)
        else:
            raise ValueError("1D array required")


    def inverse_transform(self,array):
        '''
        :param array: 1D numpy array to be transformed.
        :return: label decoded array.
        '''
        if array.ndim == 1:
            return self._classes[array]
        else:
            raise ValueError("1D array required")