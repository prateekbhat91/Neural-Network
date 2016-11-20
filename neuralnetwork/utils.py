'''
This file includes util functions required while building neural network.
'''

import numpy as np
np.random.seed(1024)

def convert_to_1D(tensor):
    '''
    Converts a numpy array into an array of 1 dimension.
    :param tensor: input numpy array
    :return: 1D array
    '''
    return np.ravel(tensor)

def checkXandY(x,y):
    '''
    checks the number of points in two numpy array.
    :param x: numpy array
    :param y: numpy array
    :return: None
    '''
    assert (x.shape[0] == y.shape[0]), "Number of data points not equal"


def change_labels(ytrain):
    '''
    change the labels according to the data required by neural network.
    Example: ytrain = [1,0,1]
    change it to: [[0,1],[1,0],[0,1]]
    :param ytrain: numpy array of 1D dimension.
    :return: numpy array where the class label is triggered.
    '''

    ytrain = ytrain.ravel()
    numclass = len(np.unique(ytrain))
    changed_ytrain = []
    for label in ytrain:
        temp = np.zeros(numclass)
        temp[int(label)] =1
        changed_ytrain.append(temp)

    return np.array(changed_ytrain)


def generate_batches(data_size,batch_size):
    '''
    Generate batches of data based on the batch size.
    :param data_size: total number of data points.
    :param batch_size: batch size of type int
    :return:
    '''
    assert (isinstance(batch_size, int)), 'batch_size should be of type int, {0} not supported.'.format(type(batch_size))
    assert (batch_size > 0), 'batch_size should be greater than zero.'
    ind = np.array([i for i in range(data_size)])
    np.random.shuffle(ind)

    for i in range(0, data_size, batch_size):
        yield ind[i:i + batch_size]
