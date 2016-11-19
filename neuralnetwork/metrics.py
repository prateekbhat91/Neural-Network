from __future__ import division, absolute_import
import neuralnetwork.utils as utils
from neuralnetwork.preprocessing import label_encoder
import numpy as np

'''
This file contains implementation of common metric functions for
classification and regression.
'''


'Classification'
def accuracy_score(true, pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: accuracy of classification
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    return (true.shape[0] - np.count_nonzero(np.subtract(true, pred)) )/ true.shape[0]



def confusion_matrix(true, pred):
    '''
    :param true: numpy array containing all the true classes.
    :param pred: numpy array containing all the predicted classes.
    :return    : confusion matrix
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)

    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    numclass = len(np.unique(true))

    # encode the classes with integers ranging from 0 to numclass-1.
    labelEncoder = label_encoder()
    labelEncoder.fit(true)
    true = labelEncoder.transform(true)
    pred = labelEncoder.transform(pred)

    # create confusion matrix.
    # Rows indicate the true class and column indicate the predicted class.
    cm = np.array([np.zeros(numclass) for _ in range(numclass)])
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm



'Regression'
def mean_absolute_error(true, pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: mean absolute error.
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    return np.sum(np.fabs(np.subtract(true, pred))) / true.shape[0]




def mean_squared_error(true, pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: mean squared error.
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    return np.sum(np.square(np.subtract(true, pred))) / true.shape[0]


def r2_score(true, pred):
    '''
   :param true: vector containing all the true classes
   :param pred: vector containing all the predicted classes
   :return    : R-squared(coefficient of determination)
   '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    assert (true.shape == pred.shape), "true and pred dimensions do not match."
    numerator = np.sum(np.square(np.subtract(true, pred)))
    denominator = np.sum(np.square(np.subtract(true, np.mean(true))))
    return 1 - (numerator / denominator)