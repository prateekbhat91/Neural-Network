import utils
import numpy as np


'Classification'
def accuracy_score(true,pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: accuracy of classification
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    if true.shape == pred.shape:
        return np.count_nonzero(np.subtract(true,pred))/true.shape[0]
    else:
        raise ValueError("true and pred dimensions do not match.")



'Regression'
def mean_absolute_error(true,pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: mean absolute error.
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    if true.shape == pred.shape:
        return np.sum(np.fabs(np.subtract(true,pred)))/true.shape[0]
    else:
        raise ValueError("true and pred dimensions do not match.")

def mean_squared_error(true,pred):
    '''
    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return: mean squared error.
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    if true.shape == pred.shape:
        return np.sum(np.square(np.subtract(true, pred))) / true.shape[0]
    else:
        raise ValueError("true and pred dimensions do not match.")


def r2_score(true,pred):
    '''
   :param true: vector containing all the true classes
   :param pred: vector containing all the predicted classes
   :return: R-squared(coefficient of determination)
   '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    if true.shape == pred.shape:
        numerator = np.sum(np.square(np.subtract(true,pred)))
        denominator = np.sum(np.square(np.subtract(true,np.mean(true))))
        return 1-(numerator/denominator)
    else:
        raise ValueError("true and pred dimensions do not match.")