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
   :return    : R-squared(coefficient of determination)
   '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)
    if true.shape == pred.shape:
        numerator   = np.sum(np.square(np.subtract(true,pred)))
        denominator = np.sum(np.square(np.subtract(true,np.mean(true))))
        return 1-(numerator/denominator)
    else:
        raise ValueError("true and pred dimensions do not match.")


def confusion_matrix(true,pred):
    '''

    :param true: vector containing all the true classes
    :param pred: vector containing all the predicted classes
    :return    : confusion matrix
    '''
    true = utils.convert_to_1D(true)
    pred = utils.convert_to_1D(pred)

    if true.shape == pred.shape:
        numclass = len(np.unique(true))
        labelEncoder = utils.label_encoder()
        labelEncoder.fit(true)
        true = labelEncoder.transform(true)
        pred = labelEncoder.transform(pred)

        cm = [[0 for _ in range(numclass)] for _ in range(numclass)]
        cm = np.array(cm)
        for p, e in zip(pred,true):
            cm[e][p] += 1
        return cm

    else:
        raise ValueError("true and pred dimensions do not match.")

if __name__ == '__main__':
    b = [1, 2, 4, 1, 2, 5, 4, 5, 2, 3, 4, 2]
    b = np.array(b)
    pred = [2,1,4,1,2,4,5,2,3,4,2,5]
    cm = confusion_matrix(b,pred)
    print(cm)