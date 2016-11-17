import numpy as np
np.random.seed(2488)

class Layer(object):
    '''
    Create Layer object for Neural Network and store all the informations related to it.
    '''
    def __init__(self,inputDim,outputDim,transformFunction=None):
        '''
        :param inputDim: Input Dimension for this Layer
        :param outputDim: Output Dimension for this Layer
        :param transformFunction: Non-Linear transform Function to be applied on the output result
        '''
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.transformFunction = transformFunction
        self.weights = np.random.normal(0,np.sqrt(2/inputDim),outputDim*inputDim).reshape(inputDim,outputDim)
        self.bias = np.zeros((1,outputDim))