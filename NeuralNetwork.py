import  numpy as np
import Layers

class NeuralNetwork(object):
    '''
     Main Class to build Neural Network
    '''
    def __init__(self,alpha=0.01,epoch=200,criteria='crossEntropy',batchSize=1):
        self.layers = -1
        self.layersObject = {}
        self.criteria = criteria
        self.batchSize = batchSize
        self.learningRate = alpha
        self.epoch = epoch

    def add(self,inputDim,outputDim,transFunction):
        '''
        Method adds layer to the Network and check whether input dimension of this
        layer matches to the output dimension of previous layer or not.

        :param inputDim:  Input Dimension for the newly created layer
        :param outputDim: Output Dimension for the newly created layer
        :param transFunction: Transformation function for this new layer
        :return: None
        '''
        if self.layers != -1:
            oldLayer = self.layersObject[self.layers]
            if oldLayer.outputDim != inputDim:
                raise ValueError("Input Dimension does not match the output dimension of previous layer")

        self.layers += 1
        objLayer = Layers(inputDim,outputDim,transFunction)
        self.layersObject[self.layers] = objLayer


    def train(self,xtrain,ytrain):
        '''

        :return:
        '''