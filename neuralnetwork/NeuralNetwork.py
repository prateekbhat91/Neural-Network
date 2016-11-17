import numpy as np
import Layers


class NeuralNetwork(object):
    """
    Main Class to build Neural Network
    """

    def __init__(self, alpha=0.01, epoch=200, criteria='crossEntropy', batchSize=1,verbose = False):
        '''

        :param alpha: learning rate(can also be changed during training of network)
        :param epoch: number of passes over data.
        :param criteria: optimization function to be used.
        :param batchSize: the batch size to be used while training.
        :param verbose: print the details while training(default False)
        '''
        self.layers = -1
        self.layersObject = {}  # stores all the layers of the network.
        self.criteria = criteria
        self.batchSize = batchSize
        self.learningRate = alpha
        self.epoch = epoch
        self.verbose = verbose

    'Iterate over the layers in neural network'
    def __iter__(self):
        return [i for i in self.layersObject]

    def __str__(self):
        return '====Neural network summary====\n' \
               'Layers:\t{0}\n' \
               'Criteria:\t{1}\n' \
               '==============================' .format(len(self.layersObject),self.criteria)

    def add(self, inputDim, outputDim, transFunction):
        """
        Method adds layer to the network and checks if the input dimension
        of the current layer matches the output dimension of previous layer.

        :param inputDim:  Input Dimension for the newly created layer.
        :param outputDim: Output Dimension for the newly created layer.
        :param transFunction: Transformation function for the layer.
        :return: None
        """
        if self.layers != -1:
            oldLayer = self.layersObject[self.layers]
            assert (oldLayer.outputDim == inputDim), "Input Dimension does not match the output dimension of previous layer"

        self.layers += 1
        self.layersObject[self.layers] = Layers(inputDim, outputDim, transFunction)

    def train(self, xtrain, ytrain):
        """
        :return:
        """


if __name__ == "__main__":
    nn = NeuralNetwork()
    print(nn.__all__)