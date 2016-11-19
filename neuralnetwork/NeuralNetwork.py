import numpy as np
from Layers.layers import *
from utils import *
from cost_functions import *
from collections import defaultdict
from neuralnetwork.Optimizers.optimizers import *

optimizers = {'SGD':SGD}
Criteria = {'SSE':SSE, 'cross_entropy':cross_entropy}

class NeuralNetwork(object):
    """
    Main Class to build Neural Network
    """

    def __init__(self, tol=0.0001, alpha=0.01, epoch=200, criteria=cross_entropy, optimizer='SGD',
                 batch_size=1, verbose=False,Lambda=0.005, decay=0.0, momentum=0.0, random_seed=None):
        '''
        :param tol: tolerance of change in error.
        :param alpha: learning rate.
        :param epoch: number of passes over data
        :param criteria: criteria for cost function(to be implemented) By default sum of squared error.
        :param batch_size: The size of the batch to be processed. By default the value is 1.
        :param verbose: If True print out the error for each epoch.
        :param decay: Decay rate of learning rate by 1 / (1 + decay * runs)
        :param momentum: value ranges from [0,1]. Use a low learning rate with high momentum and vice versa.
        :param random_seed: random seed for regeneration of results.
        '''

        'Check the arguments of neural network.'
        assert (criteria in ['SSE', 'cross_entropy']), "{0} not supported as a loss function".format(criteria)
        assert (epoch > 0), "epoch should be greater than zero"
        assert (verbose == True or verbose == False), "{0} not supported. Verbose supports True and False".format(
            verbose)
        assert (momentum >= 0 and momentum <= 1), 'momentum should be between [0,1]'
        assert (optimizer in optimizers), '{0} not supported'.format(optimizer)

        self._layernum = -1
        self._layersObject = defaultdict()  # stores all the layers of the network.
        self.criteria = Criteria[criteria]
        self.batch_size = batch_size
        self.learningRate = alpha
        self.epoch = epoch
        self.verbose = verbose
        self.tol = tol
        self.Lambda = Lambda
        self.momentum = momentum
        self.decay = decay
        self.random_seed = random_seed
        self.optimizer = optimizers[optimizer](learning_rate=self.learningRate,decay=self.decay,momentum=self.momentum)


    'Iterate over the layers in neural network'
    def __iter__(self):
        return [i for i in self._layersObject]

    def __str__(self):
        return '====Neural network summary====\n' \
               'Layers:\t{0}\n' \
               'Criteria:\t{1}\n' \
               '==============================' .format(len(self._layersObject),self.criteria)

    def add(self, layer_object ):
        """
        Method adds layer to the network and checks if the input dimension
        of the current layer matches the output dimension of previous layer.

        :param layer_object: layer to be added to neural network.
        :return: None
        """
        if self._layernum != -1:
            oldLayer = self._layersObject[self._layernum]
            assert (oldLayer.output_dim == layer_object.input_dim), \
                "Input dimension of current layer does not match the output dimension of previous layer," \
                "Input dimension of current layer = {0}, Output dimension of previous layer = {1}."\
                .format(layer_object.inputDim,oldLayer.output_dim)

        self._layernum += 1
        self._layersObject[self._layernum] = layer_object

    # @classmethod
    def train(self,nn,x, y):
        """
        :return:
        """
        checkXandY(x,y)
        self.optimizer.backprop(nn,x,y)



    def predict(self,x, nn):
        pred = []
        yhat = self.optimizer._forwardprop(nn,x, trace=False)
        for i in range(x.shape[0]):
            pred.append(yhat[i].argmax())
        return np.array(pred)






