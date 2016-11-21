from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np
from neuralnetwork.utils import generate_batches
from neuralnetwork.metrics import accuracy_score
from neuralnetwork.utils import change_labels
import copy

'This file contains implementation of base class for optimizers.'



class BaseOptimizer():
    '''
    This is a base class and should not be instantiated.
    '''

    __metaclass__ = ABCMeta

    def __init__(self,learning_rate, decay,momentum):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self._gradients = {}

    def _forwardprop(self,nn,x,trace=True):
        '''
        Forward propogation method for neural network.
        :param nn: neural network
        :param x: input data
        :param trace: (Binary)Default if True
        :return: If trace is true, function return the input of each layer and output of the last layer.
        If trace is False, the function just returns the output of last layer.

        'Example: If we have one input and two hidden layer, the output of forwardprop with trace = True is' \
        '[ (input, None),(input,None),(input,output)]'
        '''
        if trace == True:
            outputs = []
        input = x


        for i in range(len(nn._layersObject)):
            # print('input shape'), input.shape
            layer = nn._layersObject[i]
            Output = layer.activation_function(np.add(np.dot(input, layer.weights), layer.bias))
            # print ('output data shape'), Output.shape
            # raw_input('Inside forwardprop')
            if trace == True:
                if i == len(nn._layersObject)-1:
                    outputs.append((input, Output))
                else:
                    outputs.append((input, None))
            input = Output

        if trace == True:
            return outputs

        return Output


    def backprop(self, x, y,nn):
        """
        Backpropogation algorithm
        :param nn: neural network object
        :param x: training data
        :param y: training labels
        :return: None
        """
        'make a dictironary to store the gradients for each layer/ to be used for momentum'
        for i in range(len(nn._layersObject)):
            self._gradients[i] = 0

        self.set_smooth_gradients(nn)

        y_real = copy.copy(y)

        if nn._layersObject[nn._layernum].output_dim != 1:
            y = change_labels(y)


        run = 0
        while run < nn.epoch:

            self.decay_learning_rate(run)

            'generate batches of data depending on the batch size.'
            for indices in generate_batches(x.shape[0], batch_size=nn.batch_size):
                xtrain = x[indices]
                ytrain = y[indices]

                each_layer_output = self._forwardprop(nn, xtrain)


                reversedlayers = range(len(nn._layersObject))[::-1]
                outputlayerindex = reversedlayers[0]

                for i in reversedlayers:
                    if i == outputlayerindex:
                        delta = self.calculate_delta(nn._layersObject[i].derivative(each_layer_output[i][1])
                                                      , nn.criteria(ytrain, each_layer_output[i][1]))


                    else:
                        delta = self.calculate_delta(nn._layersObject[i].derivative(each_layer_output[i+1][0])
                                                      , np.dot(delta, nn._layersObject[i+1].weights.T))



                    nn._layersObject[i].weights = self.update_weights(delta, layer_input=each_layer_output[i][0],
                                                    layer_index=i,curr_weights=nn._layersObject[i].weights, Lambda=nn.Lambda)

                    nn._layersObject[i].bias = self.update_bias(delta, curr_bias=nn._layersObject[i].bias)


            if nn.verbose == True:

                pred = self._forwardprop(nn,x, trace=False)
                'calculate the training loss(SSE)'
                loss = np.sum(np.power(np.subtract(pred,y),2))/pred.shape[0]

                print('epoch:{0}/{1}, learning rate:{2}, loss:{3}'.format(run,nn.epoch,self.learning_rate,
                                                                     loss))

            run += 1



    @abstractmethod
    def update_weights(self,delta, layer_input, layer_index,curr_weights, Lambda):
        '''
        This is an abstract function and should be defined for each derived class.
        :param delta:
        :param layer_input: input of each layer
        :param layer_index: index of layer to be updated.
        :param curr_weights: current weights of layer
        :param Lambda: regularization parameter.
        :return: updated weights
        '''
        pass



    def update_bias(self, delta, curr_bias):
        '''
        This is an abstract function and should be defined for each derived class.
        :param delta:
        :param curr_bias: current bias of layer.
        :return: updated bias.
        '''
        gradient = self.learning_rate * np.sum(delta, axis=0, keepdims=True)

        new_bias = curr_bias - self.learning_rate * gradient
        return new_bias

    def calculate_delta(self, derivative, loss):
        '''
        This function calculates the value of delta.
        :param derivative: derivate of the layer.
        :param loss: loss of each layer.
        :return: delta
        '''
        return np.multiply(derivative, loss)


    def decay_learning_rate(self, run):
        '''
        This function decays the learning rate depending
        on the number of runs and decay rate.
        :param run:
        :return:
        '''
        pass


    def set_smooth_gradients(self, nn):
        '''
        Only to be used for ADAM. It initializes the smooth gradients for each layer to 0.
        :param nn: neural network
        :return: None
        '''
        pass