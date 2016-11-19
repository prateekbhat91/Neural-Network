from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np
from neuralnetwork.utils import generate_batches
from neuralnetwork.metrics import accuracy_score

class BaseOptimizer():

    __metaclass__ = ABCMeta

    def __init__(self,learning_rate, decay,momentum):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum

    def _forwardprop(self,nn,x,trace=True):
        '''
        Forward propogation method for neural network.
        :param nn: neural network
        :param x: input data
        :param trace: If trace is true, function return the input of each layer and output of the last layer.
        If trace is False, the function just returns the output of last layer.
        :return: If trace is true, function return the input of each layer and output of the last layer.
        If trace is False, the function just returns the output of last layer.

        'Example: If we have one input and two hidden layer, the output of forwardprop with trace = True is' \
        '[ (input),(input),(input,output)]'
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


    def backprop(self, nn, x, y):
        """
        Backpropogation algorithm
        :param nn: neural network object
        :param x: training data
        :param y: training labels
        :return: None
        """


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
                                                    curr_weights=nn._layersObject[i].weights, Lambda=nn.Lambda)

                    nn._layersObject[i].bias = self.update_bias(delta, curr_bias=nn._layersObject[i].bias)



            # pred = self._forwardprop(nn,x,trace=False)
            # print ('pred shape'), pred.shape
            # print('epoch:{0}, learning rate:{1}, loss:{2}'.format(run,self.learning_rate,
            #                                                       accuracy_score(y,pred)))
            # raw_input()
            run += 1




    def update_weights(self,delta, layer_input, curr_weights, Lambda):
        pass

    def update_bias(self, delta, curr_bias):
        pass

    def calculate_delta(self, derivative, loss):
        return np.multiply(derivative, loss)


    def decay_learning_rate(self, run):
        self.learning_rate *= 1 / (1 + self.decay * run)





