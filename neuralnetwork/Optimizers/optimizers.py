from base_optimizer import BaseOptimizer
from neuralnetwork.utils import generate_batches
import numpy as np


class SGD(BaseOptimizer):
    '''
    Batch Stochastic gradient descent optimizer.
    '''
    def __init__(self,learning_rate,decay,momentum):
        BaseOptimizer.__init__(self,learning_rate,decay,momentum)


    def backprop(self, nn, x, y):
        """
        Backpropogation algorithm
        :param nn: neural network object
        :param x: training data
        :param y: training labels
        :return: None
        """

        'generate batches of data depending on the batch size.'
        for indices in generate_batches(x.shape[0],batch_size=nn.batch_size):
            xtrain = x[indices]
            ytrain = y[indices]


            print ('xtrain batch shape'),xtrain.shape
            print ('ytrain batch shape'), ytrain.shape

            'Get the output of each layer' \
            'Example: If we have one input and two hidden layer, the output of forwardprop' \
            'will be:' \
            '[ (input),(input),(input,output)]'


            each_layer_output = self._forwardprop(nn,xtrain)

            reversedlayers = range(len(nn._layersObject))[::-1]
            outputlayerindex = reversedlayers[0]


            for i in reversedlayers:
                print('layer out: ',each_layer_output[i][0].shape)
                if i == outputlayerindex:
                    delta = np.multiply(nn._layersObject[i].derivative(each_layer_output[i][1]),
                                        nn.criteria(ytrain, each_layer_output[i][1]))

                    sum = np.dot(delta, nn._layersObject[outputlayerindex].weights.T)
                    gradient_weight = np.dot(each_layer_output[i ][0].T, delta) + nn.Lambda * nn._layersObject[
                        i].weights
                    gradient_bias = (nn.learningRate) * np.sum(delta, axis=0, keepdims=True)

                else:
                    delta = np.multiply(nn._layersObject[i].derivative(each_layer_output[i+1][0]), sum)
                    sum = np.dot(delta, nn._layersObject[i].weights.T)
                    gradient_weight = np.dot(each_layer_output[i + 1][0].T, delta) + nn.Lambda * nn._layersObject[
                        i].weights
                    gradient_bias = (nn.learningRate) * np.sum(delta, axis=0, keepdims=True)


                'calculate gradient'
                print('delta shape'), delta.shape
                print ('weight shape:'),nn._layersObject[i].weights.shape
                print ('gradient weight shape:'), gradient_weight.shape
                print ('gradient bias shape'), gradient_bias.shape

                nn._layersObject[i].weights -= (nn.learningRate) * gradient_weight

                print ('bias shape'), nn._layersObject[i].bias.shape
                raw_input()
                nn._layersObject[i].bias -= gradient_bias





