# Neural-Network:
This repository is an ongoing work of implementing neural network in Python.

We have tried to keep the function and variable names similar to [scikit-learn](http://scikit-learn.org/stable/) and [keras](https://keras.io/). This repository will be updated regularly so stay tuned.


## Optimizers available:
1. **Stochastic Gradient Descent**: use optimizer='SGD'.
    1. **Vanilla Backpropogation**: set momentum to 0 while creating neural network object.
    2. **Classical momentum**: set a value of momentum while creating neural network object.
2. **Nesterov Momentum**: use optimizer='Nesterov' and set a value of momentum while creating neural network object.
3. **RMSProp**: use optimizer='RMSProp' and set a value of decay while creating neural network object.
4. **Adagrad**: use optimizer='Adagrad'.
5. **Adam**:  use optimizer='Adam'.

## Layers available:
1. **Sigmoid**
2. **Tanh**
3. **ReLU**
4. **LeakyReLU**


# Usage:
Divide your data into scikit-learn format of numpy arrays, i.e. have Xtrain, ytrain ,Xtest and ytest where,

1. Xtrain = trainig examples
2. ytrain = training labels
3. Xtest = test examples
4. ytest = test labels 


The below example is for classification.
```python

'Load the libraries'
from neuralnetwork.NeuralNetwork import NeuralNetwork
from neuralnetwork.Layers.layers import *
from neuralnetwork.metrics import *
from neuralnetwork.preprocessing import label_encoder

'Lets make a neural network with 2 hidden layers with sigmoid activation'
'Define your own arguments for neural netowrk, this is just an example'
nn = NeuralNetwork(alpha=0.01, epoch=300, criteria='cross_entropy', optimizer='SGD',metric='AS',
                 batch_size=100, verbose=False, decay=0.001, momentum=0.0, random_seed=None)
nn.add(Sigmoid_Layer(input_dim1,output_dim1))              #Input layer
nn.add(Sigmoid_Layer(output_dim1,output_dim2))      #Hidden layer
nn.add(Sigmoid_Layer(output_dim2,numClasses))      #Hidden layer

'Train the neural network of '
nn.train(Xtrain,ytrain)

'predict using the trained neural network'
pred = nn.predict(Xtest)

accuracy = accuracy_score(ytest, pred)
```
## Note:
For regression the output dimension of the last layer should be 1.
