import numpy as np
from utils import *

class Layer:
    def __init__(self, features, reg=0):
        self.features = np.zeros((features, 1))
        self.error = np.zeros((features, 1))
        self.weights = np.array([])
        self.delta_accumulatror = np.array([])
        self.has_bias = False

    def set_init_weights(self, rows, cols):
        # Layer is first prepended with bias before weights are set
        assert cols == self.features.shape[0] + 1 , "Shape of weights do not match corresponding layer"
        range_mat = np.random.rand(rows, cols) * 0.2
        offset = np.ones((rows, cols)) * -0.1
        self.weights = np.add(range_mat ,  offset)
        self.delta_accumulatror = np.zeros((rows, cols))

    def add_bias(self):
        
        self.features = np.concatenate(([[1]], self.features), axis=0)
    def remove_bias(self):
        self.features = self.features[1:, :]

    def set_val(self, np_arr):
        # set val assumes bias has not been added
        self.features = np_arr

class NN:
    def __init__(self):
        self.layers = []
        self.lr = 1e-2
        self.weights_set = False
        
        self.trainX = np.array([])
        self.trainY = np.array([])
        self.crossvalX = np.array([])
        self.crossvalY = np.array([])

    def add_layer(self, layer):
        assert isinstance(layer, Layer), "layer must be an instance of class Layer"
        self.layers.append(layer)
    def init_weights(self):
        assert len(self.layers) >= 2, "there should be more than one layer"
        for i in range(len(self.layers)-1):
            cols = self.layers[i].features.shape[0]
            rows = self.layers[i+1].features.shape[0]
            self.layers[i].set_init_weights(rows, cols+1)
        self.weights_set = True

    def set_lr(self, lr):
        self.lr = lr

    def load_train_from_array(self, trainX, trainY):
        self.trainY = np.zeros((trainY.shape[0], 10)) # (m x 10) because there are 10 classes
        for i in range(trainY.shape[0]):
            self.trainY[i][trainY[i]] = 1
        self.trainX = trainX

    def set_input(self, x):
        x.flatten().reshape(self.layers[0].features.shape)

    def feed_forward(self, x):
        assert self.weights_set, "Weights are not set"

        self.set_input(x)

        for i in range(len(self.layers)-1):
            self.layers[i].add_bias()
            self.layers[i+1].set_val(np.matmul(self.layers[i].weights, self.layers[i].features))
        hyp = self.layers[len(self.layers)-1].features
        return hyp
    def back_prop(self, y):
        last = len(self.layers)-1
        self.layers[last].error = np.add(self.layers[last].features, -1 * y)
        for i in range(1, last):
            self.layers[last - i].error = np.matmul(self.layers[last-i].weights.transpose(), self.layers[last-i+1].error)
            derivative_sig = sigmoid(np.matmul(self.layers[last-i-1].weights, self.layers[last-i-1].features)) # shape without bias
            derivative_sig = np.multiply(derivative_sig, np.add(np.ones(derivative_sig.shape), -1* derivative_sig)) # sig (1 - sig)
            derivative_sig = np.concatenate((np.array([[i]]), derivative_sig), axis=0) # add bias to get compatible shape.
            print(self.layers[last-i].error.shape, derivative_sig.shape) 
            self.layers[last - i].error = np.multiply(self.layers[last-i].error, derivative_sig)
            self.layers[last - i].error = self.layers[last - i].error[1:, :] # removes bias unit

        
        ######### NEXT STEP ###########
        # - y will be supplied by self.train() 
        #   (self.train is yet to be made)
        # - back_prop -- finished calculating errors, now 
        #   update each weight matrix accumulator with errors
        # - 
        #
        #
        ###############################
