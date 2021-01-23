import numpy as np

class Layer:
    def __init__(self, features):
        self.features = np.zeros((features, 1))
        self.weights = np.array([])
        self.has_bias = False

    def set_init_weights(self, rows, cols):
        assert cols == self.features.shape[0] , "Shape of weights do not match corresponding layer"
        range_mat = np.random.rand(rows, cols) * 0.2
        offset = np.ones((rows, cols)) * -0.1
        self.weights = np.add(range_mat ,  offset)

    def add_bias(self):
        if not self.has_bias:
            self.features = np.concatenate(([[1]], self.features), axis=0)
            self.has_bias = True
        else:
            print("This layer already has a bias")

    def remove_bias(self):
        if self.has_bias:
            self.features = self.features[1:, :]
            self.has_bias = False
        else: 
            print("This layer does not have a bias")


class NN:
    def __init__(self):
        self.layers = []
        self.lr = 1e-2
    def add_layer(self, layer):
        assert isinstance(layer, Layer), "layer must be an instance of class Layer"
        self.layers.append(layer)
    def init_weights(self):
        assert len(self.layers) >= 2, "there should be more than one layer"
        for i in range(len(self.layers)-1):
            self.layers[i].add_bias()
            cols = self.layers[i].features.shape[0]
            rows = self.layers[i+1].features.shape[0]
            self.layers[i].set_init_weights(rows, cols)
