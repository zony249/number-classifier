import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

from crossval import *
from nn import *

def main():

    trainX = idx2numpy.convert_from_file("Data/TrainX")
    trainY = idx2numpy.convert_from_file("Data/TrainY")

    m = trainX.shape[0]
    rows = trainX.shape[1]
    cols = trainX.shape[2]
    
    # flattening to 60,000 x 784 matrix 
    trainX = trainX.ravel().reshape((m, rows*cols))

    # [np.array_1, np.array_2 ... np.array_k]
    k_fold_tset, k_fold_valset = create_k_folds(trainX, 10)
    
    model = NN()
    model.add_layer(Layer(5))
    model.add_layer(Layer(10))
    model.add_layer(Layer(8))
    model.init_weights()

    print(model.layers[0].weights.shape)
    print(model.layers[1].weights.shape)
    print(model.layers[2].weights.shape)


if __name__ == "__main__":

    main()
