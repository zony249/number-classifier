import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

from crossval import *
from nn import *
from utils import *

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
    model.add_layer(Layer(12))
    model.add_layer(Layer(3))
    model.init_weights()


    x = np.array([-2, -1, 0, 1, 2])
    print(x)
    
    hyp = model.feed_forward(x)
    print(hyp)
    model.back_prop(np.array([[-2], [-3], [-4]]))

    print(trainY)
    model.load_train_from_array(trainX, trainY)

if __name__ == "__main__":

    main()
