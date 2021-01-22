import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from crossVal import *


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
    
    





if __name__ == "__main__":

    main()
