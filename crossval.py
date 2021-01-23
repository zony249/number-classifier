import numpy as np
import matplotlib.pyplot as plt

def create_k_folds(tset, k):
    folds = []
    for i in range(k):
        folds.append(tset[i::k, :])
    

    trainsets = []
    valsets = []

    concatenate_container = []
    for i in range(k):
        for j in range(k):
            if j == i:
                valsets.append(folds[j])
            else:
                concatenate_container.append(folds[j])
        trainsets.append(np.concatenate(concatenate_container, axis=0))
        concatenate_container = []
    for i in trainsets:
        print(i.shape)

    for i in valsets:
        print(i.shape)

    return (trainsets, valsets)

