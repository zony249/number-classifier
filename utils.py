import numpy as np

def sigmoid(x):
    denom = np.add(np.ones(x.shape), np.exp(-1 * x))
    return np.divide(np.ones(x.shape), denom)

