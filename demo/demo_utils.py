from glm_utils.preprocessing import time_delay_embedding
from glm_utils.bases import raised_cosine
import numpy as np
from scipy.stats import zscore
from scipy.signal import convolve

# TODO: add noise
# TODO: add random_seed


def get_data(npoints: int = 2000, noise: float = 0, random_seed: int = 42):
    # define bases
    B = raised_cosine(1, 5, [1, 23], 9)
    window_size = B.shape[0]

    # define toy inputs (stimuli)
    ninputs = 4
    npoints = npoints + window_size
    inputs_x = np.random.random((npoints, ninputs))
    inputs_x = convolve(inputs_x, np.ones((10, 1))/10, mode='same')  # why?
    zscore_inputs_x = zscore(inputs_x, axis=0)

    # define toy filters
    filters = B[:, [4, 2, 4, 1]] - B[:, [1, 5, 4, 1]]*0.5

    # create toy signal from toy inputs and filters
    y = np.zeros((npoints,), dtype=float)
    for ii in range(ninputs):
        X = time_delay_embedding(
            x=zscore_inputs_x[:, ii], window_size=window_size)
        X = np.concatenate(
            (np.ones((window_size, *X.shape[1:]))*X[0, 0], X), axis=0)
        y = y + np.dot(X, filters[:, ii])

    return zscore_inputs_x[window_size:], y[window_size:], filters
