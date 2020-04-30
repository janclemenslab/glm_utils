from glm_utils.preprocessing import time_delay_embedding
from glm_utils.bases import raised_cosine
import numpy as np
from scipy.stats import zscore
from scipy.signal import convolve


def get_data(npoints: int = 2000):

    # define basis
    B = raised_cosine(1, 5, [1, 23], 9)
    window_size = B.shape[0]

    # define stimuli
    ninputs = 4
    npoints = npoints + window_size
    x = np.random.random((npoints, ninputs))
    x = convolve(x, np.ones((10, 1))/10, mode='same')  # make things a little smoother
    zscored_x = zscore(x, axis=0)

    # define filters
    filters = B[:, [4, 2, 4, 1]] - B[:, [1, 5, 4, 1]]*0.5
    filters[:, 1] *= 1.5
    filters[:, 2] *= 0.5

    # create response
    y = np.zeros((npoints,), dtype=float)
    for ii in range(ninputs):
        X = time_delay_embedding(x=zscored_x[:, ii], window_size=window_size)
        X = np.concatenate((np.ones((window_size, *X.shape[1:]))*X[0, 0], X), axis=0)
        y = y + np.dot(X, filters[:, ii])

    return zscored_x[window_size:], y[window_size:], filters
