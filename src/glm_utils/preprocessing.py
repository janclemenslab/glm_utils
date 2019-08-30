import numpy as np
from numpy.lib.stride_tricks import as_strided


def time_delay_embedding(x, y=None, window_size=None, overlap_size=None, flatten_inside_window=True, exclude_t0=True):
    """Time delay embedding with overlap.

    The embedded x will have `windows_size` fewer time points than the original x.
    If y is provided, it will be modified to match the size of the embedded x.

    Args:
        x (numpy array): 1D or 2D array
        y (numpy array, optional): If provided, will modify y to match entries in X
        window_size (int): Number of timesteps.
        overlap_size (int, optional): Equivalent to the stride of the embedding.
                                      Defaults to window_size-1 (stride=1).
        flatten_inside_window (bool, optional): Flatten 2D x-values to 1D.
                                                X will be [nb_timesteps, delays * features], otherwise [nb_timesteps, delays, features].
                                                Will always preserve shape of y features (except of the number of timesteps).
                                                Defaults to True.
        exclude_t0 (bool, optional): Exclude the current time point from the delays.
                                    `X[t]` will contain `[x[t-w-1, ..., x[t-1]]`. If `exlude_t0=False`, `x[t-w, ..., x[t]]`
                                     Done by shifting y: y=y[1:], X=X[:-1]. So this will shorten both X and y by one timestep.
                                     Defaults to True.

    Returns:
        X: delay embedded time series
        y (if provided as an argument)
    """
    if window_size is None:
        raise ValueError('Invalid arguments: window_size not specified.')

    if 0 <= x.ndim > 2:
        raise ValueError(f'Invalid arguments: x can only be one or two dimensional. Has {x.ndim} dimensions.')

    x = np.ascontiguousarray(x)  # make sure x occupies contiguous space in memory - not necessarily the case for data from xarrays

    if x.ndim == 1:
        x = x.reshape((-1, 1))

    if overlap_size is None:
        overlap_size = window_size - 1

    # get the number of overlapping windows that fit into the x
    num_windows = (x.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = x.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)
    # if there's overhang, need an extra window and a zero pad on the x
    if overhang != 0:
        num_windows += 1
        new_len = num_windows * window_size - (num_windows - 1) * overlap_size
        pad_len = new_len - x.shape[0]
        x = np.pad(x, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)

    sz = x.dtype.itemsize
    X = as_strided(
        x,
        shape=(num_windows, window_size * x.shape[1]),
        strides=((window_size-overlap_size) * x.shape[1] * sz, sz)
    )

    if not flatten_inside_window:
        X = X.reshape((num_windows, -1, x.shape[1]))

    if y is not None:
        y = y[window_size - 1::window_size - overlap_size]

    if exclude_t0:
        X = X[:-1]
    if y is not None and exclude_t0:
        y = y[1:]

    if y is not None:
        return X, y
    else:
        return X

def undersampling(x,y,window_size=1,seed=None):
    ''' Data balancing of binary data [0,1], the majority class will be reduced to the size of the minority class.

    The balanced y will have as many ones as zeros with an undersampled majority class.
    The balanced X will contain the corresponding features.

    For more sophisticated balancing, please check: http://imbalanced-learn.org/

    Args:
        x (numpy array): 1D or 2D array
        y (numpy array): 1D array, containing ones and zeros
        window_size (int): Number of timesteps.
        seed (int): Seed to initialize the random number generator

    Returns:
        X: balanced array
        y: balanced array with as many ones as zeros, the majority class will be undersampled
    '''
    np.random.seed(seed)

    # find where is response
    y_ones = np.where(y==1)[0]
    # find no response
    y_null = np.where(y==0)[0]

    # find minority and majority class
    if len(y_ones) > len(y_null):
            minority = y_null
            majority = y_ones
    elif len(y_ones) < len(y_null):
            minority = y_ones
            majority = y_null
    elif len(y_ones) == len(y_null):
            print('binary data already balanced!')
            return x, y

    # subsample majority
    freq = len(minority)/len(majority)
    n = np.floor(len(majority)*freq)
    majority = majority[np.random.choice(len(majority), int(n))]
    idx = np.r_[minority,majority]

    X = x[idx-window_size+1,...]
    y = y[idx]

    return X, y
