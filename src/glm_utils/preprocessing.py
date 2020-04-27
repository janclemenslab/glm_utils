import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import TransformerMixin


def time_delay_embedding(x, y=None, indices=None, window_size=None, flatten_inside_window=True, exclude_t0=True):
    """Time delay embedding with overlap.

    The embedded `x` will have `windows_size` fewer time points (rows) than the original `x`, cutting off the first indices.
    If `y` is provided, it will be modified to match the size of the embedded `x`.
    If `indices` is provided, will return only the `x` and `y` at the indices.

    Args:
        x (numpy array): 1D or 2D array
        y (numpy array, optional): If provided, will modify y to match entries in X
        indices (numpy array, optional): If provided, will return X and y only for the indices, taking
                                         care of any shifts in the data. Indices lower than `windows_size` will be ignored.
        window_size (int): Number of timesteps.
        flatten_inside_window (bool, optional): Flatten 2D x-values to 1D.
                                                if True: X will be [nb_timesteps, delays * features], otherwise [nb_timesteps, delays, features].
                                                Will always preserve shape of y features (except for the number of timesteps).
                                                Defaults to True.
        exclude_t0 (bool, optional): Exclude the current time point from the delays.
                                    `X[t]` will contain `[x[t-w-1, ..., x[t-1]]`. If `exlude_t0=False`, `x[t-w, ..., x[t]]`
                                     Done by shifting y: y=y[1:], X=X[:-1]. So this will shorten both X and y by one timestep.
                                     Defaults to True.

    Returns:
        X: delay embedded time series
        y: y-values correctly cut, shifted and indexed if provided as an argument.
    """
    if window_size is None:
        raise ValueError('Invalid arguments: window_size not specified.')

    if 0 <= x.ndim > 2:
        raise ValueError(f'Invalid arguments: x can only be one or two dimensional. Has {x.ndim} dimensions.')

    # make sure x occupies contiguous space in memory - not necessarily the case for data from xarrays
    x = np.ascontiguousarray(x)

    if x.ndim == 1:
        x = x.reshape((-1, 1))


    # TODO: simplify code by "hard-coding" the fixed overlap size of `window_size-1`
    # if overlap_size is None:
    overlap_size = window_size - 1

    # get the number of overlapping windows that fit into the x
    num_windows = (x.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = x.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # REMOVE THIS??
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
        strides=((window_size - overlap_size) * x.shape[1] * sz, sz)
    )

    X = X.reshape((num_windows, -1, x.shape[1]))

    if flatten_inside_window:
        X = X.transpose((0, 2, 1))  # [T, feat, tau]
        X = X.reshape((X.shape[0], -1))

    if y is not None:
        y = y[window_size - 1::window_size - overlap_size]

    if exclude_t0:
        X = X[:-1]
        if y is not None:
            y = y[1:]

    if indices is not None:
        indices = np.array(indices) - window_size  # shift all indices since we lost the beginning `window_size` values from x and y
        if not exclude_t0:  # shift one back since we removed the first index from y
            indices = indices + 1
        # make sure we do not exceed bounds
        indices = indices[indices >= 0]
        indices = indices[indices < X.shape[0]]

        X = X[indices, ...]
        if y is not None:
            y = y[indices]

    if y is not None:
        return X, y
    else:
        return X


def undersampling(x, y, seed=None):
    ''' Data balancing of binary data [0,1], the majority class will be reduced to the size of the minority class.

    The balanced y will have as many ones as zeros with an undersampled majority class.
    The balanced X will contain the corresponding features.

    For more sophisticated balancing, please check: http://imbalanced-learn.org/

    Args:
        x (numpy array): 1D or 2D array
        y (numpy array): 1D array, containing ones and zeros
        seed (int): Seed to initialize the random number generator

    Returns:
        X: balanced array
        y: balanced array with as many ones as zeros, the majority class will be undersampled
    '''
    np.random.seed(seed)

    # find where is response
    y_ones = np.where(y == 1)[0]
    # find no response
    y_null = np.where(y == 0)[0]

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
    idx = np.r_[minority, majority]

    X = x[idx, ...]
    y = y[idx]

    return X, y


class BasisProjection(TransformerMixin):
    """"""

    def __init__(self, basis):
        self.basis = basis
        self.n_times = self.basis.shape[0]  # number of time points in basis
        self.n_bases = self.basis.shape[1]  # number of cosine bumps in basis

    def transform(self, X):
        """Basis projection of the data (e.g. *delay embedded data*) `X` onto `basis`.
        Shape of X should be [observations, window_size].
        Will automatically reshape 1D inputs to [1, window_size].
        """
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_times:
            raise ValueError(f'Cannot transform X with {X.shape} shape'
                             + f' and basis with {self.basis.shape} shape.'
                             + f' X.shape[1] ({X.shape[1]}) != self.basis.shape[0] ({self.basis.shape[0]}).')
        return np.dot(X, self.basis)

    def inverse_transform(self, Xt):
        """Back projects the values (e.g. filter coefficients) to original domain.
        Shape of X should be [observations, # number of bases (columns in self.basis)].
        Will automatically reshape 1D inputs to [1, # number of bases].
        """
        Xt = np.atleast_2d(Xt)
        if Xt.shape[1] != self.n_bases:
            raise ValueError(f'Cannot transform X with {Xt.shape} shape'
                             + f' and basis with {self.basis.shape} shape.'
                             + f' X.shape[1] ({Xt.shape[1]}) != self.basis.shape[1] ({self.basis.shape[0]}).')
        return np.dot(Xt, self.basis.T)
