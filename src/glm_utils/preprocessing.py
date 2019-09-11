import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt

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

nlin = lambda x  : np.log(x +1e-20)
invnl = lambda x : np.exp(x) -1e-20

def ff(x, c, db):
    """ Gives the raised cosine of data points `x` with classified centers
    `c` with spacing between cosine peaks `db`.
    """
    kbasis = (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(x-c)*np.pi/db/2)))+1)/2
    return kbasis

def normalizecols(A):
    """ Normalize the columns of a 2D array."""
    B = A/np.tile(np.sqrt(sum(A**2,0)),(np.size(A,0),1))
    return B

def makeBasis_StimKernel(neye, ncos, kpeaks, b, nkt = None, plot = False):
    """ Creates and plots basis of raised cosines. Adapted from Weber &
    Pillow 2017.

    Github link:
        https://github.com/aiweber/GLM_and_Izhikevich/blob/master/makeBasis_StimKernel.m

    Parameters
    ----------
    neye : int
            Number of identity basis vectors at front. This will create vector
            columns with spikes to capture the data points preceding the event.
    ncos  : int
            Number of vectors that are raised cosines. Cannot be 0 or negative.
    kpeaks : list
            List of peak positions of 1st and last cosines relative to the start
            of cosine basis vectors (e.g. [0 10])
    b : int
            Offset for nonlinear scaling.  larger values -> more linear
            scaling of vectors.
    nkt : int, optional
            Desired number of vectors in basis

    Returns
    -------
    kbasis : ndarray
        A basis of raised cosines as columns. The `neye` value defines
        the number of first columns which has identity matrix at last rows
        for dense sampling of data.

    """

    kpeaks = np.array(kpeaks)
    kdt = 1 # step for the kernel

    yrnge = nlin(kpeaks + b) # nonlinear transform, b is nonlinearity of scaling

    db = (yrnge[1]-yrnge[0])/(ncos-1) #spacing between cosine peaks
    ctrs = np.linspace(yrnge[0], yrnge[1], ncos) #nlin(kpeaks)<-weird # centers of cosines


    # mxt is for the kernel, without the nonlinear transform
    mxt = invnl(yrnge[1]+2*db)-b #!!!!why is there 2*db? max time bin
    kt0 = np.arange(0,mxt,kdt) # kernel time points/ no nonlinear transform yet
    nt = len(kt0) # number of kernel time points

    # Now we transform kernel time points through nonlinearity and tile them
    e1 = np.tile(nlin(kt0+b),(ncos,1))
    # Tiling the center points for matrix multiplication
    e2 = np.tile(ctrs,(nt,1)).T

    # Creating the raised cosines
    kbasis0 = ff(e1, e2, db)

    #Concatenate identity vectors
    nkt0 = np.size(kt0,0)  #!!!! same as nt??? Redundant or not
    a1 = np.concatenate((np.eye(neye), np.zeros((nkt0,neye))),axis=0)
    a2 = np.concatenate((np.zeros((neye,ncos)),kbasis0.T),axis=0)
    kbasis = np.concatenate((a1,a2),axis=1)
    kbasis = np.flipud(kbasis)
    nkt0 = np.size(kbasis,1)

    # Modifying number of output cosines if nkt is given
    if nkt == None:
        pass
    elif nkt0 < nkt: # if desired time samples greater, add more zero basis
        kbasis = np.concatenate((np.zeros(kbasis,(nkt-nkt0,ncos+neye))),axis=0)
    elif nkt0 > nkt: # if desired time samples less, get the last nkt columns of cosines
        kbasis = kbasis[:,:nkt]

    kbasis = normalizecols(kbasis)

    if plot:
        plt.figure()
        plt.plot(kbasis)
        plt.title(f'ncos: {ncos} '+
                  f'neye: {neye} '+
                  f'nkt: {nkt}\n'+
                  f'kpeaks: {kpeaks} '+
                  f'nonlinearity (b): {b}')
    return kbasis
