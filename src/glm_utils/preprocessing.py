import numpy as np
from numpy.lib.stride_tricks import as_strided


def time_delay_embedding(data, window_size, overlap_size=0, flatten_inside_window=True, preserve_size=True):
    """Time delay embedding of the data with overlap.
    
    Args:
        data (numpy array): [description]
        window_size (int?): [description]
        overlap_size (int, optional): [description]. Defaults to 0.
        flatten_inside_window (bool, optional): [description]. Defaults to True.
        preserve_size (bool, optional): Defaults to False.
    
    Returns:
        chunked data [nb_timesteps, window_size] where nb_timesteps depends on the overlap_size and preserve_size parameters
    """
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    if overhang != 0:
        num_windows += 1
        new_len = num_windows*window_size - (num_windows-1)*overlap_size
        pad_len = new_len - data.shape[0]
        data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
    
    sz = data.dtype.itemsize
    ret = as_strided(
        data,
        shape=(num_windows, window_size*data.shape[1]),
        strides=((window_size-overlap_size)*data.shape[1]*sz, sz)
    )

    if preserve_size:
        new_len = ret.shape[0]
        pad_len = data.shape[0] - new_len
        ret = np.pad(ret, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
    
    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))
