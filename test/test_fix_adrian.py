# %%
from glm_utils.preprocessing import time_delay_embedding
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
x = np.load('test/fix_adrian_data.npy')
# %% test default
w = 10

X = time_delay_embedding(x, window_size=w)
print(x.shape, X.shape)
print(f'nb nan vals = {np.sum(np.isnan(X))}')
assert np.sum(np.isnan(X))==0

print(np.__version__)

# %%
import xarray_behave as dst

data_folder = data_folder = f"/Volumes/ukme04/apalaci/code/dat"# f"/Volumes/ukme04/#Common/backlight"# CHANGE THIS FOR YOU
# expsetup = 'backlight'
datename = 'localhost-20190703_164840'
feature_name = 'velocity_forward'
ifly = 5

metrics_dataset = dst.load(f"{data_folder}/{datename}/{datename}_metrics.zarr", lazy=False)

x0 = metrics_dataset.abs_features.sel(absolute_features=feature_name, flies=ifly).data
x = dst.metrics.remove_nan(x0)
# %%
X = time_delay_embedding(x, window_size=w)
print(x.shape, X.shape)
print(f'nb nan vals = {np.sum(np.isnan(X))}')

X = time_delay_embedding(np.ascontiguousarray(x), window_size=w)
print(x.shape, X.shape)
print(f'nb nan vals = {np.sum(np.isnan(X))}')

# %%
print('dtype', x.dtype, 'flags', x.flags, 'shape', x.shape)
print('dtype', np.array(x).dtype, 'flags', np.array(x).flags, 'shape', np.array(x).shape)

print('strides', x.strides, 'itemsize', x.itemsize, 'nbytes', x.nbytes)
print('strides', np.array(x).strides, 'itemsize',np.array(x).itemsize, 'nbytes', np.array(x).nbytes)