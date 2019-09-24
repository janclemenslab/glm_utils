# %%
from glm_utils.preprocessing import time_delay_embedding
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# %% test default
T = 100
x0 = 40+np.arange(0, T, 1)
y0 = -np.arange(0, T, 1)
y = y0
x = x0

w = 10

X = time_delay_embedding(x0, window_size=w)
print(x.shape, y.shape, X.shape)
assert X.shape[0] == T-w and X.shape[1] == w
plt.imshow(X)

X = time_delay_embedding(x0, window_size=w, exclude_t0=False)
assert X.shape[0] == T-w+1 and X.shape[1] == w
print(x.shape, y.shape, X.shape)


# %%  test exclude t0

# generate identical x and y values - corr. will be perfect when including t0, near-zero when excluding t0
x0 = np.random.random((2000, 1))
y0 = x0
from sklearn.linear_model import LinearRegression

w = 100
X, y = time_delay_embedding(x0, y0, window_size=w, exclude_t0=False)

clf = LinearRegression()
clf.fit(X, y)
print(f'without exclude: r2={clf.score(X, y):1.2}')
assert clf.score(X, y)==1.0

X, y = time_delay_embedding(x0, y0, window_size=w, exclude_t0=True)
clf = LinearRegression()
clf.fit(X, y)
print(f'with exclude: r2={clf.score(X, y):1.2}')
assert clf.score(X, y)<0.1


# %%  overlaps
T = 100
x0 = np.arange(0, T, 1)
y0 = x0

w = 10

X = time_delay_embedding(x0, window_size=w, overlap_size=w-1, exclude_t0=False)
plt.subplot(121)
plt.imshow(X)

X,y = time_delay_embedding(x0, y0, window_size=w, overlap_size=0, exclude_t0=False)
plt.subplot(233)
plt.imshow(X)
plt.subplot(236)
plt.plot(y, '.')
print('exclude_t0=True', y, X[:,-1])

X,y = time_delay_embedding(x0, y0, window_size=w, overlap_size=0, exclude_t0=True)
plt.subplot(233)
plt.imshow(X)
plt.subplot(236)
plt.plot(y, '.')
print('exclude_t0=False', y, X[:,-1])

# %%  test 2D x
# generate identical x and y values - corr. will be perfect when including t0, near-zero when excluding t0
x0 = np.random.random((200, 10))
y0 = x0
w = 10
X, y = time_delay_embedding(x0, y0, window_size=w, exclude_t0=False)
print(X.shape, y.shape)
X, y = time_delay_embedding(x0, y0, window_size=w, exclude_t0=False, flatten_inside_window=False)
print(X.shape, y.shape)

x0 = np.arange(300).reshape((-1, 3))
x0[:,0] = 10
x0[:, 1] = 20
x0[:, 2] = 30

X = time_delay_embedding(x0, window_size=w, flatten_inside_window=False)

print(x0.shape, X.shape)
plt.subplot(121)
plt.plot(x0[1:3,...].T)
plt.subplot(122)
plt.imshow(X[2])
# plt.colorbar()

# %% Exceptions
x0 = np.random.random((200, 10))
try:
    X, y = time_delay_embedding(x0, y0, exclude_t0=False)
except ValueError as e:
    print('raised excpected ValueError:')
    print(e)
else:    
    raise Exception('ERROR: did not raise excpected ValueError:')

x0 = np.random.random((200, 10, 10, 10))
try:
    X, y = time_delay_embedding(x0, y0, window_size=w, exclude_t0=False)
except ValueError as e:
    print('raised excpected ValueError:')
    print(e)
else:    
    raise Exception('ERROR: did not raise excpected ValueError:')

# %%
x0 = np.random.random((201054,)).astype(np.float64)
y0 = x0

X = time_delay_embedding(x0, window_size=10)
np.sum(np.isnan(X))

