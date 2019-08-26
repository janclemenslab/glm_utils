# glm_utils


## Example of a pipeline

```python
from tempfile import TemporaryDirectory

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline

from glm_utils.preprocessing import time_delay_embedding

# Generate dummy data.
x = np.random.random((1000, 1))
y = np.random.random((1000,))

# Do manipulations regarding X *and* y here (e.g. balancing, test-train split, selection of data).
X, y = time_delay_embedding(x, y, window_size=100)

# Transformations that only change X (normalization, feature transformation)
# should be made part of the pipeline.
steps = [('quad_exp', PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)),
         ('ridge', BayesianRidge())]

with TemporaryDirectory() as tempdir:
    model = Pipeline(steps, memory=tempdir)
    model.fit(X, y)
    y_pred, y_pred_std = model.predict(X, return_std=True)
    print(f'r2={model.score(X, y):1.2}')
```

## TODO:
need-to_haves:
- [ ] make basis and project onto basis functions
- [ ] reconstruct time-domain filter from basis functions and weights
- [ ] balancing
- [ ] add example code for standard pipelines

nice-to-haves:
- [ ] deal with multi-feature fits (need to adjust basis functions and time delay embedding accordinly)
- [ ] deal with subsets (avoid computations on data we don't use anyways)
