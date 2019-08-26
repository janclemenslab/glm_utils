# glm_utils


## Example of a pipeline

```python
from tempfile import TemporaryDirectory

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline

from glm_utils.preprocessing import time_delay_embedding

x = np.random.random(1000, 1)
y = np.random.random(1000, )

w = 100
steps = [('delay_embed', FunctionTransformer(time_delay_embedding, kw_args={'window_size': w, 'overlap_size': w-1, 'preserve_size': True})),
         ('quad_exp', PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)),
         ('ridge', BayesianRidge())]

with TemporaryDirectory() as tempdir:
    clf = Pipeline(steps, memory=tempdir)
    clf.fit(x, y)
    y_pred, y_pred_std = clf.predict(x, return_std=True)
    print(f'r2={clf.score(x, y):1.2}')
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
