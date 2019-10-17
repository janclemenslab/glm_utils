# GLM utilities

## Installation
```shell
conda install scikit-learn
pip install git+http://github.com/janclemenslab/glm_utils
```

## Usage
Check out the [wiki](https://github.com/janclemenslab/glm_utils/wiki).

## Example  code

```python
from tempfile import TemporaryDirectory

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline

from glm_utils.preprocessing import time_delay_embedding

# Generate dummy data.
x = np.random.random((1000, 1))
y = np.random.random((1000,))

# Manipulations regarding X *and* y
X, y = time_delay_embedding(x, y, window_size=100)

# Transformations that only change X are included in the pipeline.
steps = [('quad_exp', PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)),
         ('ridge', BayesianRidge())]

with TemporaryDirectory() as tempdir:
    model = Pipeline(steps, memory=tempdir)
    model.fit(X, y)
    y_pred, y_pred_std = model.predict(X, return_std=True)
    print(f'r2={model.score(X, y):1.2}')
```

## Notes
Processing steps that require manipulations of `X` *and* `y` - such as time delay embedding, balancing, test-train split - are currently not supported by sklearn pipelines (see [here](https://github.com/scikit-learn/scikit-learn/issues/4143)). These steps should be performed before the pipeline using functions that implement the following signature: 
`X, y = func(X, y=None, **kwargs)`

Processing steps that only affect `X` - feature normalization or scaling, basis function projections, PCA, polynomial expansions - should be included in the pipeline. Custom functions should follow this signature: `X = func(X, **kwargs)` and can be integrated into a pipeline with the [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer). Alternatively, a custom class implementing the [Transformer interface](https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) can be used. Custom classes have the advantage of providing access to parameters of the transform - e.g. the basis vectors used in a basis function projection.

## Useful extensions
- [Group lasso](https://group-lasso.readthedocs.io/en/latest/index.html)
- more advanced balancing with [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/)
- Generalized Additive models as a more principled way of using basis functions: [pyGAM](https://pygam.readthedocs.io/en/latest/index.html)

## TODO:
need-to_haves:
- [x] make basis and project onto basis functions
- [x] reconstruct time-domain filter from basis functions and weights
- [x] balancing
- [ ] add example code for standard pipelines

nice-to-haves:
- [ ] deal with multi-feature fits (need to adjust basis functions and time delay embedding accordinly)
- [ ] deal with subsets (avoid computations on data we don't use anyways)
