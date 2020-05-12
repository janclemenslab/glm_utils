# GLM utilities

Tools for fitting Generalized Linear Models (GLMs) with time dependence (a.k.a. filters).

Features:
- Time delay embedding ([basics demo](demo/basics.ipynb)).
- Basis functions ([bases demo](demo/basis_functions.ipynb))
- Follows [scikit-learn's](https://scikit-learn.org/) API conventions. Can be used with [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline) (see [demo](demo/pipeline.ipynb)).

For a more complete example see [multiple_inputs](demo/multiple_inputs.ipynb).


## Installation
Package is on [pypi](https://pypi.org/project/glm-utils/):
`pip install glm-utils`


## Useful extensions
- [Group lasso](https://group-lasso.readthedocs.io/en/latest/index.html). See [demo](demo/group_lasso.ipynb).
- Balancing with [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/).
- Generalized Additive Models: [pyGAM](https://pygam.readthedocs.io/en/latest/index.html). See [demo](demo/gam.ipynb).
