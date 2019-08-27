"""Tools for postprocessing models."""
import numpy as np


def unpack_quadratic_kernel(coefficients, window_size: int, quad_as_matrix: bool = True, reverse_time: bool = True):
    """Recover bias, linear, and quadratic terms from the coefficients of a quadratic regression model.

    Args:
        coefficients (numpy array): Coefficients of the regression model (`model.coef_` or for pipelines `model[-1].coef_`).
                                    Terms are organized as given by sklearn.preprocessing.PolynomialFeatures.
        window_size (int): Window size used for time-delay embedding the data.
        quad_as_matrix (bool, optional): Return quadratic terms as a symmetrical matrix (True) 
                                         or just the unique, non-redundant terms used for fitting (False).
                                         Defaults to True.
        reverse_time (bool, optional): Reverse time. Defaults to True.

    Returns:
        bias [scalar], linear [w,], quadratic terms ([w, w] or [w*w/2 - w/2,])
    """
    bias, linear, quadratic = _unpack_quadratic_features(coefficients[np.newaxis, ...],
                                                         window_size, quad_as_matrix)

    if reverse_time:
        linear = linear[:, ::-1]
        if quad_as_matrix:
            quadratic = quadratic[:, ::-1, ::-1]
        else:
            quadratic = quadratic[:, ::-1]

    return bias, linear[0], quadratic[0]


def _unpack_quadratic_features(X_quad, window_size: int, quad_as_matrix: bool = False):
    """Unpack quadratic features into bias, linear, and quadratic terms.

    Args:
        X_quad (numpy array): Quadratic features or filter coefficients obtained after quadratic expansion [time, features]
                              Terms are organized as given by sklearn.preprocessing.PolynomialFeatures.
        window_size (int): Window size [description]
        quad_as_matrix (bool, optional): Transform quadratic terms ino a matrix. Defaults to False.

    Returns:
        bias [time, ], linear [time, w], quadratic terms ([time, w, w] or [time, w*w/2 - w/2,])
    """

    # split feature axis into bias, linear and quadratic terms
    bias = X_quad[:, 0]
    linear = X_quad[:, 1:window_size+1]
    quadratic = X_quad[:, window_size+1:]

    # Only the unique quadratic features are contained in `quadratic`.
    # Expand to symmetrical square matrix:
    if quad_as_matrix:
        x_idx, y_idx = np.triu_indices(window_size)
        quadratic_matrix = np.zeros((X_quad.shape[0], window_size, window_size))
        for cnt, (q, xi, yi) in enumerate(zip(quadratic.T, x_idx, y_idx)):
            quadratic_matrix[:, xi, yi] = q
            quadratic_matrix[:, yi, xi] = q
        quadratic = quadratic_matrix

    return bias, linear, quadratic
