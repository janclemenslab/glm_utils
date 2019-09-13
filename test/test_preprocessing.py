import glm_utils.preprocessing as pre
import numpy as np

def test_transform():

    #Given
    X = np.array([[1,2], [3,3], [1,2]])
    basis = np.array([[1,0],[0,-1]])

    #When
    o = pre.BasisProjection(basis)
    Xt = o.transform(X)
    Xb = o.inverse_transform(Xt)

    #Then
    np.testing.assert_array_almost_equal(X,Xb)
