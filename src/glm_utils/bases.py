import numpy as np
import scipy.interpolate as si


def laplacian_pyramid(width, levels, step, FWHM, normalize=True):
    """ Get a 1d Laplacian pyramid basis matrix of given number of levels for
        vectors of given length.

        Args:
            width (int): Time span of the basis functions.
            levels (int): Number of levels.
            step (float): Spacing of levels (1= regular Laplacian pyramid, 0.5 = Laplacian pyramid with half-levels).
                          At each full-step, the width of the Gaussian is doubled
            FWHM (float): Full width at half-max for the Gaussians at level 1 (the finest level).
            normalize (boolean, optional): Normalize each basis to unit L2 norm. Defaults to True.

        Returns:
            [time, bases] - np matrix with basis functions.

        Adapted from:
            Mineault, P. J., Barthelmé, S. & Pack, C. C.
            Improved classification images with sparse priors in a smooth basis.
            Journal of Vision 9, 1–24 (2009).
    """

    B = list()
    rg = np.arange(0, width)
    for ii in np.arange(0, levels, step, dtype=float):
        cens = 2**(ii-2) + np.arange(int(width/(2**(ii-1))-1))*(2**(ii-1))
        if len(cens):  # check if there are any basis functions for that level
            cens = np.floor((width-(np.max(cens)-np.min(cens)+1))/2+cens)+1
            gwidth = 2**(ii-1)/2.35*FWHM
            for jj in range(1, len(cens)):
                v = np.exp(-(rg-cens[jj])**2 / 2 / gwidth**2)
                if normalize:
                    v = v / np.linalg.norm(v)
                B.append(v)
    return np.stack(B).T


def nlin(x): return np.log(x + 1e-20)


def invnl(x): return np.exp(x) - 1e-20


def ff(x, c, db):
    """ Gives the raised cosine of data points `x` with classified centers
    `c` with spacing between cosine peaks `db`.
    """
    kbasis = (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x-c)*np.pi/db/2)))+1)/2
    return kbasis


def normalizecols(A):
    """ Normalize the columns of a 2D array."""
    B = A/np.tile(np.sqrt(sum(A**2, 0)), (np.size(A, 0), 1))
    return B


def raised_cosine(neye, ncos, kpeaks, b, nkt=None):
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
    kbasis : ndarray [time, bases]
        A basis of raised cosines as columns. The `neye` value defines
        the number of first columns which has identity matrix at last rows
        for dense sampling of data.

    """

    kpeaks = np.array(kpeaks)
    kdt = 1  # step for the kernel

    yrnge = nlin(kpeaks + b)  # nonlinear transform, b is nonlinearity of scaling

    db = (yrnge[1] - yrnge[0]) / (ncos - 1)  # spacing between cosine peaks
    ctrs = np.linspace(yrnge[0], yrnge[1], ncos)  # nlin(kpeaks)<-weird # centers of cosines

    # mxt is for the kernel, without the nonlinear transform
    mxt = invnl(yrnge[1] + 2 * db) - b  # !!!!why is there 2*db? max time bin
    kt0 = np.arange(0, mxt, kdt)  # kernel time points/ no nonlinear transform yet
    nt = len(kt0)  # number of kernel time points

    # Now we transform kernel time points through nonlinearity and tile them
    e1 = np.tile(nlin(kt0 + b), (ncos, 1))
    # Tiling the center points for matrix multiplication
    e2 = np.tile(ctrs, (nt, 1)).T

    # Creating the raised cosines
    kbasis0 = ff(e1, e2, db)

    # Concatenate identity vectors
    nkt0 = np.size(kt0, 0)  # !!!! same as nt??? Redundant or not
    a1 = np.concatenate((np.eye(neye), np.zeros((nkt0, neye))), axis=0)
    a2 = np.concatenate((np.zeros((neye, ncos)), kbasis0.T), axis=0)
    kbasis = np.concatenate((a1, a2), axis=1)
    kbasis = np.flipud(kbasis)
    nkt0 = np.size(kbasis, 1)

    # Modifying number of output cosines if nkt is given
    if nkt == None:
        pass
    elif nkt0 < nkt:  # if desired time samples greater, add more zero basis
        kbasis = np.concatenate((np.zeros(kbasis, (nkt - nkt0, ncos + neye))), axis=0)
    elif nkt0 > nkt:  # if desired time samples less, get the last nkt columns of cosines
        kbasis = kbasis[:, :nkt]

    kbasis = normalizecols(kbasis)
    return kbasis


def bsplines(width, positions, degree: int = 3, periodic: bool = False):
    """Get basis matrix for b-splines.

    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep

    Usage: B = get_bspline_basis(width=100, positions=np.arange(5, 100, 10))

    Args:
        width (array-like): Time span over which the splines are evaluated.
        positions (list-like): Positions of individual basis functions.
        degree (int): Polynomial degree of the splines. Defaults to 3.
        periodic (bool): . Defaults to False.

    Returns:
        [time, bases] matrix with basis functions

    """
    t = np.arange(width)
    npositions = len(positions)
    y_dummy = np.zeros(npositions)

    positions, coeffs, degree = si.splrep(positions, y_dummy, k=degree,
                                          per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(npositions):
        coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
        bsplines.append((positions, coeffs, degree))

    B = np.array([si.splev(t, spline) for spline in bsplines])
    B = B[:, ::-1].T  # invert so bases "begin" at the right and transpose to [time x bases]
    return B


def identity(width):
    return np.identity(width)
