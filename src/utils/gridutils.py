"""
Various utility functions used in the computation of grid geometry

Acknowledgements:

    This implementation is downloaded from PorePy,
    an open-source simulation tool for fractured and deformable
    porous media developed by the University of Bergen,
    see https://github.com/pmgbergen/porepy
    which is released under the terms of the GNU General Public License
    
    The functions are a python translation of the corresponding matlab
    functions found in the Matlab Reservoir Simulation Toolbox (MRST) developed
    by SINTEF ICT, see www.sintef.no/projectweb/mrst/ . 
    
"""

import numpy as np


def rldecode(A, n):
    """ Decode compressed information. 
        
        The code is heavily inspired by MRST's function with the same name, 
        however, requirements on the shape of functions are probably somewhat
        different.
        
        >>> rldecode(np.array([1, 2, 3]), np.array([2, 3, 1]))
        [1, 1, 2, 2, 2, 3]
        
        >>> rldecode(np.array([1, 2]), np.array([1, 3]))
        [1, 2, 2, 2]
        
        Args:
            A (double, m x k), compressed matrix to be recovered. The 
            compression should be along dimension 1
            n (int): Number of occurences for each element
    """
    r = n > 0
    i = np.cumsum(np.hstack((np.zeros(1, dtype=np.int), n[r])), dtype=np.int)
    j = np.zeros(i[-1], dtype=np.int)
    j[i[1:-1:]] = 1
    B = A[np.cumsum(j)]
    return B

def mcolon(lo, hi):
    """ Expansion of np.arange(a, b) for arrays a and b.

    The code is equivalent to the following (less efficient) loop:
    arr = np.empty(0)
    for l, h in zip(lo, hi):
        arr = np.hstack((arr, np.arange(l, h, 1)))

    Parameters:
        lo (np.ndarray, int): Lower bounds of the arrays to be created.
        hi (np.ndarray, int): Upper bounds of the arrays to be created. The
            elements in hi will *not* be included in the resulting array.

        lo and hi should either have 1 or n elements. If their size are both
        larger than one, they should have the same length.

    Examples:
        >>> mcolon(np.array([0, 0, 0]), np.array([2, 4, 3]))
        array([0, 1, 0, 1, 2, 3, 0, 1, 2])

        >>> mcolon(np.array([0, 1]), np.array([2]))
        array([0, 1, 1])

        >>> mcolon(np.array([0, 1, 1, 1]), np.array([1, 3, 3, 3]))
        array([0, 1, 2, 1, 2, 1, 2])

    """
    if lo.size == 1:
        lo = lo * np.ones(hi.size, dtype="int64")
    if hi.size == 1:
        hi = hi * np.ones(lo.size, dtype="int64")
    if lo.size != hi.size:
        raise ValueError(
            "Low and high should have same number of elements, " "or a single item "
        )

    i = hi >= lo + 1
    if not any(i):
        return np.array([], dtype=np.int32)

    lo = lo[i].astype(np.int)
    hi = (hi[i] - 1).astype(np.int)
    d = hi - lo + 1
    n = np.sum(d)

    x = np.ones(n, dtype=np.int)
    x[0] = lo[0]
    x[np.cumsum(d[0:-1])] = lo[1:] - hi[0:-1]
    return np.cumsum(x)

def compute_tangent(pts):
    """ Compute a tangent vector of a set of points.

    The algorithm assume that the points lie on a plane.

    Parameters:
    pts: np.ndarray, 3xn, the points.

    Returns:
    tangent: np.array, 1x3, the tangent.

    """

    mean_pts = np.mean(pts, axis=1).reshape((-1, 1))
    # Set of possible tangent vector. We can pick any of these, as long as it
    # is nonzero
    tangent = pts - mean_pts
    # Find the point that is furthest away from the mean point
    max_ind = np.argmax(np.sum(tangent ** 2, axis=0))
    tangent = tangent[:, max_ind]
    assert not np.allclose(tangent, np.zeros(3))
    return tangent / np.linalg.norm(tangent)
