"""
Various FV specific utility functions.

Acknowledgements:

    These functions are downloaded from PorePy,
    an open-source simulation tool for fractured and deformable
    porous media developed by the University of Bergen,
    see https://github.com/pmgbergen/porepy
    which is released under the terms of the GNU General Public License
    
"""

import scipy.sparse as sps
import numpy as np

def scalar_divergence(g):
    """
    Get divergence operator for a grid.

    The operator is easily accessible from the grid itself, so we keep it
    here for completeness.

    See also vector_divergence(g)

    Parameters
    ----------
    g grid

    Returns
    -------
    divergence operator
    """
    return g.cell_faces.T.tocsr()


def vector_divergence(g):
    """
    Get vector divergence operator for a grid g

    It is assumed that the first column corresponds to the x-equation of face
    0, second column is y-equation etc. (and so on in nd>2). The next column is
    then the x-equation for face 1. Correspondingly, the first row
    represents x-component in first cell etc.

    Parameters
    ----------
    g grid

    Returns
    -------
    vector_div (sparse csr matrix), dimensions: nd * (num_cells, num_faces)
    """
    # Scalar divergence
    scalar_div = g.cell_faces

    # Vector extension, convert to coo-format to avoid odd errors when one
    # grid dimension is 1 (this may return a bsr matrix)
    # The order of arguments to sps.kron is important.
    block_div = sps.kron(scalar_div, sps.eye(g.dim)).tocsc()

    return block_div.transpose().tocsr()

def gradient_of_scalar(g, f):

    """
    Calculates the reactive mineral surface area
    as the gradient of the porosity.

    For now assumes structured grid

    Inputs:
    g: the grid
    f: the cell-centre porosity field
        dimensions: (g.num_cells)
    
    Returns:
    b: np.array(g.num_cells).
        Norm of the gradient of f.
    """
 
    f = f.reshape((g.Nx, g.Ny), order = 'F')
    bx, by = np.gradient(f, g.dx, g.dx)
    bx = np.ravel(bx, order='f')
    by = np.ravel(by, order='f')
    
    b = np.sqrt(bx**2 + by**2)
          
    return b
