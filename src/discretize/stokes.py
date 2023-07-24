"""
Various functions used in the discretized Stokes equations

Acknowledgements:

    The TPFA implementation of the laplacian terms, i.e.
    discretize_momentum and discretize_pressure_correction,
    is in practice a translation of the corresponding functions in
    PorePy, an open-source simulation tool for fractured and deformable
    porous media developed by the University of Bergen,
    see https://github.com/pmgbergen/porepy
    which is released under the terms of the GNU General Public License
    
"""

import numpy as np
import scipy.sparse as sps
from utils import fvutils

thresh = 1e-16

def discretize_momentum(g, bnd, mu, u_bound, direction, keyword):

    """
    Discretize the viscous laplacian term
    div dot (mu grad u)
    in the momentum equation using a two-point flux approximation.

    Inputs:
    g: the grid
    bnd: boundary conditions
    mu: the viscous coefficient
        dimensions: num_cells
        Units [Pa s]
    u_bound: boundary values of the velocity
        (assigned for Dirichlet-type boundary conditions).
        dimensions: num_faces
        Units [m / s]
    direction: direction of the discretized velocity component
    keyword: can be either 'microcontinuum' or 'NoSlip'
    
    Returns:
    A: sps.csr_matrix (g.num_cells, g.num_cells)
        discretization of the viscous term, cell center contribution
        Units [Pa s]
    rhs: np.array (g.num_cells).
        discretized boundary conditions
        Units [Pa m]
    """
    
    # Retrieve cell-faces relations
    div = fvutils.scalar_divergence(g)
    fi, ci, sgn = sps.find(g.cell_faces)

    # Normal vectors for each face (here and there side)
    # Note normal vectors are weighted with face areas
    n = g.face_normals[:, fi]
    # Switch signs where relevant
    n *= sgn
    # Diffusion for each cell
    perm = mu[ci]
    # Take product between diffusion and normal vectors
    nk = perm * n

    # Distance from face center to cell center
    fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]

    # Take dot product between nk and distance
    nk *= fc_cc
    t_face = nk.sum(axis=0)

    # Calculate transmissibilities
    dist_face_cell = np.power(fc_cc, 2).sum(axis=0)
    t_face = np.divide(t_face, dist_face_cell)

    # Return harmonic mean
    t = 1 / np.bincount(fi, weights = 1 / t_face)

    # Boundary conditions
    # Types 'vel' and 'wall' are of Dirichlet type for velocity
    # Type 'pres' is of Neumann type for velocity (zero-gradient)
    is_dir = np.logical_or(bnd.is_vel, bnd.is_wall)
    is_neu = bnd.is_pres

    # Type 'symm' is of Dirichlet type in the main direction
    # and Neumann type (zero gradient) in the other two

    if any(bnd.is_sym):
        if direction == 'x':
            is_dir = np.logical_or(is_dir, bnd.is_sym_x)
            is_other = np.logical_or(bnd.is_sym_y, bnd.is_sym_z)
            is_neu = np.logical_or(bnd.is_pres, is_other)
        elif direction == 'y':
            is_dir = np.logical_or(is_dir, bnd.is_sym_y)
            is_other = np.logical_or(bnd.is_sym_x, bnd.is_sym_z)
            is_neu = np.logical_or(bnd.is_pres, is_other)
        elif direction == 'z':
            is_dir = np.logical_or(is_dir, bnd.is_sym_z)
            is_other = np.logical_or(bnd.is_sym_x, bnd.is_sym_y)
            is_neu = np.logical_or(bnd.is_pres, is_other)
        else:
            raise ValueError("symmetry must be x, y, or z")

    # Move Neumann faces to Neumann transmissibility
    bndr_ind = g.get_boundary_faces()
    t_b = np.zeros(g.num_faces)
    t_b[is_dir] = -t[is_dir]
    t_b[is_neu] = 1
    t_b = t_b[bndr_ind]
    t[is_neu] = 0

    # Return dirichlet condition for internal walls if no-slip model
    if keyword == 'NoSlip':
        wall_ind = np.where(bnd.is_wall)[0]
        int_wall = wall_ind[~np.isin(wall_ind, bndr_ind)]       
        t[int_wall] *= 2

    # Create flux matrix
    flux = sps.coo_matrix((t[fi] * sgn, (fi, ci))).tocsr()

    # Create boundary flux matrix
    # for the no-slip model, internal walls would be no-slip anyway
    # so we don't care
    bndr_sgn = (g.cell_faces[bndr_ind, :]).data
    sort_id = np.argsort(g.cell_faces[bndr_ind, :].indices)
    bndr_sgn = bndr_sgn[sort_id]
    bound_flux = sps.coo_matrix(
        (t_b * bndr_sgn, (bndr_ind, bndr_ind)), (g.num_faces, g.num_faces)
    ).tocsr()

    # Return the divergence of the flux
    A = div * flux
    rhs = - div * bound_flux * u_bound

    # For the no-slip model, we eliminate internal wall cells
    # The corresponding rows and columns are set to zero
    if keyword == 'NoSlip':
        def csr_row_set_nz_to_val(csr, row, value=0):
            """
            Set all nonzero elements (elements currently
            in the sparsity pattern to the given value.
            Useful to set to 0 mostly.
            """       
            if not isinstance(csr, sps.csr_matrix):
                raise ValueError('Matrix given must be of CSR format.')
            csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

        for j in bnd.list_wall:
            csr_row_set_nz_to_val(A, j)

        # restore diagonal elements
        values = A.diagonal()
        values[bnd.list_wall] = 1
        A.setdiag(values)
        rhs[bnd.list_wall] = 0

        A.eliminate_zeros()

    return A, rhs

def discretize_pressure_correction(g, bnd, ac, keyword):

    """
    Discretize the laplacian
    div dot (1/ac grad p)
    in the pressure correction equation using TPFA

    Inputs:
    g: the grid
    bnd: boundary conditions
    ac: central coefficients of the discretized momentum equation
        dimensions: (g.dim, num_cells)
        Units [Pa s]
    keyword: can be either 'microcontinuum' or 'NoSlip'
    
    Returns:
    A: sps.csr_matrix (g.num_cells, g.num_cells)
        discretization of the laplacian, cell center contribution
        Units [1 / (Pa s)]
    rhs: np.array (g.num_cells).
        discretized boundary conditions
        Units [1 / s]
    """

    # Retrieve cell-faces relations
    div = fvutils.scalar_divergence(g)
    fi, ci, sgn = sps.find(g.cell_faces)
   
    # Normal vectors for each face (here and there side)
    n = g.face_normals[0:g.dim, fi] / g.face_areas[fi]
    # Switch signs where relevant
    n *= sgn
    # Diffusion for each cell
    perm = 1 / ac[:, ci]
    # Take product between diffusion and normal vectors
    nk = perm * n
    
    # Distances from face centers to cell centers
    fc_cc = g.face_centers[0:g.dim, fi] - g.cell_centers[0:g.dim, ci]
    # Take dot product between nk and distance
    nk *= fc_cc
    t_face = nk.sum(axis=0)
    # Take norm of the distance face - cell
    dist_face_cell = np.sqrt(np.power(fc_cc, 2).sum(axis=0))
    
    # Return arithmetic mean
    num = np.bincount(fi, weights = t_face)
    den = np.bincount(fi, weights = dist_face_cell)
    t = num / den

    # Not sure why we need to impose this
    # and not come authomatic as for the harmonic mean
    t[bnd.is_pres] *= 2

    # Boundary conditions
    # Type 'pres' is of Dirichlet type for pressure
    # Others are of Neumann type: zero-gradient
    is_no_slip = np.logical_or(bnd.is_wall, bnd.is_vel)
    is_neu = np.logical_or(is_no_slip, bnd.is_sym)
    t[is_neu] = 0
    
    # Create flux matrix
    flux = sps.coo_matrix((t[fi] * sgn, (fi, ci))).tocsr()

    # Return the divergence of the flux
    A = - div * flux

    # For the no-slip model, we eliminate internal wall cells
    # The corresponding rows and columns are set to zero
    if keyword == 'NoSlip':
        def csr_row_set_nz_to_val(csr, row, value=0):
            """
            Set all nonzero elements (elements currently
            in the sparsity pattern to the given value.
            Useful to set to 0 mostly.
            """
            if not isinstance(csr, sps.csr_matrix):
                raise ValueError('Matrix given must be of CSR format.')
            csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

        for j in bnd.list_wall:
            csr_row_set_nz_to_val(A, j)

        # restore diagonal elements
        values = A.diagonal()
        values[bnd.list_wall] = 1
        A.setdiag(values)

        A.eliminate_zeros()

    return A

def get_face_pressure(g, bnd, p, p_bound):

    """
    Return the pressure field on the grid faces
    using linear interpolation.

    Inputs:
    g: the grid
    bnd: boundary conditions
    p: the cell-centre pressure field
        dimensions: num_cells
        units: Pa
    p_bound: boundary values of the pressure (assigned for pres-type
        boundary conditions).
        Dimensions: num_faces
        units: Pa
    
    Returns:
    pf: np.array (g.num_faces).
        Pressure on the faces.
        units: Pa
    """
    
    pf = np.zeros(g.num_faces)

    # Retrieve cell-faces relations
    fi, ci, sgn = sps.find(g.cell_faces)

    # For internal faces, return the arithmetic mean
    # For now this assumes a structured cartesian grid
    # For general quadrilateral grids, this should be
    # generalized in the future taking into account
    # the distances between face and cell centers
    pf = 0.5 * np.bincount(fi, weights=p[ci])

    # Boundary faces
    is_dir = np.where(bnd.is_pres)[0]
    is_vel = np.logical_or(bnd.is_wall, bnd.is_vel)
    is_neu = np.logical_or(bnd.is_sym, is_vel)

    # pres is dirichlet, assign boundary value
    pf[is_dir] = p_bound[is_dir]

    # all the other boundaries do not have a condition for pressure
    # easy method is to use zero-neumann condition
    pf[is_neu] *= 2

    return pf

def discretize_pressure_gradient(g, p):

    """
    Return the discretized pressure gradient
    on the cell centers, weighted with the face areas.

    Inputs:
    g: the grid
    p: the pressure field on the faces
        dimensions: num_faces
        Units: Pa
    
    Returns:
    b: vector field (g.dim * g.num_cells).
        Cell-centers Pressure gradient.
        Units: Pa m in 2D (Pa m2 in 3D)
    """
    
    # Retrieve cell-faces relations
    grad = fvutils.scalar_divergence(g)

    # Initialize variables
    pn = np.zeros((g.dim, g.num_faces))
    b = np.zeros((g.dim, g.num_cells))

    # Take product between face pressure and the face normal
    # Note the face normal is weighted with the face area
    for i in np.arange(g.dim):
        pn[i] = p*g.face_normals[i]

    # Take the cell-centre gradient
    for i in np.arange(g.dim):
        b[i]=grad*pn[i]
         
    return b

def vector_linear_interpolation(g, fi, ci, u):

    """
    Calculates linear interpolation of cell-centers vector fields on the faces.
    For now works only for structured grids

    Inputs:
    g: the grid
    fi: faces of cells.
        dimensions: (2 * g.dim) * g.num_cells
    ci: cell indices relative to fi
        dimensions: (2 * g.dim) * g.num_cells
    u: the vector field on the cell-centers
        dimensions: (g.dim, g.num_cells)
    
    Returns:
    u_faces np.ndarray (g.dim * g.num_faces).
        Linear interpolation of u values on the faces

    """

    assert fi.size == ci.size

    # Initialize vector field
    uf = np.zeros((g.dim, g.num_faces))

    # Do linear interpolation
    for i in np.arange(g.dim):
        uf[i] = 0.5 * np.bincount(fi, weights=u[i,ci])
   
    return uf

def get_face_fluxes(g, bnd, u, ac, p, u_bound, p_faces):

    """
    Calculate the normal velocity component on the faces for use in the
    continuity equation: div u = 0
    using the momentum interpolation method by Rhie and Chow
    for non-staggered grids.

    Reference

    Rhie, C. M., & Chow, W. L. (1983).
    Numerical study of the turbulent flow past an airfoil
    with trailing edge separation. AIAA journal, 21(11), 1525-1532.

    Inputs:
    g: the grid
    bnd: boundary conditions
    u: the velocity field (g.dim * g.num_cells)
        Units: m/s
    ac: central coefficients of the discretized momentum equation
        dimension (g.dim, g.num_cells)
        Units: Pa s    
    p: the cell centres pressure field (g.num_cells)
        Units: Pa
    u_bound: boundary values of the velocity (assigned for Dirichlet-type
        boundary conditions). dimensions: num_faces
        Units: m/s
    p_faces: the pressure field on the faces (g.num_faces)
        Units: Pa
    
    Returns:
    u_n: np.array(g.num_faces)
        Cell-face normal component of the velocity
        Units: m/s
    """
    
    # Retrieve cell-faces relations
    div = fvutils.scalar_divergence(g)
    fi, ci, sgn = sps.find(g.cell_faces)
    
    # Initialize cell-face velocity vector (and other internal variables)
    u_faces = np.zeros((g.dim, g.num_faces))

    cell_gradp = np.zeros((g.dim, g.num_cells))
    cell_correction = np.zeros((g.dim, g.num_faces))

    # Internal faces have three contributions
    # For now, a structured cartesian grid is assumed.
    
    # 1 - Linear interpolation of cells velocity
    u_faces = vector_linear_interpolation(g, fi, ci, u)

    # 2 - Correction term due to pressure gradient across the face
    # Take linear interpolation of central coefficients
    acf = vector_linear_interpolation(g, fi, ci, 1/ac)
    # Return pressure correction across the face
    face_correction = acf * np.bincount(fi, weights=p[ci]*sgn) * g.face_normals[0:g.dim,:]

    # 3 - Correction term from Linear interpolation of cells pressure gradient
    # Take Half pressure gradient across the cell
    for i in np.arange(g.dim):
        cell_gradp[i] = 0.5 * div * (p_faces * g.face_normals[i]) / ac[i]

    # Return pressure correction across the cell
    for i in np.arange(g.dim):
        cell_correction[i] = np.bincount(fi, weights=cell_gradp[i,ci])

    # Assemble the three contributions
    u_faces = u_faces + face_correction + cell_correction
    # Take normal component
    u_n = u_faces * g.face_normals[0:g.dim,:] / g.face_areas
    u_n = u_n.sum(axis=0)

    # Boundary faces that need not a correction
    # vel is dirichlet
    unbnd = u_bound[:, bnd.is_vel] * g.face_normals[0:g.dim, bnd.is_vel] / g.face_areas[bnd.is_vel]
    u_n[bnd.is_vel] = unbnd.sum(axis=0)
    # wall and sym are also dirichlet (no-slip condition)
    is_no_slip = np.logical_or(bnd.is_wall, bnd.is_sym)
    u_n[is_no_slip] = 0

    # For pres faces, things can get a bit more complicated
    # A zero-Neumann condition is enforced in the momentum equations
    # There are two ways of return the boundary velocity
    # The easiest is to return the velocity in the adjacent cell   
    ind_pres = np.ravel(np.argwhere(bnd.is_pres))
    # find the adjacent cells and sort them
    ind_cells=g.cell_face_as_dense()[:,ind_pres]
    ind_cells=np.sort(ind_cells[ind_cells >= 0])

    # assign the adjacent cell velocity and take normal component
    for i in np.arange(g.dim):
        u_faces[i, bnd.is_pres] =u[i, ind_cells]

    unbnd = u_faces[:, bnd.is_pres] * g.face_normals[0:g.dim, bnd.is_pres] / g.face_areas[bnd.is_pres]

    # Return boundary normal velocity
    u_n[bnd.is_pres] = unbnd.sum(axis=0)

    return u_n

def discretize_continuity(g, q):

    """
    Return the discretized continuity imbalance (div u)
    on the cell centers.

    Inputs:
    g: the grid
    q: the normal component of velocity on the faces
        dimensions: num_faces
    
    Returns:
    b: scalar field (g.num_cells).
        discretized divergence of velocity.
        Units: m/s
    f: scalar field (g.num_cells).
        discretized divergence of velocity integrated over the control volume.
        Units: m2/s in 2D (m3/s in 3D)
    """
    
    # Retrieve cell-faces relations
    div = fvutils.scalar_divergence(g)
    # Calculate the flow rate
    flux = q * g.face_areas

    # Take the divergence
    b = div * q
    f = div * flux

    # this threshold may not be needed
    threshold_indices = abs(b) < thresh
    b[threshold_indices] = 0
    threshold_indices = abs(f) < thresh
    f[threshold_indices] = 0
    
    return b, f 
    
