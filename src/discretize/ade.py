"""
Various functions used in the discretized Advection-Dispersion equation

Acknowledgements:

    The TPFA implementation of the diffusion term and the upwind
    implementation of the advection term, are in practice a translation
    of the corresponding functions in PorePy, an open-source simulation
    tool for fractured and deformable porous media developed by the
    University of Bergen,
    see https://github.com/pmgbergen/porepy
    which is released under the terms of the GNU General Public License
    
"""

import numpy as np
import scipy.sparse as sps
from utils import fvutils

def discretize_diffusion(g, bnd, mu, u_bound):

    """
    Discretize the diffusion term
    div dot (D grad c)
    in the ADE using a two-point flux approximation.

    Inputs:
    g: the grid
    bnd: boundary conditions
    mu: the diffusion coefficient
        dimensions: num_cells
        Units [m2 s]
    u_bound: boundary values of the concentration
        (assigned for Dirichlet-type boundary conditions).
        dimensions: num_faces
        Units [mol/m3]
    
    Returns:
    A: sps.csr_matrix (g.num_cells, g.num_cells)
        discretization of the diffusion term, cell center contribution
    rhs: np.array (g.num_cells).
        discretized boundary conditions
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
    t = 1 / np.bincount(fi, weights=1 / t_face)

    # Boundary conditions
    is_dir = bnd.is_dir
    is_neu = bnd.is_neu

    # Move Neumann faces to Neumann transmissibility
    bndr_ind = g.get_boundary_faces()
    t_b = np.zeros(g.num_faces)
    t_b[is_dir] = -t[is_dir]
    t_b[is_neu] = 1
    t_b = t_b[bndr_ind]
    t[is_neu] = 0
    # Create flux matrix
    flux = sps.coo_matrix((t[fi] * sgn, (fi, ci))).tocsr()

    # Create boundary flux matrix
    bndr_sgn = (g.cell_faces[bndr_ind, :]).data
    sort_id = np.argsort(g.cell_faces[bndr_ind, :].indices)
    bndr_sgn = bndr_sgn[sort_id]
    bound_flux = sps.coo_matrix(
        (t_b * bndr_sgn, (bndr_ind, bndr_ind)), (g.num_faces, g.num_faces)
    ).tocsr()

    # Return the divergence of the flux
    A = div * flux
    rhs = - div * bound_flux * u_bound

    return A, rhs

def discretize_advection(g, bnd, q, c_bound):

    """
    Discretize the advection term
    div dot (c u)
    in the ADE using a first-order upwind scheme.

    Inputs:
    g: the grid
    bnd: boundary conditions (use flow conditions)
    q: the flux field
        dimensions: num_faces
        Units [m s]
    c_bound: boundary values of the concentration
        (assigned for Dirichlet-type boundary conditions).
        dimensions: num_faces
        Units [mol/m3]
    
    Returns:
    A: sps.csr_matrix (g.num_cells, g.num_cells)
        discretization of the diffusion term, cell center contribution
    rhs: np.array (g.num_cells).
        discretized boundary conditions
    """
    
    #Weight fluxes with face areas
    flux = np.copy(q) * g.face_areas

    # Compute the face flux respect to the real direction of the normals
    indices = g.cell_faces.indices
    flow_faces = g.cell_faces.copy()
    flow_faces.data *= flux[indices]

    # Retrieve the faces boundary and their numeration in the flow_faces
    # We need to impose no-flow for the inflow faces without boundary
    # condition
    mask = np.unique(indices, return_index=True)[1]
    bc_neu = g.get_boundary_faces()

    # If boundary conditions are imposed remove the faces from this
    # procedure. The outflow boundary is considered of dirichlet type in
    # this example but it doesn't matter because of the upwind scheme
    is_dir = bnd.is_dir#np.logical_or(bnd.is_vel, bnd.is_pres)
    bc_dir = np.where(is_dir)[0]
    bc_neu = np.setdiff1d(bc_neu, bc_dir, assume_unique=True)
    bc_dir = mask[bc_dir]

    # Remove Dirichlet inflow
    inflow = flow_faces.copy()
    inflow.data[bc_dir] = inflow.data[bc_dir].clip(max=0)
    flow_faces.data[bc_dir] = flow_faces.data[bc_dir].clip(min=0)

    # Remove all Neumann
    bc_neu = mask[bc_neu]
    flow_faces.data[bc_neu] = 0

    # Determine the outflow faces
    if_faces = flow_faces.copy()
    if_faces.data = np.sign(if_faces.data)

    # Compute the inflow/outflow related to the cells of the problem
    flow_faces.data = flow_faces.data.clip(min=0)

    flow_cells = if_faces.transpose() * flow_faces
    flow_cells.tocsr()
    A = flow_cells.astype(np.float)

    # Impose the boundary conditions
    bc_val_dir = np.zeros(g.num_faces)
    if np.any(bnd.is_dir):
        bc_val_dir[is_dir] = c_bound[is_dir]

    # We assume that for Neumann boundary condition a positive 'bc_val'
    # represents an outflow for the domain. A negative 'bc_val' represents
    # an inflow for the domain.
    bc_val_neu = np.zeros(g.num_faces)

    """
    is_neu = np.logical_or(bnd.is_wall, bnd.is_pres)

    if np.any(is_neu):
        ind_neu = np.where(is_neu)[0]
        bc_val_neu[ind_neu] = c_bound[ind_neu]
    """

    rhs = (
        -inflow.transpose() * bc_val_dir
        - np.abs(g.cell_faces.transpose()) * bc_val_neu
        )

    return flow_cells, rhs
