"""
Solves coupled flow and reactive transport for Molins' benchmark problem.

This model reproduces results presented Section 3 of

Starnoni and Sanchez-Vila (2023)
Pore-scale modelling of microbially enhanced carbon mineralization
(Submitted)
"""

import numpy as np
import scipy.sparse as sps
from utils import fvutils
from discretize import stokes, ade
import scipy.sparse.linalg
from models import flowSolver
from importExport import readf, printf
import math

thresh = 1e-12

def get_discharge(g, u):

    """
    Calculates the flow rate on all planes normal to the flow direction.

    Inputs:
    g: the grid
    u: the cell-centre velocity field
        dimensions: (g.dim, g.num_cells)
    
    Returns:
    Q: np.array(g.Nz).
        Flow rate along the flow direction.
        One value for each element along the flow direction.
    """
    
    Q = np.zeros(g.Nx)
    for i in np.arange(g.Nx):
        xc = g.cell_centers[0, i]
        cells = np.ravel(np.argwhere(
            ((g.cell_centers[0] > xc-1e-8) & (g.cell_centers[0] < xc+1e-8))
            ))
        Q[i] = np.sum(u[cells]) * g.dx
    return Q

def get_outflow_concentration(g, c, faces, q_faces):

    """
    Calculates the outflow concentration.

    Inputs:
    g: the grid
    c: the cell-centre concentration field
        dimensions: (g.num_cells)
    faces: the outflow faces
    q_faces: the flux on the outflow faces
    
    Returns:
    c_out: outflow concentration.
    q_out: outflow discharge
    """

    cells = g.cell_face_as_dense()[0,faces]
    num = np.sum(c[cells] * q_faces * g.face_areas[faces])
    den = np.sum(q_faces * g.face_areas[faces])

    c_out = num/den

    return c_out, den
 
def solveRT(g, d, bndf, bndt, s):

    """
    Solves coupled Stokes flow and reactive transport for a single
    dissolving mineral grain using first-order kinetics

    Inputs:
    g: the grid
    d: data dictionary for fluid properties and operating conditions
    bndf: boundary conditions for flow
    bndt: boundary conditions for transport
    s: data dictionary for the flow and linear solvers
    
    Returns:
    p: solution of pressure
    u: solution of velocity
    c: solution of concentration
    phi: final porosity field
    l_time: list of time steps
    l_rate: list of average reaction rate
    l_surf: list of reactive surface area
    l_vol: list of mineral volume
    """

    # Retrieve Input data
    rho = d["fluid density"]
    mu = d["fluid viscosity"]
    initial_permeability = d["initial permeability"]
    Vm = d["molar volume"]
    xi = d["stoichiometric coefficient"]
    diff = d["diffusion coefficient"]
    kappa_gamma = d["rate constant"]
    c_in = d["inflow concentration"]

    flowfile = d["flow filename"]
    fluxfile = d["flux filename"]
    evolutionFile = d["evolution filename"]
    updateFlowCriterion = d["update flow criterion"]

    dt = s["time step"]
    t_end = d["end time"]
    t_print = d["print geometry"]
    tol_steady_state = s["tolerance steady-state"]
    tol_c = s["tolerance bicg transport"]

    # Assign boundary conditions
    u_bound = d["boundary velocity"]
    p_bound = d["boundary pressure"]
    c_bound = d["boundary concentration"]

    # Initialize variables
    p, u = readf.read_flowfield(g, flowfile)
    c = np.zeros(g.num_cells)

    phi = d["initial porosity field"]
    phi_s = 1 - phi
       
    # Initialize advection matrices
    q = readf.read_fluxfield(g, fluxfile)
    U, U_bound = ade.discretize_advection(g, bndf, q, c_bound)
   
    # Initialize monitors        
    l_time = []
    l_rate = []
    l_surf = []
    l_vol = []

    t = 0
    c_out_old = 0
    mineralVolume_old = 0

    MineralVolumeOld = np.sum(phi_s * g.cell_volumes) 

    while t <= t_end:

        c_old = c.copy()

        # Step 1 - Update effective properties    
        sigma = fvutils.gradient_of_scalar(g, phi)   

        mineralArea = np.sum(sigma * g.cell_volumes)
        mineralVolume = np.sum(phi_s * g.cell_volumes)

        relMineralVolumeChange = abs(mineralVolume-MineralVolumeOld) / MineralVolumeOld
        
        # Step 1a - Update flow if required
        if relMineralVolumeChange > updateFlowCriterion:
            print('-------------------------------------------------- Update flow --------------------------------------------------')
            d["porosity field"] = phi
            p, u, q = flowSolver.SimpleAlgorithm(g, d, bndf, s, p, u)
            U, U_bound = ade.discretize_advection(g, bndf, q, c_bound)
            MineralVolumeOld = mineralVolume
            
        psi = 4 * phi * phi_s
        sigma *= psi

        A, A_bound = ade.discretize_diffusion(
            g, bndt, diff*phi, c_bound
            )
        
        # Step 2 - Solve transport
        # Assemble all contributions to the ADE
        M = sps.diags(phi * g.cell_volumes / dt)
        R = sps.diags(- xi * kappa_gamma * sigma * g.cell_volumes)
        a = M + U + A + R
        b = U_bound + A_bound + phi * g.cell_volumes / dt * c_old

        # Solve linear system       
        c, flag = scipy.sparse.linalg.bicgstab(
            a, b, x0 = c_old, tol = tol_c, atol=1e-20#, callback=report
            )

        if flag != 0:
            print('failed bicg, exit Code', flag)
            assert False

        # Step 4 - Calculate average reaction rate
      
        c_out, Q = get_outflow_concentration(
            g, c, bndf.is_pres, q[bndf.is_pres]
            )
        rrate = Q * (c_out - c_in) / (xi * mineralArea)

        if t % 1 == 0:
            print('time', t)
            print('Mineral volume', mineralVolume) 
            l_time.append(t)
            l_surf.append(mineralArea)
            l_vol.append(mineralVolume)
            l_rate.append(rrate)
            
        c_out_old = c_out
        mineralVolume_old = mineralVolume

        # Step 5 - Update porosity      
        phi_s -= dt * sigma * kappa_gamma * c * Vm
        phi = 1 - phi_s
        assert np.all(phi) <= 1
        assert np.all(phi_s) >= 0

        if not np.all(phi_s >= 0.0):
            print('Warning solid porosity < 0')
            inds=np.argwhere(phi_s<0.0)
            phi_s[inds] = 0
            assert False
            
        if not np.all(phi <= 1.0):
            print('Warning porosity > 1')
            inds=np.argwhere(phi>1.0)
            phi[inds] = 1.
            assert False

        assert np.all(phi+phi_s) == 1

        # Step 5a - Print geometry
        if t % t_print == 0:
            outfile = evolutionFile.format(int(t))
            out_dict = {
                "phi": phi.reshape((g.Nx, g.Ny, 1), order = 'F'),
                }
            printf.write_outFile(g, out_dict, outfile)            

        # Step 6 - Proceed to next time step
        t = round(t+dt, 6)
                
    return p, u, c, phi, l_time, l_rate, l_surf, l_vol
