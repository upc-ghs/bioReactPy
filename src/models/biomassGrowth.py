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

def solveRT(g, d, bnd):

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
    diff = d["diffusion coefficient"]
    dt = d["time step"]
    u0 = d["velocity"]
    mu_max = d["mu max"]
    _lambda = d["lambda"]
    Ked = d["Ked"]
    Kea = d["Kea"]
    Y = d["yield coefficient"]
    t_end = d["end time"]
    cB0 = d["initial biomass"]

    # Assign boundary conditions
    ed_bound = d["boundary concentration ED"]
    ea_bound = d["boundary concentration EA"]
    b_bound = d["boundary concentration biomass"]

    # Initialize variables
    phi = 1 * np.ones(g.num_cells)
       
    # Initialize advection matrices
    q = np.zeros((g.dim, g.num_faces))
    q[0]=u0
    qf = q * g.face_normals[0:g.dim]/g.face_areas
    qn=qf.sum(axis=0)

    Ud, Ud_bound = ade.discretize_advection(g, bnd, qn, ed_bound)
    Ua, Ua_bound = ade.discretize_advection(g, bnd, qn, ea_bound)
    Ub, Ub_bound = ade.discretize_advection(g, bnd, qn, b_bound)
    Up, Up_bound = ade.discretize_advection(g, bnd, qn, b_bound)

    Ad, Ad_bound = ade.discretize_diffusion(g, bnd, diff*phi, ed_bound)
    Aa, Aa_bound = ade.discretize_diffusion(g, bnd, diff*phi, ea_bound)
    Ab, Ab_bound = ade.discretize_diffusion(g, bnd, diff*phi, b_bound)
    Ap, Ap_bound = ade.discretize_diffusion(g, bnd, diff*phi, b_bound)

    ced = np.zeros(g.num_cells)
    cea = np.zeros(g.num_cells)
    cp = np.zeros(g.num_cells)
    cb = cB0 * np.ones(g.num_cells)

    tol_c = 1e-5
    tol_ss = 1e-5

    listt = []
    listd = []
    lista = []
    listp = []
    listb = []

    t = 0
   
    while t <= t_end:

        ced_old = ced.copy()
        cea_old = cea.copy()
        cp_old = cp.copy()
        cb_old = cb.copy()
       
        # Step 2 - Solve transport
        # donor
        M = sps.diags(phi * g.cell_volumes / dt)
        R = -np.divide(ced, ced+Ked)*np.divide(cea, cea+Kea)*mu_max/Y*cb
        a = M + Ud + Ad
        b = Ud_bound + Ad_bound + (phi/dt*ced_old + R)*g.cell_volumes

        # Solve linear system       
        ced, flag = scipy.sparse.linalg.bicgstab(
            a, b, x0 = ced_old, tol = tol_c, atol=1e-20#, callback=report
            )

        if flag != 0:
            print('failed bicg donor, exit Code', flag)
            assert False

        # acceptor
        a = M + Ua + Aa
        b = Ua_bound + Aa_bound + (phi/dt*cea_old + R)*g.cell_volumes

        # Solve linear system       
        cea, flag = scipy.sparse.linalg.bicgstab(
            a, b, x0 = cea_old, tol = tol_c, atol=1e-20#, callback=report
            )

        if flag != 0:
            print('failed bicg acceptor, exit Code', flag)
            assert False

        # product
        a = M + Up + Ap
        b = Up_bound + Ap_bound + (phi/dt*cp_old - R)*g.cell_volumes

        # Solve linear system       
        cp, flag = scipy.sparse.linalg.bicgstab(
            a, b, x0 = cp_old, tol = tol_c, atol=1e-20#, callback=report
            )

        if flag != 0:
            print('failed bicg acceptor, exit Code', flag)
            assert False

        # biomass
        R = np.divide(ced, ced+Ked)*np.divide(cea, cea+Kea)*mu_max-_lambda
        #R = sps.diags(R*g.cell_volumes)
        #a = M - R
        #b = phi*g.cell_volumes/dt*cb_old
        # Update
        cb = 1 / (1 - dt*R) * cb_old

        """
        cb, flag = scipy.sparse.linalg.bicgstab(
            a, b, x0 = cb_old, tol = tol_c, atol=1e-20#, callback=report
            )

        if flag != 0:
            print('failed bicg biomass, exit Code', flag)
            assert False
        """

        if t % 1 == 0:

            avgced = np.sum(ced)/g.num_cells
            avgcea = np.sum(cea)/g.num_cells
            avgcp = np.sum(cp)/g.num_cells
            avgcb = np.sum(cb)/g.num_cells

            listt.append(t)
            listd.append(avgced)
            lista.append(avgcea)
            listp.append(avgcp)
            listb.append(avgcb)

            print(t, avgcb)

        # Step 6 - Proceed to next time step
        t = round(t+dt, 8)

    c = {
        "concentration ED": ced,
        "concentration EA": cea,
        "concentration product": cp,
        "concentration biomass": cb,
        }
    l = {
        "time": listt,
        "average ED": listd,
        "average EA": lista,
        "average product": listp,
        "average biomass": listb,
        }   
                          
    return c, l
