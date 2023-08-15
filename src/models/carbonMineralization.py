"""
Solves coupled Stokes flow and reactive transport
for basalt dissolution / calcite precipitation kinetics

This model reproduces results presented Section 4.2 of

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

thresh = 1e-16
eps = 1e-4
                
def solveRT(g, d, bndf, bndt, s, h):

    """
    Solves coupled Stokes flow and reactive transport
    for basalt dissolution / calcite precipitation kinetics

    We have 4 global transport equations (2 conservative + 2 reactive)
    and 5 local chemical equations (3 independent)

    Inputs:
    g: the grid
    d: data dictionary for fluid properties and operating conditions
    bndf: boundary conditions for flow
    bndt: boundary conditions for transport
    s: data dictionary for the flow and linear solvers
    h: data dictionary for chemical data
    
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
    VmCalcite = d["molar volume calcite"]
    VmWollastonite = d["molar volume wollastonite"]
    diff = d["diffusion coefficient"]
    _kappaDiss = d["dissolution rate constant"]
    _kappaPrec = d["precipitation rate constant"]
    startDissolutionTime = d["start Dissolution Time"]
    
    flowfile = d["input flow filename"]
    fluxfile = d["flux filename"]
    updateFlowCriterion = d["update flow criterion"]

    dt = s["time step"]
    t_end = d["end time"]
    tolNR = s["tolerance speciation"]

    gammaH = h["activity coefficient H"]
    gammaCO3 = h["activity coefficient CO3"]
    gammaCa = h["activity coefficient Ca"]
    gammaSiO2 = h["activity coefficient SiO2"]
    gammaCO2 = h["activity coefficient CO2"]
    gammaCaHCO3 = h["activity coefficient CaHCO3"]

    gamma = np.array(
        [gammaCO3, gammaH, gammaCO2, gammaCa, gammaSiO2]
        )

    logKeqH2CO3 = h["equilibrium constant carbonate"]
    logKeqCaCo3 = h["equilibrium constant calcite"]
    
    Ksp = math.pow(10, logKeqCaCo3)

    # Assign boundary conditions
    u_bound = d["boundary velocity"]
    p_bound = d["boundary pressure"]
    c1_bound = d["boundary concentration first component"]
    c2_bound = d["boundary concentration second component"]
    c3_bound = d["boundary concentration third component"]
    c4_bound = d["boundary concentration fourth component"]
 
    # Initialize variables
    p, u = readf.read_flowfield(g, flowfile)

    cH = d["initial concentration H"]
    cCO3 = d["initial concentration CO3"]
    cCO2 = d["initial concentration CO2"]
    cCa = d["initial concentration Ca"]
    cSiO2 = d["initial concentration Si"]
    
    c1 = -cCa + cCO3 + cSiO2 + cCO2
    c2 = cH + 2*cSiO2 + 2*cCO2
    c3 = cSiO2
    c4 = cCa

    phi = d["initial porosity field"]
    phi_ws = 1 - phi
    phi_cc = np.zeros(g.num_cells)
    phi_s = phi_ws + phi_cc
          
    # Initialize advection matrices
    q = readf.read_fluxfield(g, fluxfile)
   
    U1, U1_bound = ade.discretize_advection(g, bndt, q, c1_bound)
    U2, U2_bound = ade.discretize_advection(g, bndt, q, c2_bound)
    U3, U3_bound = ade.discretize_advection(g, bndt, q, c3_bound)
    U4, U4_bound = ade.discretize_advection(g, bndt, q, c4_bound)
   
    # Initialize monitors        
    l_time = []
    l_surf = []
    l_vol = []

    l_H = []
    l_Ca = []
    l_CO3 = []
    l_SiO2 = []
    l_CO2 = []
    l_CaSiO3 = []
    l_CaCO3 = []

    t = 0
    mineralVolume_old = 0

    MineralVolumeOld = np.sum(phi_s * g.cell_volumes)

    SIprec = np.zeros(g.num_cells)
    kappaPrec = np.zeros(g.num_cells)

    # Precipitation is only activated when the local SI > 1
    # Until then, calcium is treated in the same fashion as silica
    precipitation = False

    SIp = np.zeros(g.num_cells)

    while t <= t_end:

        # Step 0 - Store old values of components
        c1_old = c1.copy()
        c2_old = c2.copy()
        c3_old = c3.copy()
        c4_old = c4.copy()

        # Step 1 - Update effective properties    
        sigma = fvutils.gradient_of_scalar(g, phi)
        threshCells = np.ravel(np.argwhere(sigma < eps))
        sigma[threshCells] = 0

        mineralArea = np.sum(sigma * g.cell_volumes)
        mineralVolume = np.sum(phi_s * g.cell_volumes)

        Wollastonite = np.sum(phi_ws * g.cell_volumes)
        Calcite = np.sum(phi_cc * g.cell_volumes)

        eps_vol = abs(Wollastonite + Calcite - mineralVolume)

        # Check mass conservation
        assert eps_vol < thresh

        MineralVolumeChange = abs(mineralVolume-MineralVolumeOld)

        relMineralVolumeChange = MineralVolumeChange / MineralVolumeOld
        
        # Step 5 - Update flow if necessary
        if relMineralVolumeChange > updateFlowCriterion:
            print(
                '-------------------------------------------------- Update flow --------------------------------------------------')
            d["porosity field"] = phi
            p, u, q = flowSolver.SimpleAlgorithm(g, d, bnd, s, p, u)

            U1, U1_bound = ade.discretize_advection(g, bndt, q, c1_bound)
            U2, U2_bound = ade.discretize_advection(g, bndt, q, c2_bound)
            U3, U3_bound = ade.discretize_advection(g, bndt, q, c3_bound)
            U4, U4_bound = ade.discretize_advection(g, bndt, q, c4_bound)

            MineralVolumeOld = mineralVolume
            
        psi = 4 * phi * phi_s

        A1, A1_bound = ade.discretize_diffusion(g, bndt, diff*phi, c1_bound)
        A2, A2_bound = ade.discretize_diffusion(g, bndt, diff*phi, c2_bound)
        A3, A3_bound = ade.discretize_diffusion(g, bndt, diff*phi, c3_bound)
        A4, A4_bound = ade.discretize_diffusion(g, bndt, diff*phi, c4_bound)
        
        # Step 2 - Solve transport
        M = sps.diags(phi * g.cell_volumes / dt)

        # Component 1 - conservative
        a = M + A1 + U1
        b = U1_bound + A1_bound + phi * g.cell_volumes / dt * c1_old
        c1 = scipy.sparse.linalg.spsolve(a, b)

        # Component 2 - conservative
        a = M + A2 + U2
        b = U2_bound + A2_bound + phi * g.cell_volumes / dt * c2_old
        c2 = scipy.sparse.linalg.spsolve(a, b)

        # Component 3 - Silica (kinetic reactive)
        if t < startDissolutionTime:
            kappaDiss = 0
        else:
            kappaDiss = _kappaDiss
        a = M + A3 + U3
        R = kappaDiss * sigma * psi * g.cell_volumes
        b = U3_bound + A3_bound + phi * g.cell_volumes / dt * c3_old + R
        c3 = scipy.sparse.linalg.spsolve(a, b)

        # Component 4 - Calcium (kinetic reactive)
        if precipitation:
            a = M + A4 + U4
            Rd = kappaDiss * sigma * psi
            Rp = kappaPrec * sigma
            R = (Rd - Rp) * g.cell_volumes
            b = U4_bound + A4_bound + phi * g.cell_volumes / dt * c4_old + R
            c4 = scipy.sparse.linalg.spsolve(a, b)
        else:
            c4 = c3
           
        # Do speciation only in relevant cells,
        # where actual dissolution/precipitation is used
        # to calculate the porosity change
        # This is to save computational effort when diss./prec. is
        # localized in cells around the mineral grains. The model
        # as it is can already handle speciation in all cells,
        # however at a not negligible computational cost, and also
        # needs very small time steps in the initial calculations to
        # maintain the Newton-Raphson solution stable.
        spCells = np.ravel(np.argwhere(sigma > 0))
        for j in spCells:

            x = g.cell_centers[0, j]

            check = False

            if check:
                print('cell', j, 'sigma', sigma[j], 'phi', phi[j])

            cSiO2[j] = c3[j]
            cCa[j] = c4[j]

            x2 = cH[j]

            x3 = cCO2[j]

            x1 = cCO3[j]

            x0 = np.array([x1, x2, x3])
               
            b = np.array(
                [c1[j], c2[j], logKeqH2CO3, c3[j], c4[j]]
                )
            x, flag = newton_raphson(
                x0, b, gamma, tol = tolNR, maxiter = 10#, check=True
                )
            if flag != 0:
                print(
                    '-------------- WARNING failed speciation in cell', j, flag
                    )
                x, flag = newton_raphson(
                    x0, b, gamma, tol = tolNR, maxiter = 10, check=True
                    )
                assert False
                
            cCO3[j] = x[0]
            cH[j] = x[1]
            cCO2[j] = x[2]

        # Calculate rate of precipitation
        # from the chemical activities
        SIp = gammaCa*cCa * gammaCO3*cCO3 / Ksp
        #print(SIp[spCells])
        #print(sigma[spCells], psi[spCells])
        #print(cCa[spCells], cCO3[spCells])
        maxSIp = np.amax(SIp[spCells])

        # Check if conditions for precipitation are met
        if ((maxSIp > 1) and (not precipitation)):
            precipitation = True

        maxSIpInd = np.argmax(SIp[spCells])

        # Calculate dissolution / precipitation rates [units of 1/s]
        kappaPrec =  np.maximum(0, - _kappaPrec * (1 - SIp))
        rPrec = sigma * kappaPrec * VmCalcite
        rDiss = sigma * psi * kappaDiss * VmWollastonite

        # Calculate and store monitor quantities    
        if t % 1 == 0:

            print('time', t)
           
            l_time.append(int(t))
            l_surf.append(mineralArea)
            l_vol.append(mineralVolume)

            # Calculate locally average species concentrations
            phiV = phi * g.cell_volumes
            den = np.sum(phiV[spCells])

            avgH = np.sum(cH[spCells] * phiV[spCells]) / den
            avgCO3 = np.sum(cCO3[spCells] * phiV[spCells]) / den
            avgCO2 = np.sum(cCO2[spCells] * phiV[spCells]) / den
            avgCa = np.sum(cCa[spCells] * phiV[spCells]) / den
            avgSiO2 = np.sum(cSiO2[spCells] * phiV[spCells]) / den      

            l_H.append(avgH)
            l_CO3.append(avgCO3)
            l_CO2.append(avgCO2)
            l_Ca.append(avgCa)
            l_SiO2.append(avgSiO2)                   

            l_CaSiO3.append(Wollastonite)
            l_CaCO3.append(Calcite)

            print(
                'max precipitation index', maxSIp, 'in cell', maxSIpInd
                )
            print(
                'Average H', avgH, 'Average CO3', avgCO3
                )
            print(
                'Average CO2', avgCO2
                )
            print(
                'Average Ca', avgCa, 'Average SiO2', avgSiO2
                )
            
            print('Total Mineral volume', mineralVolume)
            print('Wollastonite volume', Wollastonite)
            print('Calcite volume', Calcite)
           
        # Step 4 - Update porosity
        phi_ws -= dt * rDiss
        phi_cc += dt * rPrec
        phi_s = phi_ws + phi_cc
        phi = 1 - phi_s

        assert np.all(phi >= 0)
        assert np.all(phi <= 1)

        assert np.all(phi_s >= 0)
        assert np.all(phi_s <= 1)

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

        """
        # is speciation is done in all cells, we need to progressively
        # adjust the time step, since very small time steps are needed
        # in the initial calculations to maintain Newton-Raphson stable
        if t == 0.02:
            dt = 0.0001
        if t == 0.1:
            dt = 0.001
        if t == 1:
            dt = 0.0025
        """
        
        # Step 6 - Proceed to next time step
        t = round(t+dt, 6)

    # Save return quantities in relevant dictionaries  
    monitors = {
        "time": l_time,
        "mineral surface": l_surf,
        "mineral volume": l_vol,
        "outflow H": l_H,
        "outflow Ca": l_Ca,
        "outflow CO3": l_CO3,
        "outflow SiO2": l_SiO2,
        "outflow CO2": l_CO2,
        "wollastonite volume": l_CaSiO3,
        "calcite volume": l_CaCO3,
        }

    species = {
        "H": cH.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "Ca": cCa.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "CO3": cCO3.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "SiO2": cSiO2.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "CO2": cCO2.reshape((g.Nx, g.Ny, 1), order = 'F'),
        }

    volumefractions = {
        "phi": phi.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "phi_cc": phi_cc.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "sigma": sigma.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "sigmaPsi": (sigma * psi).reshape((g.Nx, g.Ny, 1), order = 'F'),
        "SIp": SIp.reshape((g.Nx, g.Ny, 1), order = 'F'),
        }    

    return p, u, species, volumefractions, monitors

def jacobian(x, g):

    """
    Calculates the jacobian of the system evaluated at the current values

    Inputs:
    x: the current values
    g: the activity coefficients
   
    Returns:
    jac: the jacobian of the system
    dimensions: number of equations x number of variables
    """

    ln10 = math.log(10)

    r1 = np.array([1, 0, 1])
    r2 = np.array([0, 1, 2])
    r3 = np.array([1/(x[0]*ln10), 2/(x[1]*ln10), -1/(x[2]*ln10)])

    jac = np.vstack((r1, r2, r3))

    return jac

def function(x, b, g):

    """
    Calculates the function values evaluated at the current values

    Inputs:
    x: the current values
    b: the right hand side
    g: the activity coefficients
   
    Returns:
    func: the values of the functions
    dimensions: number of equations
    """

    # order is CO3, H, CO2

    SiO2 = b[3]
    Ca = b[4]

    g0 = g[0]
    g1 = g[1]
    g2 = g[2]

    def log10(a):
        return math.log10(a)

    f1 = x[0] + x[2] - Ca + SiO2 - b[0]
    f2 = x[1] + 2*x[2] + 2*SiO2 - b[1]
    f3 = log10(g0*x[0]) + 2*log10(g1*x[1]) - log10(g2*x[2]) - b[2]
    
    func = np.array([f1, f2, f3])

    return func

def newton_iteration(x, b, g):

    """
    Perform a Newton-Raphson iteration

    Inputs:
    x: the current values
    b: the right hand side
    g: the activity coefficients
   
    Returns:
    x_new: the updated values, dimensions: number of variables
    """

    j = jacobian(x, g)

    f = function(x, b, g)

    y = np.linalg.solve(j, -f)

    x_new = x + y
   
    return x_new

def newton_raphson(x_init, b, g, tol, maxiter, check=False):

    """
    Solves local system of chemical equations using Newton-Raphson

    Inputs:
    x_init: the initial guess
    b: the right hand side
    g: the activity coefficients
    tol: convergence criterion
    maxiter: maximum number of iterations

    Optionals
    check: used for debugging
    
    Returns:
    x_new: the converged solution, dimensions: number of variables
    flag: can be either 0 if solution is converged, or negative if failed
    """

    counter = 0
    flag = 0

    x_old = x_init.copy()

    if check:
        print('b', b)
        print(counter, x_old)

    initial_res = 0

    while counter < maxiter:

        counter += 1

        x_new = newton_iteration(x_old, b, g)
        if check:
            print(counter, x_new)

        res = np.linalg.norm(x_old-x_new) / np.linalg.norm(x_new)
        if check:
            print('residual', res)
        if res < tol:
            break

        x_old = x_new

    if counter > maxiter - 2:
        flag = - counter 

    return x_new, flag
