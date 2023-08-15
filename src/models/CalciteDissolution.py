"""
Solves coupled Stokes flow and reactive transport
for calcite dissolution kinetics

This model reproduces results presented Section 4.2 of

Starnoni (2023)
bioReactPy
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
import xlsxwriter

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

    d["kozeny-carman"] = 'xie'

    # Retrieve Input data
    rho = d["fluid density"]
    mu = d["fluid viscosity"]
    initial_permeability = d["initial permeability"]
    VmCalcite = d["molar volume calcite"]
    kappaDiss = d["dissolution rate constant"]
    
    flowfile = d["input flow filename"]
    fluxfile = d["flux filename"]
    updateFlowCriterion = d["update flow criterion"]

    dt = s["time step"]
    t_end = d["end time"]
    tolNR = s["tolerance speciation"]

    gammaH = h["activity coefficient H"] * np.ones(g.num_cells)
    gammaCO2 = h["activity coefficient CO2"] * np.ones(g.num_cells)
    gammaCO3 = h["activity coefficient CO3"] * np.ones(g.num_cells)
    gammaCa = h["activity coefficient Ca"] * np.ones(g.num_cells)
    gammaSO4 = h["activity coefficient SO4"] * np.ones(g.num_cells)
    gammaH2SO4 = h["activity coefficient H2SO4"] * np.ones(g.num_cells)
    gammaCAHCO3 = h["activity coefficient CAHCO3"] * np.ones(g.num_cells)
    A = h["activity constant"]

    logKeqH2CO3 = h["equilibrium constant carbonate"]
    logKeqCaCo3 = h["equilibrium constant calcite"]
    logKeqH2SO4 = h["equilibrium constant sulphuric acid"]
    logKeqCAHCO3 = h["equilibrium constant CAHCO3"]
    
    Ksp = math.pow(10, logKeqCaCo3)
    print('Ksp', Ksp)

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
    cSO4 = d["initial concentration SO4"]
    cH2SO4 = d["initial concentration H2SO4"]
    cCAHCO3 = d["initial concentration CAHCO3"]
    
    c1 = cH + 2*cH2SO4 + 2*cCO2 + cCAHCO3
    c2 = cSO4 + cH2SO4
    c3 = cCa - cCO2 - cCO3
    c4 = cCO2 + cCO3 + cCAHCO3

    phi = d["initial porosity field"].copy()
    phi_cc = d["initial calcite field"]

    phi_cc_init = phi_cc.copy()

    phi_inert = d["initial inert field"]
          
    # Initialize advection matrices
    q = readf.read_fluxfield(g, fluxfile)
   
    U1, U1_bound = ade.discretize_advection(g, bndt, q, c1_bound)
    U2, U2_bound = ade.discretize_advection(g, bndt, q, c2_bound)
    U3, U3_bound = ade.discretize_advection(g, bndt, q, c3_bound)
    U4, U4_bound = ade.discretize_advection(g, bndt, q, c4_bound)
   
    t = 0
    mineralVolume_old = 0

    MineralVolumeOld = np.sum(phi_cc * g.cell_volumes)

    IAP = np.zeros(g.num_cells)

    sigma_init = d["specific area"]

    _1d = 24 * 60 * 60
    _1y = 365 * _1d

    t_list = []
    q_list = []

    print_years = np.array([10, 100])

    while t <= t_end:

        # Step 0 - Store old values of components
        c1_old = c1.copy()
        c2_old = c2.copy()
        c3_old = c3.copy()
        c4_old = c4.copy()

        # Step 1 - Update effective properties
        term = np.power(np.divide(phi_cc, phi_cc_init), 2/3)
        sigma = sigma_init * term

        mineralVolume = np.sum(phi_cc * g.cell_volumes)

        MineralVolumeChange = abs(mineralVolume-MineralVolumeOld)

        relMineralVolumeChange = MineralVolumeChange / MineralVolumeOld
        
        # Step 5 - Update flow if necessary
        if relMineralVolumeChange > updateFlowCriterion:
            print(
                '-------------------------------------------------- Update flow --------------------------------------------------')
            d["porosity field"] = phi.copy()
            p, u, q = flowSolver.SimpleAlgorithm(g, d, bndf, s, p, u)

            U1, U1_bound = ade.discretize_advection(g, bndt, q, c1_bound)
            U2, U2_bound = ade.discretize_advection(g, bndt, q, c2_bound)
            U3, U3_bound = ade.discretize_advection(g, bndt, q, c3_bound)
            U4, U4_bound = ade.discretize_advection(g, bndt, q, c4_bound)

            MineralVolumeOld = mineralVolume
       
        # Step 2 - Solve transport
        M = sps.diags(phi * g.cell_volumes / dt)

        # Component 1 - conservative
        a = M + U1
        b = U1_bound + phi * g.cell_volumes / dt * c1_old
        c1 = scipy.sparse.linalg.spsolve(a, b)

        # Component 2 - conservative
        a = M + U2
        b = U2_bound + phi * g.cell_volumes / dt * c2_old
        c2 = scipy.sparse.linalg.spsolve(a, b)

        # Component 3 - conservative
        a = M + U3
        b = U3_bound + phi * g.cell_volumes / dt * c3_old
        c3 = scipy.sparse.linalg.spsolve(a, b)

        # Component 4 - kinetic reactive
        a = M + U4
        IAP = gammaCa * cCa * gammaCO3 * cCO3
        Rd = kappaDiss * sigma * np.maximum(0, (1 - IAP / Ksp))
        R = Rd * phi_cc * g.cell_volumes
        b = U4_bound + phi * g.cell_volumes / dt * c4_old + R
        c4 = scipy.sparse.linalg.spsolve(a, b)
           
        # Do speciation only in relevant cells,
        for j in np.arange(g.num_cells):

            x = g.cell_centers[0, j]

            check = False

            if check:
                print('cell', j, 'sigma', sigma[j], 'phi', phi[j])

            x1 = cCO3[j]
            x2 = cH[j]
            x3 = cCO2[j]
            x4 = cCa[j]
            x5 = cSO4[j]
            x6 = cH2SO4[j]
            x7 = cCAHCO3[j]

            x0 = np.array([x1, x2, x3, x4, x5, x6, x7])

            gamma = np.array(
                [gammaCO3[j], gammaH[j], gammaCO2[j], gammaCa[j],
                 gammaSO4[j], gammaH2SO4[j], gammaCAHCO3[j]]
                )
            
            b = np.array(
                [c1[j], c2[j], c3[j], c4[j],
                 logKeqH2CO3, logKeqH2SO4, logKeqCAHCO3]
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
            cCa[j] = x[3]
            cSO4[j] = x[4]
            cH2SO4[j] = x[5]
            cCAHCO3[j] = x[6]

            """       
            #update activity coefficient
            I = 0.5 * (
                cH[j] + cCAHCO3[j] + 4 * (cCa[j]+cCO3[j]+cSO4[j])
                ) / rho
            gamma1 = math.exp(- A * math.sqrt(I))*1e-3
            gamma2 = math.exp(- A * 4 * math.sqrt(I))*1e-3

            gammaH[j] = gamma1
            gammaCAHCO3[j] = gamma1
            gammaCO3[j] = gamma2
            gammaCa[j] = gamma2
            gammaSO4[j] = gamma2
            """
            
        # Calculate dissolution / precipitation rates [units of 1/s]
        rDiss = Rd * VmCalcite
          
        # Step 4 - Update porosity
        dphi = dt * rDiss
        phi_cc -= dphi
        phi += dphi

        threshold_cells = phi_cc < eps
        phi_cc[threshold_cells] = 0

        _1d = 24 * 60 * 60

        assert np.all(phi >= 0)
        assert np.all(phi <= 1)

        assert np.all(phi_cc >= 0)
        assert np.all(phi_cc <= 1)

        if not np.all(phi_cc >= 0.0):
            print('Warning solid porosity < 0')
            inds=np.argwhere(phi_cc<0.0)
            phi_cc[inds] = 0
            assert False
            
        if not np.all(phi <= 1.0):
            print('Warning porosity > 1')
            inds=np.argwhere(phi>1.0)
            phi[inds] = 1.
            assert False

        assert np.all(phi+phi_cc+phi_inert) == 1
        
        # Step 6 - Proceed to next time step
        if t / _1d == 10:
            dt = 600

        if t % _1y == 0:
            print('time', t/_1y, 'year')
            print(phi_cc)
            print(p)
            t_list.append(t/_1y)
            q_list.append(q[g.num_faces-1])

        if int(t/_1y) in print_years:
            
            filepath = 'outputFiles/monitors/xie{0}.xlsx'
            outfile = filepath.format(int(t/_1y))
            workbook = xlsxwriter.Workbook(outfile)
            monitors = {
                "phi": phi,
                "phi_cc": phi_cc,
                "p": p,
                "N": g.num_cells,
                }
            print_monitors(monitors, workbook)

        t = round(t+dt, 6)
        

    monitors = {
        "phi": phi,
        "phi_cc": phi_cc,
        "time": np.array(t_list),
        "flux": np.array(q_list),
        }    

    return p, u, monitors

def print_monitors(l, wb):
    
    worksheet = wb.add_worksheet()

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0

    array = np.array(
        [np.arange(l["N"]), l["p"], l["phi"], l["phi_cc"]]
        )
    # Iterate over the data and write it out row by row.

    for col, data in enumerate(array):
        worksheet.write_column(row, col, data)
    wb.close()

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

    # order is CO3 (0), H (1), CO2 (2), Ca (3), SO4 (4), H2SO4 (5),
    # CAHCO3 (6)

    ln10 = math.log(10)

    r1 = np.array([0, 1, 2, 0, 0, 2, 1])
    r2 = np.array([0, 0, 0, 0, 1, 1, 0])
    r3 = np.array([-1, 0, -1, 1, 0, 0, 0])
    r4 = np.array([1, 0, 1, 0, 0, 0, 1])
    r5 = np.array(
        [1/(x[0]*ln10), 2/(x[1]*ln10), -1/(x[2]*ln10), 0, 0, 0, 0]
        )
    r6 = np.array(
        [0, 2/(x[1]*ln10), 0, 0, 1/(x[4]*ln10), -1/(x[5]*ln10), 0]
        )
    r7 = np.array(
        [1/(x[0]*ln10), 1/(x[1]*ln10), 0, 1/(x[3]*ln10), 0, 0,
         -1/(x[6]*ln10)]
        )

    jac = np.vstack((r1, r2, r3, r4, r5, r6, r7))

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

    # order is CO3 (0), H (1), CO2 (2), Ca (3), SO4 (4), H2SO4 (5),
    # CAHCO3 (6)
    g0 = g[0]
    g1 = g[1]
    g2 = g[2]
    g3 = g[3]
    g4 = g[4]
    g5 = g[5]
    g6 = g[6]

    def log10(a):
        return math.log10(a)

    f1 = x[1] + 2 * x[2] + 2 * x[5] + x[6] - b[0]
    f2 = x[4] + x[5] - b[1]
    f3 = x[3] - x[0] - x[2] - b[2]
    f4 = x[0] + x[2] + x[6] - b[3]
    f5 = log10(g0*x[0]) + 2*log10(g1*x[1]) - log10(g2*x[2]) - b[4]
    f6 = log10(g4*x[4]) + 2*log10(g1*x[1]) - log10(g5*x[5]) - b[5]
    f7 = log10(g0*x[0]) + log10(g1*x[1]) + log10(g3*x[3]) - log10(g6*x[6]) - b[6]

    func = np.array([f1, f2, f3, f4, f5, f6, f7])

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

def jacobian2(x, g):

    """
    Calculates the jacobian of the system evaluated at the current values

    Inputs:
    x: the current values
    g: the activity coefficients
   
    Returns:
    jac: the jacobian of the system
    dimensions: number of equations x number of variables
    """

    # order is CO3 (0), Ca (1), CO2 (2), SO4 (3), H2SO4 (4),
    # CaHCO3 (5), CASO4 (6), CAOH (7)

    ln10 = math.log(10)

    r1 = np.array([0, 0, 0, 1, 1, 0])
    r2 = np.array([0, 1, 0, 0, 0, 1])#np.array([-1, 1, -1, 0, 0, 0])
    r3 = np.array([1, 0, 1, 0, 0, 1])
    r4 = np.array(
        [1/(x[0]*ln10), 0, -1/(x[2]*ln10), 0, 0, 0]
        )
    r5 = np.array(
        [0, 0, 0, 1/(x[3]*ln10), -1/(x[4]*ln10), 0]
        )
    r6 = np.array(
        [1/(x[0]*ln10), 1/(x[1]*ln10), 0, 0, 0, -1/(x[5]*ln10)]
        )
    """
    r7 = np.array(
        [0, 1/(x[1]*ln10), 0, 1/(x[3]*ln10), 0, 0, -1/(x[6]*ln10)]
        )
    """
    
    jac = np.vstack((r1, r2, r3, r4, r5, r6))

    return jac

def function2(x, b, g):

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

    # order is CO3 (0), Ca (1), CO2 (2), SO4 (3), H2SO4 (4),
    # CaHCO3 (5), CASO4 (6), CAOH (7)
     
    g0 = g[0]
    g1 = g[1]
    g2 = g[2]
    g3 = g[3]
    g4 = g[4]
    g5 = g[5]
    g6 = g[6]
    
    aH = g6 * b[6]

    def log10(a):
        return math.log10(a)

    f1 = x[3] + x[4] - b[0] #S tot
    f2 = x[1] + x[5] - b[1]#- x[0] - x[2] - b[1] #Ca tot
    f3 = x[0] + x[2] + x[5] - b[2] # C tot
    f4 = log10(g0*x[0]) + 2*log10(aH) - log10(g2*x[2]) - b[3] #H2CO3
    f5 = log10(g3*x[3]) + 2*log10(aH) - log10(g4*x[4]) - b[4] #H2SO4
    f6 = log10(g1*x[1]) + log10(aH) + log10(g0*x[0]) - log10(g5*x[5]) - b[5] #CAHCO3
    #f7 = log10(g1*x[1]) + log10(g3*x[3]) - log10(g6*x[6]) - b[6] #CASO4
    
    func = np.array([f1, f2, f3, f4, f5, f6])

    return func

def newton_iteration2(x, b, g):

    """
    Perform a Newton-Raphson iteration

    Inputs:
    x: the current values
    b: the right hand side
    g: the activity coefficients
   
    Returns:
    x_new: the updated values, dimensions: number of variables
    """

    j = jacobian2(x, g)

    f = function2(x, b, g)

    y = np.linalg.solve(j, -f)

    x_new = x + y
   
    return x_new

def Initial_newton_raphson(x_init, b, g, tol, maxiter, check=False):

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

        x_new = newton_iteration2(x_old, b, g)
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
