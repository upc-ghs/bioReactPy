"""
Solves coupled Stokes flow and reactive transport with biomass growth
for basalt dissolution / microbial calcite precipitation kinetics

This model reproduces results presented Section 4.3 of

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
eps = 1e-1

def solveRT(g, d, bndf, bndt, s, h):

    """
    Solves coupled Stokes flow and reactive transport with biomass growth

    We have 8 global transport equations
    (3 conservative + 3 reactive + 2 B,N)
    and 8 local chemical equations (5 independent)
    
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
    VmCalcite = d["molar volume calcite"]
    VmWollastonite = d["molar volume wollastonite"]
    diff = d["diffusion coefficient"]
    _kappaDiss = d["dissolution rate constant"]
    _kappaPrec = d["precipitation rate constant"]
    startDissolutionTime = d["start Dissolution Time"]

    kAttach = d["Biomass attachment rate"]
    kDec1 = d["Endogenous decay rate"]
    kDec2 = d["Decay rate due to calcite precipitation"]
    KU = d["Half saturation constant of urea"]
    KN = d["Half saturation constant of nutrient"]
    kappaMu = d["Maximum substrate utilization rate"]
    Y = d["Yield coefficient"]
    MU = d["molar mass urea"]
    kappaUreo = d["Urea growth rate"]
    rhoB = d["density of biofilm"]

    flowfile = d["flow filename"]
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
    gammaUrea = h["activity coefficient Urea"]
    gammaNH3 = h["activity coefficient NH3"]
    gammaNH4 = h["activity coefficient NH4"]

    gamma = np.array(
        [gammaCO3, gammaH, gammaCO2, gammaCa, gammaSiO2, gammaUrea, gammaNH3, gammaNH4]
        )

    logKeqH2CO3 = h["equilibrium constant carbonate"]
    logKeqCaCo3 = h["equilibrium constant calcite"]
    logKeqNH4 = h["equilibrium constant nitrogen"]
    
    Ksp = math.pow(10, logKeqCaCo3)

    # Assign boundary conditions
    u_bound = d["boundary velocity"]
    p_bound = d["boundary pressure"]
    c1_bnd = d["boundary concentration first component"]
    c2_bnd = d["boundary concentration second component"]
    c3_bnd = d["boundary concentration third component"]
    c4_bnd = d["boundary concentration fourth component"]
    c5_bnd = d["boundary concentration fifth component"]
    c6_bnd = d["boundary concentration sixth component"]
    cB_bnd = d["boundary concentration biomass"]
    cN_bnd = d["boundary concentration nutrient"]
    
    # Initialize variables
    p, u = readf.read_flowfield(g, flowfile)

    cH = d["initial concentration H"]
    cCO3 = d["initial concentration CO3"]
    cCO2 = d["initial concentration CO2"]
    cCa = d["initial concentration Ca"]
    cSiO2 = d["initial concentration Si"]
    cUrea = d["initial concentration Urea"]
    cNH3 = d["initial concentration NH3"]
    cNH4 = d["initial concentration NH4"]
    cB = d["initial biomass concentration"]
    cN = d["initial nutrient concentration"]

    c1 = cCO2 + cCO3 - cCa + cSiO2 + cUrea
    c2 = cH + 2*cCO2 + 2*cSiO2 + 2*cUrea + cNH4
    c3 = cNH3 + cNH4 + 2*cUrea
    c4 = cSiO2
    c5 = cCa
    c6 = cUrea
    
    phi = d["initial porosity field"]
    phi_m = d["initial mineral field"]
    phi_b = d["initial biofilm"]

    phi_ws = phi_m.copy()
    phi_cc = np.zeros(g.num_cells)
     
    # Initialize advection matrices
    q = readf.read_fluxfield(g, fluxfile)

    U1, U1_bnd = ade.discretize_advection(g, bndt, q, c1_bnd)
    U2, U2_bnd = ade.discretize_advection(g, bndt, q, c2_bnd)
    U3, U3_bnd = ade.discretize_advection(g, bndt, q, c3_bnd)
    U4, U4_bnd = ade.discretize_advection(g, bndt, q, c4_bnd)
    U5, U5_bnd = ade.discretize_advection(g, bndt, q, c5_bnd)
    U6, U6_bnd = ade.discretize_advection(g, bndt, q, c6_bnd)
    Ub, Ub_bnd = ade.discretize_advection(g, bndt, q, cB_bnd)
    Un, Un_bnd = ade.discretize_advection(g, bndt, q, cN_bnd)
    
    # Initialize monitors        
    l_time = []
    l_surf = []
    l_vol = []
    l_bio = []

    l_H = []
    l_Ca = []
    l_CO3 = []
    l_SiO2 = []
    l_CO2 = []
    l_Urea = []
    l_NH3 = []
    l_NH4 = []
    l_CaSiO3 = []
    l_CaCO3 = []

    t = 0
    mineralVolume_old = 0

    MineralVolumeOld = np.sum(phi_m * g.cell_volumes)

    SIprec = np.zeros(g.num_cells)
    kappaPrec = np.zeros(g.num_cells)

    # Precipitation is only activated when the local SI > 1
    # Until then, calcium is treated in the same fashion as silica
    precipitation = False

    SIp = np.zeros(g.num_cells)

    spCells = []

    while t <= t_end:

        # Step 0 - Store old values of components
        c1_old = c1.copy()
        c2_old = c2.copy()
        c3_old = c3.copy()
        c4_old = c4.copy()
        c5_old = c5.copy()
        c6_old = c6.copy()
        cB_old = cB.copy()
        cN_old = cN.copy()

        spCellsOld = spCells.copy()

        # Step 1 - Update effective properties    
        sigma = fvutils.gradient_of_scalar(g, phi)
        threshCells = np.ravel(np.argwhere(sigma < eps))
        sigma[threshCells] = 0

        mineralArea = np.sum(sigma * g.cell_volumes)
        mineralVolume = np.sum(phi_m * g.cell_volumes)
        biofilmVolume = np.sum(phi_b * g.cell_volumes)

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
            p, u, q = Stokes.SimpleAlgorithm(g, d, bnd, s, p, u)

            U1, U1_bnd = ade.discretize_advection(g, bndt, q, c1_bnd)
            U2, U2_bnd = ade.discretize_advection(g, bndt, q, c2_bnd)
            U3, U3_bnd = ade.discretize_advection(g, bndt, q, c3_bnd)
            U4, U4_bnd = ade.discretize_advection(g, bndt, q, c4_bnd)
            U5, U5_bnd = ade.discretize_advection(g, bndt, q, c5_bnd)
            U6, U6_bnd = ade.discretize_advection(g, bndt, q, c6_bnd)
            Ub, Ub_bnd = ade.discretize_advection(g, bndt, q, cB_bnd)
            Un, Un_bnd = ade.discretize_advection(g, bndt, q, cN_bnd)

            MineralVolumeOld = mineralVolume

        psi = 4 * phi * phi_m

        A1, A1_bnd = ade.discretize_diffusion(g, bndt, diff*phi, c1_bnd)
        A2, A2_bnd = ade.discretize_diffusion(g, bndt, diff*phi, c2_bnd)
        A3, A3_bnd = ade.discretize_diffusion(g, bndt, diff*phi, c3_bnd)
        A4, A4_bnd = ade.discretize_diffusion(g, bndt, diff*phi, c4_bnd)
        A5, A5_bnd = ade.discretize_diffusion(g, bndt, diff*phi, c5_bnd)
        A6, A6_bnd = ade.discretize_diffusion(g, bndt, diff*phi, c6_bnd)
        Ab, Ab_bnd = ade.discretize_diffusion(g, bndt, diff*phi, cB_bnd)
        An, An_bnd = ade.discretize_diffusion(g, bndt, diff*phi, cN_bnd)
        
        # Step 2 - Solve transport
        M = sps.diags(phi * g.cell_volumes / dt)

        # Component 1 - conservative
        a = M + U1 + A1
        b = U1_bnd + A1_bnd + phi * g.cell_volumes / dt * c1_old
        c1 = scipy.sparse.linalg.spsolve(a, b)

        # Component 2 - conservative
        a = M + U2 + A2
        b = U2_bnd + A2_bnd + phi * g.cell_volumes / dt * c2_old
        c2 = scipy.sparse.linalg.spsolve(a, b)

        # Component 3 - conservative
        a = M + U3 + A3
        b = U3_bnd + A3_bnd + phi * g.cell_volumes / dt * c3_old
        c3 = scipy.sparse.linalg.spsolve(a, b)
        
        # Component 4 - Silica (kinetic reactive)
        # dissolution only occurs on cells containing the mineral
        if t < startDissolutionTime:
            kappaDiss = 0
        else:
            kappaDiss = _kappaDiss
        a = M + U4 + A4
        R = kappaDiss * sigma * psi * g.cell_volumes
        b = U4_bnd + A4_bnd + phi * g.cell_volumes / dt * c4_old + R
        c4 = scipy.sparse.linalg.spsolve(a, b)

        # Component 5 - Calcium (kinetic reactive)
        # preciptation can occur everywhere near the mineral
        if precipitation:
            Rdiss = kappaDiss * sigma * psi
            Rprec = kappaPrec * sigma
            R = (Rdiss - Rprec) * g.cell_volumes
            b = U5_bnd + A5_bnd + phi * g.cell_volumes / dt * c5_old + R
            c5 = scipy.sparse.linalg.spsolve(a, b)
        else:
            c5 = c4

        # Component 6 - Urea (kinetic reactive)

        # First calculate molality from molar concentration
        # c is in mol/cm3, rho is in g/cm3, MU is in g/mol
        # as a result, molU is in mol/g
        molU = np.divide(c6_old, rho - c6_old * MU)
        # then calculate the Michaelis‐Menten kinetic term
        term = np.divide(molU, KU + molU)
        # finally, calculate the rate of ureolysis [mol/cm3 s]
        R = kappaUreo * rhoB * phi_b * term
        # assemble right hand side
        b = U6_bnd + A6_bnd + (phi / dt * c6_old - R) * g.cell_volumes
        c6 = scipy.sparse.linalg.spsolve(a, b)
           
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
        checkSpCells = np.array_equal(spCells, spCellsOld)
        if not checkSpCells:
            print('Warning: new SpCells')
            mask = np.in1d(spCells, spCellsOld, invert=True)
            newSpCells = spCells[mask]            
            print(newSpCells)
            print(sigma[newSpCells])

        for j in spCells:

            x = g.cell_centers[0, j]

            check = False

            if check:
                print('cell', j, 'sigma', sigma[j], 'phi', phi[j])

            cSiO2[j] = c4[j]
            cCa[j] = c5[j]
            cUrea[j] = c6[j]

            x1 = cCO3[j] 
            x2 = cH[j]
            x3 = cCO2[j]
            x4 = cNH3[j]
            x5 = cNH4[j]

            x0 = np.array([x1, x2, x3, x4, x5])
               
            b = np.array(
                [c1[j], c2[j], c3[j], logKeqH2CO3, logKeqNH4, c4[j], c5[j], c6[j]]
                )
            x, flag = newton_raphson(
                x0, b, gamma, tol = tolNR, maxiter = 10, check = False
                )

            if flag != 0:
                print(
                    '-------------- WARNING failed speciation in cell', j, flag
                    )
                x, flag = speciation.newton_raphson(
                    x0, b, gamma, tol = tolNR, maxiter = 10, check=True
                    )
                assert False

            assert np.all(x > 0)
                
            cCO3[j] = x[0]
            cH[j] = x[1]
            cCO2[j] = x[2]
            cNH3[j] = x[3]
            cNH4[j] = x[4]

        # Calculate rate of precipitation
        # from the chemical activities
        SIp = gammaCa*cCa * gammaCO3*cCO3 / Ksp
        maxSIp = np.amax(SIp[spCells])            

        # Check if conditions for precipitation are met
        if ((maxSIp > 1) and (not precipitation)):
            precipitation = True

        maxSIpInd = np.argmax(SIp[spCells])
        #print('time', t, 'max precipitation SI', maxSIp, 'in cell', maxSIpInd)

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
            l_bio.append(biofilmVolume)

            # Calculate locally average species concentrations
            phiV = phi * g.cell_volumes
            den = np.sum(phiV[spCells])

            avgH = np.sum(cH[spCells] * phiV[spCells]) / den
            avgCO3 = np.sum(cCO3[spCells] * phiV[spCells]) / den
            avgCO2 = np.sum(cCO2[spCells] * phiV[spCells]) / den
            avgCa = np.sum(cCa[spCells] * phiV[spCells]) / den
            avgSiO2 = np.sum(cSiO2[spCells] * phiV[spCells]) / den

            avgNH3 = np.sum(cNH3[spCells] * phiV[spCells]) / den
            avgNH4 = np.sum(cNH4[spCells] * phiV[spCells]) / den
            avgUrea = np.sum(cUrea[spCells] * phiV[spCells]) / den          

            l_H.append(avgH)
            l_CO3.append(avgCO3)
            l_CO2.append(avgCO2)
            l_Ca.append(avgCa)
            l_SiO2.append(avgSiO2)

            l_NH3.append(avgNH3)
            l_NH4.append(avgNH4)
            l_Urea.append(avgUrea)                      

            l_CaSiO3.append(Wollastonite)
            l_CaCO3.append(Calcite)

            print(
                'max precipitation index', maxSIp, 'in cell', maxSIpInd
                )
            print(
                'Average H', avgH, 'Average CO3', avgCO3
                )
            print(
                'Average CO2', avgCO2, 'Average Urea', avgUrea
                )
            print(
                'Average Ca', avgCa, 'Average SiO2', avgSiO2
                )
            print(
                'Average NH3', avgNH3, 'Average NH4', avgNH4
                ) 
            
            print('Total Mineral volume', mineralVolume)
            print('Wollastonite volume', Wollastonite)
            print('Calcite volume', Calcite)
            print('biofilm volume', biofilmVolume)

        # Component 7 - Suspended biomass
        # attachement occurs in proximity of the mineral (sigma != 0)
        muN = kappaMu * np.divide(cN, KN + cN)
        Rgrowth = muN * phi
        Rattach = kAttach * sigma
        R = sps.diags((Rgrowth - Rattach) * g.cell_volumes)
        a = M + Ub + Ab - R
        b = Ub_bnd + Ab_bnd + phi * g.cell_volumes / dt * cB_old
        cB = scipy.sparse.linalg.spsolve(a, b)

        # Update biofilm
        Rgrowth = muN * phi_b * rhoB
        Rattach = kAttach * sigma * cB
        Rdecay = (kDec1 + kDec2 * rPrec) * phi_b * rhoB
        
        phi_b += dt * (Rgrowth + Rattach - Rdecay) / rhoB
      
        # Step 4 - Update porosity
        phi_ws -= dt * rDiss
        phi_cc += dt * rPrec

        phi_m = phi_ws + phi_cc

        phi = 1 - phi_m - phi_b
       
        assert np.all(phi >= 0)
        assert np.all(phi <= 1)

        assert np.all(phi_m >= 0)
        assert np.all(phi_m <= 1)

        assert np.all(phi_m >= 0)
        assert np.all(phi_m <= 1)

        # Component 8 - Nutrient
        Rn = - muN / Y * (phi_b * rhoB + phi * cB) * g.cell_volumes
        a = M + Un + An
        b = Un_bnd + An_bnd + phi * g.cell_volumes / dt * cN_old + Rn
        cN = scipy.sparse.linalg.spsolve(a, b)

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
        "sum H": l_H,
        "sum Ca": l_Ca,
        "sum CO3": l_CO3,
        "sum SiO2": l_SiO2,
        "sum CO2": l_CO2,
        "sum NH3": l_NH3,
        "sum NH4": l_NH4,
        "sum urea": l_Urea,        
        "wollastonite volume": l_CaSiO3,
        "calcite volume": l_CaCO3,
        "biofilm": l_bio,
        }

    species = {
        "H": cH.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "Ca": cCa.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "CO3": cCO3.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "SiO2": cSiO2.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "CO2": cCO2.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "NH3": cNH3.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "NH4": cNH4.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "Urea": cUrea.reshape((g.Nx, g.Ny, 1), order = 'F'),
        }

    volumefractions = {
        "phi": phi.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "phi_cc": phi_cc.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "sigma": sigma.reshape((g.Nx, g.Ny, 1), order = 'F'),
        "sigmaPsi": (sigma * psi).reshape((g.Nx, g.Ny, 1), order = 'F'),
        "SIp": SIp.reshape((g.Nx, g.Ny, 1), order = 'F'),
        }    

    return p, u, species, volumefractions, monitors

def jacobian(x, g, b):

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

    g0 = g[0]
    g1 = g[1]
    g2 = g[2]
    g3 = g[6]
    g4 = g[7]
    
    r1 = np.array([1, 0, 1, 0, 0])
    r2 = np.array([0, 1, 2, 0, 1])
    r3 = np.array([0, 0, 0, 1, 1])
    r4 = np.array([1/(x[0]*ln10), 2/(x[1]*ln10), -1/(x[2]*ln10), 0, 0])
    r5 = np.array([0, 1/(x[1]*ln10), 0, 1/(x[3]*ln10), -1/(x[4]*ln10)])
    #r4 = np.array([g0*math.pow(g1*x[1], 2), 2*g0*x[0]*g1*g1*x[1], -b[3]*g2, 0, 0])
    #r5 = np.array([0, g1*g3*x[3], 0, g1*x[1]*g3, -b[4]*g4])
    
    jac = np.vstack((r1, r2, r3, r4, r5))

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
    
    # order is CO3, H, CO2, NH3, NH4

    SiO2 = b[5]
    Ca = b[6]
    Urea = b[7]

    g0 = g[0]
    g1 = g[1]
    g2 = g[2]
    g3 = g[6]
    g4 = g[7]

    def log10(a):
        return math.log10(a)

    f1 = x[0] + x[2] - Ca + SiO2 + Urea - b[0]
    f2 = x[1] + 2*x[2] + 2*SiO2 + 2*Urea + x[4] - b[1]
    f3 = x[3] + x[4] + 2*Urea - b[2]
    f4 = log10(g0*x[0]) + 2*log10(g1*x[1]) - log10(g2*x[2]) - b[3]
    f5 = log10(g1*x[1]) + log10(g3*x[3]) - log10(g4*x[4]) - b[4]
    #f4 = g0*x[0] * math.pow(g1*x[1], 2) - b[3] * g2*x[2]
    #f5 = g1*x[1] * g3*x[3] - b[4] * g4*x[4]
    
    func = np.array([f1, f2, f3, f4, f5])

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
    
    j = jacobian(x, g, b)

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
