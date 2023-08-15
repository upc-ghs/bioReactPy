"""

********* Calcite dissolution Benchmark *********

This example solves calcite dissolution at the continuum scale.

This is the runscript to obtain the results presented in Section 4.3 of

Starnoni (2023)
bioReactPy
(Submitted)

For geometry and flow, see xie1Dflow.py

"""

import numpy as np
import xlsxwriter
import math

from geometry import structured
from geometry.bc import BoundaryCondition, BoundaryConditionTransport
from models import CalciteDissolution
from importExport import readf, printf
import time

thresh = 1e-16

# Input Files
flowfile = 'xie.vtk'
fluxfile = 'xie.xlsx'

# Output files
cfile = 'outputFiles/concentrations/xie.vtk'
workbook = xlsxwriter.Workbook('outputFiles/monitors/xie120.xlsx')
phiFile = 'outputFiles/plots/xieCalcite.vtk'

_1d = 24 * 60 * 60
_1y = 365 * _1d
t_end = 120 * _1y

FlowModel = 'microcontinuum'
FlowSolver = 'Simple'

# Geometry from Xie et al, 2015 [m]
L = 2
N = 80

# Create grid
g = structured.CartGrid(N, L)
g.compute_geometry()

print('Nx', g.Nx, 'dx', g.dx)
print('Number of cells', g.num_cells)
  
# BC for flow
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > L - 1e-10))

bnd_faces = np.concatenate((left_faces, right_faces))

bnd_cond_faces = ['pres'] * bnd_faces.size

bc_flow = BoundaryCondition(
    g, bnd_faces, bnd_cond_faces
    )

# BC for transport
bnd_cond_faces = ['dir'] * left_faces.size
bc_transport = BoundaryConditionTransport(
    g, left_faces, bnd_cond_faces
    )

# Input data
rho_water = 1000 # kg/m3
mu = 1e-3 # Pa s
p_in = 70 # Pa
p_out = 0

# Assign boundary conditions values
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

p_bound[right_faces] = p_out
p_bound[left_faces] = p_in

porosity = 0.35
calcite = 0.3
inert = 0.35

initial_permeability = 1.186e-11 #[m2]

phi = porosity * np.ones(g.num_cells)
phi_cc = calcite * np.ones(g.num_cells)
phi_inert = inert * np.ones(g.num_cells)

# Activity coefficients
# For now assume dilute solution
gammaH = 733e-6 #convert 1000 cm3/mol to m3/mol
gammaCO3 = 290e-6
gammaCa = 290e-6
gammaCO2 = 733e-6
gammaSO4 = 290e-6
gammaH2SO4 = 733e-6
gammaCAHCO3 = 733e-6
gammaCAOH = 1000e-6
gammaCASO4 = 1000e-6

# Equilibrium constants
logK1 = math.log10(1.70e-3) # CO2 => H2CO3
logK2 = math.log10(2.5e-4) # H2CO3 => H + HCO3
logK3 = math.log10(5e-11) # HCO3 => H + CO3
logKeqH2CO3 = logK1 + logK2 + logK3 # CO2 => 2H + CO3
logKeqCaCo3 = -8.4750 #Ca + CO3 => CaCO3

logH2SO4 = -2 # H2SO4 <=> H + HSO4
logHSO4 = 1.9 # HSO4 <=> H + SO4

KeqHSO4 = math.pow(10, logHSO4)

logKeqH2SO4 = logH2SO4 + logHSO4
print('logKeqH2CO3', logKeqH2CO3)
print('logKeqCaCo3', logKeqCaCo3)
print('logKeqH2SO4', logKeqH2SO4)

logKeqCAHCO3 = -11.435
logKeqCAOH = -1.4#-12.78

print('logKeqCAHCO3', logKeqCAHCO3)
print('logKeqCAOH', logKeqCAOH)

KeqH2CO3 = math.pow(10, logKeqH2CO3)
KeqH2SO4 = math.pow(10, logKeqH2SO4)
KeqCAHCO3 = math.pow(10, logKeqCAHCO3)
KeqCAOH = math.pow(10, logKeqCAOH)
KeqCASO4 = 2.4e-5

logKeqCASO4 = math.log10(KeqCASO4)
print('logKeqCASO4', logKeqCASO4)

Ksp = math.pow(10, logKeqCaCo3)

# Initial conditions
pH_init = 9.38
cCa_init = 1.566e-4 * rho_water #units of mol/kg H2O
cCO3_init = 2.56e-4 * rho_water #mol/kg H2O
cSO4_init = 9.97e-11 * rho_water #mol/kg H2O

cH_init = math.pow(10, -pH_init)/gammaH #units of mol/m3
aH = gammaH * cH_init

# guess of primary species [from xie supp. information]
CO3_guess = 2.428e-5 * rho_water
Ca_guess = 1.364e-4 * rho_water
SO4_guess = 8.838e-11 * rho_water

aCO3 = gammaCO3 * CO3_guess
aCa = gammaCa * Ca_guess
aSO4 = gammaSO4 * SO4_guess

# CO2 is in equilibrium with water, H and CO3
aCO2 = aCO3 * math.pow(aH, 2) / KeqH2CO3
CO2_guess = aCO2/gammaCO2

# H2SO4 is in equilibrium with water, H and SO4
aH2SO4 = aSO4 * math.pow(aH, 2) / KeqH2SO4
H2SO4_guess = aH2SO4/gammaH2SO4

# CAHCO3 is in equilibrium with water, H, Ca and CO3
aCAHCO3 = aH * aCa * aCO3 / KeqCAHCO3
CAHCO3_guess = aCAHCO3/gammaCAHCO3

"""
# CASO4 is in equilibrium with water, H, and Ca
aCASO4 = aCa * aSO4 / KeqCASO4
CASO4_guess = aCASO4/gammaCASO4

# CAOH is in equilibrium with water, H, and Ca
aCAOH = aCa / aH / KeqCAOH
CAOH_guess = aCAOH/gammaCAOH
print('CAOH_guess', CAOH_guess)
assert False
"""

x1 = CO3_guess
x2 = Ca_guess
x3 = CO2_guess
x4 = SO4_guess
x5 = H2SO4_guess
x6 = CAHCO3_guess
#x7 = CASO4_guess

#x8 = CAOH_guess

x0 = np.array([x1, x2, x3, x4, x5, x6])

gamma = np.array(
    [gammaCO3, gammaCa, gammaCO2, gammaSO4, gammaH2SO4,
     gammaCAHCO3, gammaH]
    )
tolNR = 1e-5

b = np.array(
    [cSO4_init, cCa_init, cCO3_init, logKeqH2CO3, logKeqH2SO4,
     logKeqCAHCO3, cH_init]
    )
x, flag = CalciteDissolution.Initial_newton_raphson(
    x0, b, gamma, tol = tolNR, maxiter = 10, check=True
    )

cCO3_init = x[0]
cCa_init = x[1]
cCO2_init = x[2]
cSO4_init = x[3]
cH2SO4_init = x[4]
cCAHCO3_init = x[5]
#cCASO4_init = x[6]
#cCAOH_init = x[7]

print('initial cH', cH_init)
print('initial cCO3', cCO3_init)
print('initial cCO2', cCO2_init)
print('initial cCa', cCa_init)
print('initial cSO4', cSO4_init)
print('initial cH2SO4', cH2SO4_init)
print('initial cCAHCO3', cCAHCO3_init)
#print('initial cCASO4', cCASO4_init)
#print('initial cCAOH', cCAOH_init)

aCO3 = cCO3_init*gammaCO3
aCa = cCa_init*gammaCa
aSO4 = cSO4_init*gammaSO4

# add OH
logKeqOH = -14
KeqOH = math.pow(10, logKeqOH)
aOH = KeqOH / aH
cOH = aOH/gammaH

# add HCO3
logKeqHCO3 = -10.329
KeqHCO3 = math.pow(10, logKeqHCO3)
aHCO3 = aH * aCO3 / KeqHCO3
cHCO3 = aHCO3/gammaH

# add CaOH
aCAOH = aCa * aOH * KeqCAOH
cCAOH = aCAOH/gammaH
print('cCAOH', cCAOH)

# add HSO4-
aHSO4 = aH * aSO4 / KeqHSO4
cHSO4 = aHSO4/gammaH
print('cHSO4', cHSO4)

#molality
A = 0.51
B = 3.29

I = 0.5 * (
    cH_init + cCAHCO3_init + 4 * (cCa_init+cCO3_init+cSO4_init)
    + cOH + cHCO3 + cHSO4 + cCAOH
    ) #/ 1000 #in mol/dmc
print('I calculated', I)
a1 = 2.87 # Ca
a2 = 2.81 # CO3
log1 = - 4 * A * math.sqrt(I)/(1+B*a1*math.sqrt(I))
log2 = - 4 * A * math.sqrt(I)/(1+B*a2*math.sqrt(I))
log1 = - 4 * A * (math.sqrt(I)/(1+math.sqrt(I))-0.20825*I)
log2 = log1
gamma1 = math.pow(10, log1)*1e-3
gamma2 = math.pow(10, log2)*1e-3
print('gammaCa', gamma1)
print('gammaCO3', gamma2)
SI = gamma1 * cCa_init * gamma2 * cCO3_init
print('calculated SI', SI/Ksp)

# Initialize species concentrations with Initial conditions
cH_in = cH_init * np.ones(g.num_cells)
cCO3_in = cCO3_init * np.ones(g.num_cells)
cCO2_in = cCO2_init * np.ones(g.num_cells)
cCa_in = cCa_init * np.ones(g.num_cells)
cSO4_in = cSO4_init * np.ones(g.num_cells)
cH2SO4_in = cH2SO4_init * np.ones(g.num_cells)
cCAHCO3_in = cCAHCO3_init * np.ones(g.num_cells)

# Boundary conditions
pH_bnd = 3
cCa_bnd = 9.97e-5 * rho_water #units of mol/kg H2O
cCO3_bnd = 9.97e-3 * rho_water #9.97e-3 #mol/kg H2O
cSO4_bnd = 6.457e-4 * rho_water #mol/kg H2O

cH_bnd = math.pow(10, -pH_bnd)/gammaH #units of mol/m3
aH = gammaH * cH_bnd

# guess [from xie supp. information]
CO3_guess = 2.087e-13 * rho_water
Ca_guess = 7.652e-5 * rho_water
SO4_guess = 4.863e-4 * rho_water

aCO3 = gammaCO3 * CO3_guess
aCa = gammaCa * Ca_guess
aSO4 = gammaSO4 * SO4_guess

# CO2 is in equilibrium with water, H and CO3
aCO2 = aCO3 * math.pow(aH, 2) / KeqH2CO3
CO2_guess = aCO2/gammaCO2

# H2SO4 is in equilibrium with water, H and SO4
aH2SO4 = aSO4 * math.pow(aH, 2) / KeqH2SO4
H2SO4_guess = aH2SO4/gammaH2SO4

# CAHCO3 is in equilibrium with water, H, Ca and CO3
aCAHCO3 = aH * aCa * aCO3 / KeqCAHCO3
CAHCO3_guess = aCAHCO3/gammaCAHCO3

"""
# CASO4 is in equilibrium with water, H, and Ca
aCASO4 = aCa * aSO4 / KeqCASO4
CASO4_guess = aCASO4/gammaCASO4

# CAOH is in equilibrium with water, H, and Ca
aCAOH = aCa / aH / KeqCAOH
CAOH_guess = aCAOH/gammaCAOH
print('CAOH_guess', CAOH_guess)
assert False
"""

x1 = CO3_guess
x2 = Ca_guess
x3 = CO2_guess
x4 = SO4_guess
x5 = H2SO4_guess
x6 = CAHCO3_guess
#x7 = CASO4_guess
#x8 = CAOH_guess

x0 = np.array([x1, x2, x3, x4, x5, x6])

gamma = np.array(
    [gammaCO3, gammaCa, gammaCO2, gammaSO4, gammaH2SO4,
     gammaCAHCO3, gammaH]
    )
tolNR = 1e-5

b = np.array(
    [cSO4_bnd, cCa_bnd, cCO3_bnd, logKeqH2CO3, logKeqH2SO4,
     logKeqCAHCO3, cH_bnd]
    )
x, flag = CalciteDissolution.Initial_newton_raphson(
    x0, b, gamma, tol = tolNR, maxiter = 10, check=True
    )

cCO3_bnd = x[0]
cCa_bnd = x[1]
cCO2_bnd = x[2]
cSO4_bnd = x[3]
cH2SO4_bnd = x[4]
cCAHCO3_bnd = x[5]
#cCASO4_init = x[6]
#cCAOH_init = x[7]

print('boundary cH', cH_bnd)
print('boundary cCO3', cCO3_bnd)
print('boundary cCO2', cCO2_bnd)
print('boundary cCa', cCa_bnd)
print('boundary cSO4', cSO4_bnd)
print('boundary cH2SO4', cH2SO4_bnd)
print('boundary cCAHCO3', cCAHCO3_bnd)
#print('initial cCASO4', cCASO4_init)
#print('initial cCAOH', cCAOH_init)

I = 0.5 * (
    cH_bnd + cCAHCO3_bnd + 4 * (cCa_bnd+cCO3_bnd+cSO4_bnd)
    )/rho_water
print('I boundary', I)

"""
#molality
cH_bnd = cH_bnd/rho_water
cCa_bnd = cCa_bnd/rho_water
cCO3_bnd = cCO3_bnd/rho_water
cCO2_bnd = cCO2_bnd/rho_water
cSO4_bnd = cSO4_bnd/rho_water
cH2SO4_bnd = cH2SO4_bnd/rho_water

I = 0.5 * (cH_bnd + 4 * (cCa_bnd+cCO3_bnd+cSO4_bnd))
print('I', I)
A = 1.172
lngammaH = - A * math.sqrt(I)
gammaH = math.exp(lngammaH)
print('gammaH', gammaH)

lngammaCa = - A * 4 * math.sqrt(I)
gammaCa = math.exp(lngammaCa)
print('gammaCa', gammaCa)
assert False
"""

# Assign boundary conditions to the components
c1_bound = np.zeros(g.num_faces)
c2_bound = np.zeros(g.num_faces)
c3_bound = np.zeros(g.num_faces)
c4_bound = np.zeros(g.num_faces)

c1_bound[left_faces] = cH_bnd + 2*cH2SO4_bnd + 2*cCO2_bnd + cCAHCO3_bnd
c2_bound[left_faces] = cSO4_bnd + cH2SO4_bnd
c3_bound[left_faces] = cCa_bnd - cCO2_bnd - cCO3_bnd
c4_bound[left_faces] = cCO2_bnd + cCO3_bnd + cCAHCO3_bnd

# Critical time stations
UpdateFlow = 0.001

VmCalcite = 36.93e-6 # convert cm3/mol to m3/mol
kappaDiss = 5e-5 # mol/m2/s
sigma = 1 # m2/m3

# Store data for flow in a dictionary
dataFlow = {
    "fluid density": rho_water,
    "fluid viscosity": mu,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    "boundary concentration first component": c1_bound,
    "boundary concentration second component": c2_bound,
    "boundary concentration third component": c3_bound,
    "boundary concentration fourth component": c4_bound,
    "initial concentration H": cH_in,
    "initial concentration CO3": cCO3_in,
    "initial concentration CO2": cCO2_in,
    "initial concentration Ca": cCa_in,
    "initial concentration SO4": cSO4_in,
    "initial concentration H2SO4": cH2SO4_in,
    "initial concentration CAHCO3": cCAHCO3_in,
    "initial porosity field": phi,
    "initial calcite field": phi_cc,
    "initial inert field": phi_inert,
    "initial permeability": initial_permeability,
    "molar volume calcite": VmCalcite,
    "specific area": sigma,
    "diffusion coefficient": 0,
    "dissolution rate constant": kappaDiss,
    "end time": t_end,
    "input flow filename": flowfile,
    "flux filename": fluxfile,
    "update flow criterion": UpdateFlow,
    }

# Store chemical data in a dictionary
dataChemical = {
    "activity coefficient H": gammaH,
    "activity coefficient CO2": gammaCO2,
    "activity coefficient CO3": gammaCO3,
    "activity coefficient Ca": gammaCa,
    "activity coefficient SO4": gammaSO4,
    "activity coefficient H2SO4": gammaH2SO4,
    "activity coefficient CAHCO3": gammaCAHCO3,
    "equilibrium constant carbonate": logKeqH2CO3,
    "equilibrium constant calcite": logKeqCaCo3,
    "equilibrium constant sulphuric acid": logKeqH2SO4,
    "equilibrium constant CAHCO3": logKeqCAHCO3,
    "activity constant": A,
    }

# SOR data
omega_u = 0.5
omega_p = 0.1
iter_u = 25
tol_u = 0.2
tol_p = 1e-2 # inner tolerance pressure correction
convergence_criterion = 1e-4
tol_continuity = 1e-8
tol_discharge = 1e-2
outer_iterations = 500
cfl = 1
q = readf.read_fluxfield(g, fluxfile)
u_max = np.amax(q)
print('u_max', u_max)

dt = cfl * g.dx / u_max
print('umax-based time step', dt)

dt = 50

print('time step', dt)

# Store data for the linear solver in a dictionary
dataSolver = {
    "omega u": omega_u,
    "omega p": omega_p,
    "inner iterations u": iter_u,
    "inner tolerance u": tol_u,
    "inner tolerance p": tol_p,
    "tolerance continuity": tol_continuity,
    "convergence criterion": convergence_criterion,
    "tolerance discharge": tol_discharge,
    "max outer iterations": outer_iterations,
    "tolerance speciation": tolNR,
    "time step": dt,
    "Flow model": FlowModel,
    }

# Solve Coupled Flow & Reactive Transport
start = time.time()
p, u, l = CalciteDissolution.solveRT(
    g, dataFlow, bc_flow, bc_transport, dataSolver, dataChemical
    )
end = time.time()
elapsed_time = end - start
hours, mod = divmod(elapsed_time, 3600)
mins, secs = divmod(mod, 60)
print(
    'elapsed time in hours', int(hours), 'h', int(mins), 'm', int(secs), 's'
    ) 

#Postprocess
 
# 1 Unpack and print monitor quantities
worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0

array = np.array(
    [np.arange(g.num_cells), p, l["phi"], l["phi_cc"],
     l["time"], l["flux"]], dtype=object
    )
# Iterate over the data and write it out row by row.

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()
    
