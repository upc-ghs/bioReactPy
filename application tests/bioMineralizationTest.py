"""

********* Carbon BIOmineralization in an ideal porous medium *********

This example solves the time evolution of flow and concentration fields
in an idealized porous medium, in the PRESENCE of biomass growth,
using the microcontinuum approach.

This is the runscript to obtain the results presented in Section 4.3 of

Starnoni and Sanchez-Vila (2023)
Pore-scale modelling of microbially enhanced carbon mineralization
(submitted)

For geometry and flow, see soulaineFlow.py

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

For the reference geochemical problem without biomass growth,
see carbonMineralizationTest.py
Now, urea is injected together with dissolved CO2 through the inflow
boundary, with the aim of increasing the pH of the aqueous solution,
eventually becoming basic, and hence enhancing carbon precipitation.

Eight transport equations are solved globally and eight chemical
equations are solved locally to give the species concentrations.
"""

import numpy as np
import xlsxwriter
import math

from geometry import structured
from geometry.bc import BoundaryCondition, BoundaryConditionTransport
from models import bioMineralization
from utils import fvutils
from importExport import readf, printf
import time

thresh = 1e-16

def create_geometry(g, a, seed, numberSpheres, incr):

    # centre of the first sphere
    xc = seed[0]
    yc = seed[1]

    y = g.cell_centers[1] - yc

    #first sphere
    x = g.cell_centers[0] - xc
    r = np.sqrt(x**2 + y**2)
    cells = np.ravel(np.argwhere(r <= a))
        
    for j in range (1, numberSpheres):
        xc += incr
        x = g.cell_centers[0] - xc
        r = np.sqrt(x**2 + y**2)
        new_cells = np.ravel(np.argwhere(r <= a))
        cells = np.concatenate((cells, new_cells))

    return cells

# Input Files
flowfile = 'outputFiles/plots/SoulaineN12.vtk'
fluxfile = 'outputFiles/fluxes/SoulaineN12.xlsx'

# Output files
cfile = 'outputFiles/concentrations/n12kub0.002.vtk'
workbook = xlsxwriter.Workbook('outputFiles/monitors/n12kub0.002.xlsx')
phiFile = 'outputFiles/plots/n12kub0.002.vtk'

FlowModel = 'microcontinuum'
FlowSolver = 'Simple'

# This is the key parameter we investigate here
# k_ub = {0.001, 0.002, 0.004, 0.01}
k_ub = 0.002
print('Urease content in biofilm', k_ub)

N = 12 # must be multiple of 3!!
t_end = 60

# Use the reference geochemical case
injection = 'moderate'

diff = 4e-7 #cm2/s

# Basalt dissolution rate from Daval et al (2009)
logK25Golubev = - 6.42 # Golubev et al. (2005)

logK25 = logK25Golubev

kappaDiss = math.pow(10, logK25) * 1e-4 #Units of [mol/cm2 s]

startDissolutionTime = 0

# precipitation constant is maintained fixed
kappaPrec = 6.67e-11 # from Chou et al (1989)

# Geometry from Soulaine et al, 2017 [cm]
Nl = 1 # number of layers
NS = 10 # number of spheres

factor = 1
Lx = 1.2 *factor
Ly = 0.1 *factor
a = 0.03 *factor
N = 12 # must be multiple of 3!!
dx = a / N
Nx = round(Lx / dx)
Ny = round(Ly / dx)
domain = np.array([Lx, Ly])
basedim = np.array([Nx, Ny])

# Create grid
g = structured.CartGrid(basedim, domain)
g.compute_geometry()

print('Nx', g.Nx, 'Ny', g.Ny, 'dx', g.dx)
print('Number of cells', g.num_cells)

print('Injection conditions', injection)
print('diffusivity', diff)

# Create Solid mineral Grain cells
# Start with the first main layer
xc = 0.15 * factor
yc = Ly/2
seed = np.array([xc, yc])
Delta = Ly
calcite_cells = create_geometry(g, a, seed, NS, Delta)
    
# BC for flow
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))

flow_faces = np.concatenate((left_faces, right_faces))
wall_faces = np.concatenate((bot_faces, top_faces))

bnd_faces = np.concatenate((flow_faces, wall_faces))
bnd_cond_faces = ['vel'] * left_faces.size + ['pres'] * right_faces.size + ['wall'] * wall_faces.size

bc_flow = BoundaryCondition(
    g, bnd_faces, bnd_cond_faces
    )

# BC for transport
bnd_cond_faces = ['dir'] * left_faces.size
bc_transport = BoundaryConditionTransport(
    g, left_faces, bnd_cond_faces
    )

#Input data
rho_f = 1 # g/cm3
ni = 1e-2 # cm2/s
mu = rho_f * ni
u_in = 0.115
print('inlet velocity', u_in)
p_out = 0
solid_porosity = 0.001
initial_permeability = 1e-15

VmCalcite = 36.9 #calcite molar volume
VmWollastonite = 39.83 #cm3/mol

print('dissolution rate', kappaDiss)
print('precipitation rate', kappaPrec)
print('dissolution start time', startDissolutionTime)

# Calculate dimensionless numbers
Re = 0.09
Lc = Re * mu / (rho_f * u_in)
print('characteristic length', Lc)

Ae = math.pi * 2 * a / Ly**2 #cm-1
print('Ae', Ae)

# Activity coefficients
# For now assume dilute solution
gammaH = 1000 #cm3/mol
gammaCO3 = 1000
gammaCa = 1000
gammaSiO2 = 1000
gammaCO2 = 1000
gammaUrea = 1000
gammaNH3 = 1000
gammaNH4 = 1000

Pe = u_in * Lc / diff
r = kappaDiss*gammaH # cm/s
Da = r / (Ae * diff)
print('Peclet number', Pe)
print('Damkholer number', Da)

#******************** Biological input data *************************

# Data ureolysis from Qin et al (2016)
k_urea = 0.7067 # mol / g s

kappaUreo = k_urea * k_ub
print('kappa Ureolysis', kappaUreo)

# Biomass attachment rate
kAttach = 6.15e-8 #cm/s
print('Biomass attachment rate', kAttach)

# half saturation constant urea
KU = 0.355e-3 # mol/g

# half saturation constant nutrient
KN = 7.99e-4 * 1e-3 # g/l to g/cm3

# Maximum substrate utilization rate
kappaMu = 4.1667e-5 # 1/s
print('growth rate', kappaMu)

# Endogenous decay rate
kDec1 = 3.18e-7 # 1/s
print('Endogenous decay rate', kDec1)

# Decay rate due to calcite precipitation
kDec2 = 1.0 # [-]
print('Decay rate due to calcite precipitation', kDec2)

# Yield coefficient
Y = 0.5
print('Yield coefficient', Y)

# molar mass urea
MU = 60.06 # g/mol
# density of biofilm
rhoBio = 0.2e-3 # g/l to g/cm3

# Equilibrium constants
logK1 = math.log10(1.70e-3) # CO2 => H2CO3
logK2 = math.log10(2.5e-4) # H2CO3 => H + HCO3
logK3 = math.log10(5e-11) # HCO3 => H + CO3

logKeqH2CO3 = logK1 + logK2 + logK3 # CO2 => 2H + CO3
logKeqCaCo3 = -8.48 #Ca + CO3 => CaCO3
logKeqNH4 = -9.252

print('logKeqH2CO3', logKeqH2CO3)
print('logKeqCaCo3', logKeqCaCo3)
print('logKeqNH4', logKeqNH4)

# Initial conditions from Gysi and Stefansson (2012)
pH_init = 7.54
cCO2_init = 354e-6 #units of mol/kg
cCa_init = 71e-6 #mol/kg

cH_init = math.pow(10, -pH_init)/gammaH #units of mol/cm3

# need to convert mol/kg into mol/cm3
# multiply times density of water
rho_water = rho_f * 1e-3 #units kg/cm3
cCO2_init *= rho_water
cCa_init *= rho_water
cSi_init = cCa_init
print('initial cH', cH_init)
print('initial cCO2', cCO2_init)
print('initial cCa', cCa_init)
print('initial cSi', cSi_init)

# CO2 is in quilibrium with water, H and carbonate
KeqH2CO3 = math.pow(10, logKeqH2CO3)
aCO2 = gammaCO2 * cCO2_init
aH = gammaH * cH_init
aCO3 = aCO2 * KeqH2CO3 / math.pow(aH, 2)
cCO3_init = aCO3/gammaCO3
print('initial cCO3', cCO3_init)

# Initial conditions biological problem  from Qin et al (2016)
cUrea_init = 0 # mol/l to mol/cm3
cNtot_init = 0.187e-12 # mol/l to mol/cm3
print('initial Urea concentration', cUrea_init)
print('initial total Nitrogen', cNtot_init)

cBio_init = 0
print('initial biomass concentration', cBio_init)

cNut_init = 0
print('initial nutrient concentration', cNut_init)

# NH4 and NH3 are in equilibrium with H
KeqNH4 = math.pow(10, logKeqNH4)
term = KeqNH4/aH
cNH4 = cNtot_init / (1+term)
cNH3 = cNtot_init - cNH4
assert cNH4+cNH3 == cNtot_init
print('initial cNH4', cNH4)
print('initial cNH3', cNH3)

# Initialize species concentrations with Initial conditions
cH_in = cH_init * np.ones(g.num_cells)
cCO3_in = cCO3_init * np.ones(g.num_cells)
cCO2_in = cCO2_init * np.ones(g.num_cells)
cCa_in = cCa_init * np.ones(g.num_cells)
cSi_in = cSi_init * np.ones(g.num_cells)
cUrea_in = cUrea_init * np.ones(g.num_cells)
cNH3_in = cNH3 * np.ones(g.num_cells)
cNH4_in = cNH4 * np.ones(g.num_cells)
cB_in = cBio_init * np.ones(g.num_cells)
cN_in = cNut_init * np.ones(g.num_cells)

# Initialize porosity field
phi_m = np.zeros(g.num_cells)
phi_m[calcite_cells] = 1 - solid_porosity

# Initialize biofilm
initial_biofilm = 0.1
print('initial biofilm volume fraction', initial_biofilm)

sigma = fvutils.gradient_of_scalar(g, phi_m)
biofilmCells = np.ravel(np.argwhere((sigma > 1e-4) & (phi_m < 1e-10)))
        
phi_b = np.zeros(g.num_cells)
phi_b[biofilmCells] = initial_biofilm

phi = 1 - phi_m - phi_b

# Boundary conditions
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

c1_bnd = np.zeros(g.num_faces)
c2_bnd = np.zeros(g.num_faces)
c3_bnd = np.zeros(g.num_faces)
c4_bnd = np.zeros(g.num_faces)
c5_bnd = np.zeros(g.num_faces)
c6_bnd = np.zeros(g.num_faces)

cB_bnd = np.zeros(g.num_faces)
cN_bnd = np.zeros(g.num_faces)

p_bound[right_faces] = p_out

m = 12 * mu * u_in / pow(Ly,2)
y = g.face_centers[1, left_faces]
u_ex = m / (2*mu) * y * (Ly - y)

# Use the fully-developed flow field
u_bound[0, left_faces] = u_in
u_bound[0, left_faces] = u_ex

# Injected solution also from Gysi and Stefansson (2012)
# low pCO2: pH = 5.65, CO2 = 12 mmol/kg
# moderate: pH = 4.29, CO2 = 40.0 mmol/kg
# elevated: pH = 3.59, CO2 = 305.2 mmol/kg
cCO2_bnd = 40.0e-3 * rho_water
pH_bnd = 4.29

cH_bnd = math.pow(10, -pH_bnd)/gammaH #units of mol/cm3
cCa_bnd = 0.3e-3 * rho_water
cSi_bnd = cCa_bnd

print('boundary cH', cH_bnd)
print('boundary cCO2', cCO2_bnd)
print('boundary cCa', cCa_bnd)
print('boundary cSi', cSi_bnd)

# CO2 is in quilibrium with water, H and carbonate
aCO2 = gammaCO2 * cCO2_bnd
aH = gammaH * cH_bnd
aCO3 = aCO2 * KeqH2CO3 / math.pow(aH, 2)
cCO3_bnd = aCO3/gammaCO3
print('boundary cCO3', cCO3_bnd)

# Boundary conditions biological problem from Qin et al (2016)
cUrea_bnd = 0.33e-3 # mol/l to mol/cm3
cNtot_bnd = cNtot_init # mol/l to mol/cm3
print('boundary urea concentration', cUrea_bnd)
print('boundary total Nitrogen', cNtot_bnd)

cBio_bnd = 0.01e-3 #g/l to g/cm3, same units of rhoBio
print('boundary biomass concentration', cBio_bnd)

cNut_bnd = 3e-3 #g/l to g/cm3, same units of rhoBio and KN
print('boundary nutrient', cNut_bnd)

# NH4 and NH3 are in equilibrium with H
term = KeqNH4/aH
cNH4_bnd = cNtot_bnd / (1 + term)
cNH3_bnd = cNtot_bnd - cNH4_bnd
assert cNH4_bnd+cNH3_bnd == cNtot_bnd
print('boundary cNH4', cNH4_bnd)
print('boundary cNH3', cNH3_bnd)

# Assign boundary conditions to the components
c1_bnd[left_faces] = cCO2_bnd + cCO3_bnd - cCa_bnd + cSi_bnd + cUrea_bnd
c2_bnd[left_faces] = cH_bnd + 2*cCO2_bnd + 2*cSi_bnd + 2*cUrea_bnd + cNH4_bnd
c3_bnd[left_faces] = cNtot_bnd + 2*cUrea_bnd
c4_bnd[left_faces] = cSi_bnd
c5_bnd[left_faces] = cCa_bnd
c6_bnd[left_faces] = cUrea_bnd

cB_bnd[left_faces] = cBio_bnd
cN_bnd[left_faces] = cNut_bnd

# Critical time stations
UpdateFlow = 0.001

# Store data for flow in a dictionary
dataFlow = {
    "fluid density": rho_f,
    "fluid viscosity": mu,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    "boundary concentration first component": c1_bnd,
    "boundary concentration second component": c2_bnd,
    "boundary concentration third component": c3_bnd,
    "boundary concentration fourth component": c4_bnd,
    "boundary concentration fifth component": c5_bnd,
    "boundary concentration sixth component": c6_bnd,
    "boundary concentration biomass": cB_bnd,
    "boundary concentration nutrient": cN_bnd,
    "initial concentration H": cH_in,
    "initial concentration CO3": cCO3_in,
    "initial concentration CO2": cCO2_in,
    "initial concentration Ca": cCa_in,
    "initial concentration Si": cSi_in,
    "initial concentration Urea": cUrea_in,
    "initial concentration NH3": cNH3_in,
    "initial concentration NH4": cNH4_in,
    "initial biomass concentration": cB_in,
    "initial nutrient concentration": cN_in,
    "initial porosity field": phi,
    "initial mineral field": phi_m,
    "initial biofilm": phi_b,
    "initial permeability": initial_permeability,
    "molar volume calcite": VmCalcite,
    "molar volume wollastonite": VmWollastonite,
    "diffusion coefficient": diff,
    "dissolution rate constant": kappaDiss,
    "precipitation rate constant": kappaPrec,
    "Biomass attachment rate": kAttach,
    "end time": t_end,
    "flow filename": flowfile,
    "flux filename": fluxfile,
    "update flow criterion": UpdateFlow,
    "start Dissolution Time": startDissolutionTime,
    "molar mass urea": MU,
    "Half saturation constant of urea": KU,
    "Half saturation constant of nutrient": KN,
    "Maximum substrate utilization rate": kappaMu,
    "Yield coefficient": Y,
    "Urea growth rate": kappaUreo,
    "Endogenous decay rate": kDec1,
    "Decay rate due to calcite precipitation": kDec2,
    "density of biofilm": rhoBio,
    }

# Store chemical data in a dictionary
dataChemical = {
    "activity coefficient H": gammaH,
    "activity coefficient CO3": gammaCO3,
    "activity coefficient Ca": gammaCa,
    "activity coefficient SiO2": gammaSiO2,
    "activity coefficient CO2": gammaCO2,
    "activity coefficient Urea": gammaUrea,
    "activity coefficient NH3": gammaNH3,
    "activity coefficient NH4": gammaNH4,
    "equilibrium constant carbonate": logKeqH2CO3,
    "equilibrium constant nitrogen": logKeqNH4,
    "equilibrium constant calcite": logKeqCaCo3,
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
tol_steady_state = 1e-8
tol_trasport = 1e-6
tolNR = 1e-5
    
cfl = 1
q = readf.read_fluxfield(g, fluxfile)
u_max = np.amax(q)
print('u_max', u_max, 'u_in', u_in)

dt = cfl * g.dx / u_max
print('umax-based time step', dt)

dt = 0.004 #/16

if N == 12:
    dt = 0.002

print('time step', dt)

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
p, u, c, phi, l = bioMineralization.solveRT(
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
t = l["time"]
S = l["mineral surface"]
V = l["mineral volume"]
H = l["sum H"]
Ca = l["sum Ca"]
CO3 = l["sum CO3"]
SiO2 = l["sum SiO2"]
CO2 = l["sum CO2"]
NH3 = l["sum NH3"]
NH4 = l["sum NH4"]
Urea = l["sum urea"]
Ws = l["wollastonite volume"]
Cc = l["calcite volume"]
B = l["biofilm"]

t_arr = np.array(t)
s_arr = np.array(S)
v_arr = np.array(V)
H_arr = np.array(H)
Ca_arr = np.array(Ca)
CO3_arr = np.array(CO3)
SiO2_arr = np.array(SiO2)
CO2_arr = np.array(CO2)
NH3_arr = np.array(NH3)
NH4_arr = np.array(NH4)
Urea_arr = np.array(Urea)

ws_arr = np.array(Ws)
Cc_arr = np.array(Cc)
B_arr = np.array(B)

worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0

array = np.array(
    [t_arr, s_arr, v_arr, ws_arr, Cc_arr, B_arr,
     t_arr, H_arr, CO3_arr, CO2_arr, Ca_arr, SiO2_arr, Urea_arr, NH3_arr, NH4_arr]
    )
# Iterate over the data and write it out row by row.

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

# 2 print outfiles    
printf.write_outFile(g, c, cfile)
printf.write_outFile(g, phi, phiFile)

