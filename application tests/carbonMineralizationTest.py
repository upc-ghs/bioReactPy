"""

********* Carbon mineralization in an ideal porous medium *********

This example solves the time evolution of flow and concentration fields
in an idealized porous medium, in the ABSENCE of biomass growth,
using the microcontinuum approach.

This is the runscript to obtain the results presented in Section 4.2 of

Starnoni and Sanchez-Vila (2023)
Pore-scale modelling of microbially enhanced carbon mineralization
(submitted)

For geometry and flow, see soulaineFlow.py

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

Dissolved CO2 is injected into the domain through the inflow boundary,
the primary wollastonite mineral grains undergo dissolution thereby
releasing calcium ions, which then react with the carbonate
to precipitate in the form of secondary calcite.

Four transport equations are solved globally and five chemical equations
are solved locally to give the species concentrations.
"""

import numpy as np
import xlsxwriter
import math

from geometry import structured
from geometry.bc import BoundaryCondition, BoundaryConditionTransport
from models import carbonMineralization
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
cfile = 'outputFiles/concentrations/n12.vtk'
workbook = xlsxwriter.Workbook('outputFiles/monitors/n12.xlsx')
phiFile = 'outputFiles/plots/n12.vtk'

FlowModel = 'microcontinuum'
FlowSolver = 'Simple'

N = 12 # must be multiple of 3!!

# Two CO2 injection conditions to test
injection = 'moderate'
#injection = 'elevated'

t_end = 30

# We investigate here diffusivity and dissolution rate
diff = 4e-7 #cm2/s

# Basalt dissolution rate from Daval et al (2009)
LogK25Weissbart = - 8.04 # Weissbart and Rimstidt (2000)
logK25Xie = - 7.61 # Xie and Walther (1994)
logK25Golubev = - 6.42 # Golubev et al. (2005)

logK25 = logK25Golubev
#logK25 = logK25Xie

kappaDiss = math.pow(10, logK25) * 1e-4 #Units of [mol/cm2 s]
#kappaDiss = 1e-10
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
neu_faces = np.concatenate((right_faces, wall_faces))
bnd_cond_faces = ['dir'] * left_faces.size + ['neu'] * neu_faces.size
bc_transport = BoundaryConditionTransport(
    g, bnd_faces, bnd_cond_faces
    )

# Input data
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
gammaCaHCO3 = 1000

Pe = u_in * Lc / diff

r = kappaDiss*gammaH # cm/s
Da = r / (Ae * diff)
print('Peclet number', Pe)
print('Damkholer number', Da)

# Equilibrium constants
logK1 = math.log10(1.70e-3) # CO2 => H2CO3
logK2 = math.log10(2.5e-4) # H2CO3 => H + HCO3
logK3 = math.log10(5e-11) # HCO3 => H + CO3

logKeqH2CO3 = logK1 + logK2 + logK3 # CO2 => 2H + CO3
logKeqCaCo3 = -8.48 #Ca + CO3 => CaCO3

print('logKeqH2CO3', logKeqH2CO3)
print('logKeqCaCo3', logKeqCaCo3)

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

# Initialize species concentrations with Initial conditions
cH_in = cH_init * np.ones(g.num_cells)
cCO3_in = cCO3_init * np.ones(g.num_cells)
cCO2_in = cCO2_init * np.ones(g.num_cells)
cCa_in = cCa_init * np.ones(g.num_cells)
cSi_in = cSi_init * np.ones(g.num_cells)

# Initialize porosity field
phi = np.ones(g.num_cells)
phi[calcite_cells] = solid_porosity

# Boundary conditions
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

c1_bound = np.zeros(g.num_faces)
c2_bound = np.zeros(g.num_faces)
c3_bound = np.zeros(g.num_faces)
c4_bound = np.zeros(g.num_faces)

p_bound[right_faces] = p_out

m = 12 * mu * u_in / pow(Ly,2)
y = g.face_centers[1, left_faces]
u_ex = m / (2*mu) * y * (Ly - y)

# Use the fully-developed flow field
u_bound[0, left_faces] = u_in
u_bound[0, left_faces] = u_ex

# Injected solution also from Gysi and Stefansson (2012)
# moderate: pH = 4.29, CO2 = 40.0 mmol/kg
# elevated: pH = 3.59, CO2 = 305.2 mmol/kg
cCO2_bnd = 305.0e-3 * rho_water
pH_bnd = 3.59
if injection == 'moderate':
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

# Assign boundary conditions to the components
c1_bound[left_faces] = cCO2_bnd + cCO3_bnd - cCa_bnd + cSi_bnd
c2_bound[left_faces] = cH_bnd + 2*cCO2_bnd + 2*cSi_bnd
c3_bound[left_faces] = cSi_bnd
c4_bound[left_faces] = cCa_bnd

# Critical time stations
UpdateFlow = 0.01

# Store data for flow in a dictionary
dataFlow = {
    "fluid density": rho_f,
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
    "initial concentration Si": cSi_in,
    "initial porosity field": phi,
    "initial permeability": initial_permeability,
    "molar volume calcite": VmCalcite,
    "molar volume wollastonite": VmWollastonite,
    "diffusion coefficient": diff,
    "dissolution rate constant": kappaDiss,
    "precipitation rate constant": kappaPrec,
    "end time": t_end,
    "input flow filename": flowfile,
    "flux filename": fluxfile,
    "update flow criterion": UpdateFlow,
    "start Dissolution Time": startDissolutionTime,
    }

# Store chemical data in a dictionary
dataChemical = {
    "activity coefficient H": gammaH,
    "activity coefficient CO3": gammaCO3,
    "activity coefficient Ca": gammaCa,
    "activity coefficient SiO2": gammaSiO2,
    "activity coefficient CO2": gammaCO2,
    "activity coefficient CaHCO3": gammaCaHCO3,
    "equilibrium constant carbonate": logKeqH2CO3,
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
tolNR = 1e-5
cfl = 1
q = readf.read_fluxfield(g, fluxfile)
u_max = np.amax(q)
print('u_max', u_max, 'u_in', u_in)

dt = cfl * g.dx / u_max
print('umax-based time step', dt)

dt = 0.004

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
p, u, c, phi, l = carbonMineralization.solveRT(
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
H = l["outflow H"]
Ca = l["outflow Ca"]
CO3 = l["outflow CO3"]
SiO2 = l["outflow SiO2"]
CO2 = l["outflow CO2"]
Ws = l["wollastonite volume"]
Cc = l["calcite volume"]

t_arr = np.array(t)
s_arr = np.array(S)
v_arr = np.array(V)
H_arr = np.array(H)
Ca_arr = np.array(Ca)
CO3_arr = np.array(CO3)
SiO2_arr = np.array(SiO2)
CO2_arr = np.array(CO2)
ws_arr = np.array(Ws)
Cc_arr = np.array(Cc)

worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0

array = np.array(
    [t_arr, s_arr, v_arr, ws_arr, Cc_arr,
     t_arr, H_arr, CO3_arr, CO2_arr, Ca_arr, SiO2_arr]
    )
# Iterate over the data and write it out row by row.

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

# 2 print outfiles    
printf.write_outFile(g, c, cfile)
printf.write_outFile(g, phi, phiFile)
    
