"""

********* Steady flow for Soulaine ideal porous medium *********

This example solves flow in an idealized porous medium. The steady-state
flow solution is then used as initial conditions for the bio-geochemical
simulations, see 'runCarbonMineralizion' and 'runBiomineralization'.

The computational domain is rectangular (12 × 1 mm ) with ten circular
grains (diameter of 0.6 mm ) homogeneously distributed along the tube.
This idealized porous medium is commonly used to investigate complex flow
and transport in porous media at the pore scale. e.g. in reference:

Soulaine et al. (2017). Mineral dissolution and wormholing from a
pore-scale perspective. Journal of Fluid Mechanics, 827, 457-483.

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

Boundary conditions consist of velocity at the inflow and pressure
at the outflow, no-slip conditions at the tube solid boundaries and
internal grain boundaries.

a: radius of the spheres
N: number of cells along the radius

"""

import numpy as np
import xlsxwriter

from geometry import structured
from geometry.bc import BoundaryCondition
from models import flowSolver
from utils import fvutils
from importExport import printf
import time

            
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

# Output files
filename = 'outputFiles/plots/SoulaineN12.vtk'
workbook = xlsxwriter.Workbook('outputFiles/residuals/SoulaineFlowResidualsN12.xlsx')
qfworkbook = xlsxwriter.Workbook('outputFiles/fluxes/SoulaineN12.xlsx')

FlowModel = 'NoSlip'
FlowSolver = 'Simple'

# Geometry from Soulaine et al, 2017 [cm]
Nl = 1 # number of layers
NS = 10 # number of spheres

Lx = 1.2
Ly = 0.1
a = 0.03
N = 12
dx = a/N
Nx = round(Lx / dx)
Ny = round(Ly / dx)
domain = np.array([Lx, Ly])
basedim = np.array([Nx, Ny])

# Create grid
g = structured.CartGrid(basedim, domain)
g.compute_geometry()

print('Nx', g.Nx, 'Ny', g.Ny, 'dx', g.dx)
print('Number of cells', g.num_cells)

# Create Solid mineral Grain cells
# Start with the first main layer
xc = 0.15
yc = Ly/2
seed = np.array([xc, yc])
Delta = Ly
grain_cells = create_geometry(g, a, seed, NS, Delta)

#Boundary conditions 
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))

flow_faces = np.concatenate((left_faces, right_faces))
wall_faces = np.concatenate((bot_faces, top_faces))

bnd_faces = np.concatenate((flow_faces, wall_faces))
bnd_cond_faces = ['vel'] * left_faces.size + ['pres'] * right_faces.size + ['wall'] * wall_faces.size

bnd_wall_cells = []
if FlowModel == 'NoSlip':
    bnd_wall_cells = grain_cells
    
bound_cond = BoundaryCondition(
    g, bnd_faces, bnd_cond_faces, wall_cells = bnd_wall_cells
    )

#Input data
rho = 1 # g/cm3
ni = 1e-2 # cm2/s
mu = rho * ni
u_in = 0.115#8e-5
p_out = 0

# Assign boundary conditions values
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

p_bound[right_faces] = p_out
u_bound[0, left_faces] = u_in

m = 12 * mu * u_in / pow(Ly,2)
y = g.face_centers[1, left_faces]
u_ex = m / (2*mu) * y * (Ly - y)
u_bound[0, left_faces] = u_ex

# Store flow data in a dictionary
dataFlow = {
    "fluid density": rho,
    "fluid viscosity": mu,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    }

# Initialize porosity field (in a microcontinuum framework)
if FlowModel == 'microcontinuum':
    solid_porosity = 0.001
    initial_permeability = 1e-15
    phi = np.ones(g.num_cells)
    phi[grain_cells] = solid_porosity

    dataFlow["porosity field"] = phi
    dataFlow["initial permeability"] = initial_permeability
    
# Linear Solver data
omega_u = 0.5 # under-relaxation velocity
omega_p = 0.1 # under-relaxation pressure correction
iter_u = 25 # inner iterations velocity
tol_u = 0.2 # inner tolerance velocity
tol_p = 1e-2 # inner tolerance pressure correction
convergence_criterion = 1e-5 # establish when flow solution has converged (for SIMPLE)
tol_continuity = 1e-10 # Absolute tolerance for continuity imbalance
outer_iterations = 2000 # max number of iterations
tol_steady_state = 1e-4 # establish when flow solution has reached steady-state (for PISO)

# Store data for linear solver in a dictionary
dataSolver = {
    "omega u": omega_u,
    "omega p": omega_p,
    "inner iterations u": iter_u,
    "inner tolerance u": tol_u,
    "inner tolerance p": tol_p,
    "tolerance continuity": tol_continuity,
    "convergence criterion": convergence_criterion,
    "max outer iterations": outer_iterations,
    "tolerance steady-state": tol_steady_state,
    "Flow solver": FlowSolver,
    "Flow model": FlowModel,
    }

#Initialize variables
p_guess = np.zeros(g.num_cells)
u_guess = np.zeros((g.dim, g.num_cells))
 
#Solve Flow
return_residuals = True

start = time.time()
p, u, q, outer, res_u, res_div, res_q = flowSolver.StokesSolver(
    g, dataFlow, bound_cond, dataSolver, p_guess, u_guess, return_residuals
    )
end = time.time()
print('elapsed time', end - start)
            
q_ex = u_in * Ly
print('exact discharge', q_ex)

qmax = np.amax(q)
print('max velocity', qmax)

# Postprocess and write monitor results
t_arr = np.array(outer)
u_arr = np.array(res_u)
div_arr = np.array(res_div)
q_arr = np.array(res_q)

worksheet = workbook.add_worksheet()
qfworksheet = qfworkbook.add_worksheet()

row = 0

array = np.array([t_arr, u_arr, div_arr, q_arr])
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

array = np.array([np.arange(g.num_faces), q])
for col, data in enumerate(array):
    qfworksheet.write_column(row, col, data)
qfworkbook.close()

p = p.reshape((Nx, Ny, 1), order = 'F')
vx = u[0].reshape((Nx, Ny, 1), order = 'F')
vy = u[1].reshape((Nx, Ny, 1), order = 'F')
vz = np.zeros(g.num_cells).reshape((Nx, Ny, 1), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, filename)


