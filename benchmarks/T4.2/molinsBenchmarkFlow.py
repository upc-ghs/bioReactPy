"""

********* Steady flow for Molins dissolution benchmark *********

This example solves flow in an idealized porous medium. The steady-state
flow solution is then used as initial conditions for the dissolution
benchmark, see part2benchmark.py.

Molins et al. (2021). Simulation of mineral dissolution at the pore scale
with evolving fluid-solid interfaces: Review of approaches and benchmark
problem set. Computational Geosciences, 25, 1285-1318.

A two-dimensional rectangular domain is considered, with a circular
mineral grain placed at its center

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
from importExport import printf
import time

# Output files
filename = 'molinsBenchmarkN8.vtk'
qfworkbook = xlsxwriter.Workbook('molinsBenchmarkN8.xls')

FlowModel = 'microcontinuum'
FlowSolver = 'Simple'

#Geometry
a = 0.01
N = 8
dx = a / N
Lx = 0.1
Ly = 0.05
Nx = int(Lx / dx)
Ny = int(Ly / dx)
domain = np.array([Lx, Ly])
basedim = np.array([Nx, Ny])

# Create grid
g = structured.CartGrid(basedim, domain)
g.compute_geometry()

print('Nx', g.Nx, 'Ny', g.Ny, 'dx', g.dx)
print('Number of cells', g.num_cells)

#Boundary conditions 
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))

xc = g.cell_centers[0]-domain[0]/2
yc = g.cell_centers[1]-domain[1]/2
r = np.sqrt(xc**2 + yc**2)
calcite_cells = np.ravel(np.argwhere(r <= a))

flow_faces = np.concatenate((left_faces, right_faces))
wall_faces = np.concatenate((bot_faces, top_faces))

bnd_faces = np.concatenate((flow_faces, wall_faces))
bnd_cond_faces = ['vel'] * left_faces.size + ['pres'] * right_faces.size + ['wall'] * wall_faces.size

bnd_wall_cells = []
if FlowModel == 'NoSlip':
    bnd_wall_cells = calcite_cells
    
bound_cond = BoundaryCondition(
    g, bnd_faces, bnd_cond_faces, wall_cells = bnd_wall_cells
    )

#Input data
rho = 1
ni = 1e-2
mu = rho * ni
u_in = 0.12
p_out = 0

# Assign boundary conditions values
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

p_bound[right_faces] = p_out
u_bound[0, left_faces] = u_in

# Store flow data in a dictionary 
dataFlow = {
    "fluid density": rho,
    "fluid viscosity": mu,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    "kozeny-carman": 'molins',
    }

# Initialize porosity field (in a microcontinuum framework)
if FlowModel == 'microcontinuum':
    solid_porosity = 1e-2
    initial_permeability = 1e-11
    phi = np.ones(g.num_cells)
    phi[calcite_cells] = solid_porosity

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
tol_discharge = 1e-3 # Absolute tolerance for discharge
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
    "tolerance discharge": tol_discharge,
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
"""
# Postprocess and write monitor results
t_arr = np.array(outer)
u_arr = np.array(res_u)
p_arr = np.array(res_div)
q_arr = np.array(res_q)

worksheet = workbook.add_worksheet()

array = np.array([t_arr, u_arr, p_arr, q_arr])
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()
"""

row = 0
qfworksheet = qfworkbook.add_worksheet()
array = np.array([np.arange(g.num_faces), q])
for col, data in enumerate(array):
    qfworksheet.write_column(row, col, data)
qfworkbook.close()

p = p.reshape((Nx, Ny, 1), order = 'F')
vx = u[0].reshape((Nx, Ny, 1), order = 'F')
vy = u[1].reshape((Nx, Ny, 1), order = 'F')
vz = np.zeros(g.num_cells).reshape((Nx, Ny, 1), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, filename)


