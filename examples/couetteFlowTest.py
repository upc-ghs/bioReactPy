"""

********* Couette flow between parallel plates *********

This example solves one the fundamental benchmark
in low Reynolds-number hydrodynamics: the Couette solution
for steady flow between parallel plates.

The grid is a structured grid of size Nx, Ny.
In the future, there is a concrete plan to extend
the formulation to general quadrilateral grids.

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

Boundary conditions consist of Pressure at the inflow and outflow,
and no-slip conditions at the tube solid boundaries
Alternatively, velocity at the inflow boundary can also be tested.

The numerical flow field (pressure and velocity)
is compared against the anlytical solution.
"""

import numpy as np
import xlsxwriter

from geometry import structured
from geometry.bc import BoundaryCondition
from models import flowSolver
from utils import fvutils
from importExport import printf
import time

# Output files
filename = 'outputFiles/plots/couetteN16.vtk'
workbook = xlsxwriter.Workbook('outputFiles/residuals/couetteN16.xlsx')

FlowModel = 'microcontinuum'
FlowSolver = 'Simple'

#Geometry               
Lx = 0.1
Ly = 0.01
N = 16
dx = Ly / N
Nx = round(Lx / dx)
Ny = round(Ly / dx)
domain = np.array([Lx, Ly])
basedim = np.array([Nx, Ny])

g = structured.CartGrid(basedim, domain)
g.compute_geometry()

print('Nx', g.Nx, 'Ny', g.Ny, 'dx', g.dx)
print('dimensions', g.dim, 'Number of cells', g.num_cells)

#Boundary conditions
solid_cells = []
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))

flow_faces = np.concatenate((left_faces, right_faces))
wall_faces = np.concatenate((bot_faces, top_faces))

bound_faces = np.concatenate((flow_faces, wall_faces))
bound_cond_faces = ['pres'] * left_faces.size + ['pres'] * right_faces.size
bound_cond_faces += ['wall'] * wall_faces.size

bound_cond = BoundaryCondition(g, bound_faces, bound_cond_faces)

#Input data
rho = 1000
mu = 0.001
p_left = 2
p_right = 1
m = (p_right - p_left) / Lx
d = Ly

p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))
p_bound[right_faces] = p_right
p_bound[left_faces] = p_left

yf = g.face_centers[1]
#u_bound[0, left_faces] = - m / (2*mu) * yf[left_faces] * (Ly-yf[left_faces])

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
    phi[solid_cells] = solid_porosity

    dataFlow["porosity field"] = phi
    dataFlow["initial permeability"] = initial_permeability
    
#SOR data
omega_u = 0.5
omega_p = 0.1
iter_u = 25
tol_u = 0.2
tol_p = 1e-2 # inner tolerance pressure correction
convergence_criterion = 1e-4 # establish when flow solution has converged (for SIMPLE)
tol_continuity = 1e-10 # Absolute tolerance for continuity imbalance
tol_discharge = 1e-5 # Absolute tolerance for discharge
outer_iterations = 2000 # max number of iterations
tol_steady_state = 1e-4 # establish when flow solution has reached steady-state (for PISO)

cflmax = 1
dt = cflmax * g.dx / 1.25e-2

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
    "time step": dt,
    "Flow solver": FlowSolver,
    "Flow model": FlowModel,
    }

#Initialize variables
p_guess = np.zeros(g.num_cells)
u_guess = np.zeros((g.dim, g.num_cells))

# Solve Flow
return_residuals = True

#Solve Flow
start = time.time()
p, u, q, outer, res_u, res_div, res_q = flowSolver.StokesSolver(
    g, dataFlow, bound_cond, dataSolver, p_guess, u_guess, return_residuals
    )
end = time.time()
print('elapsed time', end - start)

# Analytical solution
yc = g.cell_centers[1]
u_ex = - m / (2*mu) * yc * (Ly-yc)
p_ex = p_left + m * g.cell_centers[0]

u_diff=u_ex-u[0]
eu=np.sqrt(np.sum(g.cell_volumes * u_diff**2))/np.sqrt(np.sum(g.cell_volumes * u_ex**2))
print('velocity error', eu)

p_diff=p_ex-p
ep=np.sqrt(np.sum(g.cell_volumes * p_diff**2))/np.sqrt(np.sum(g.cell_volumes * p_ex**2))
print('pressure error', ep)
    
# Postprocess and write monitor results
t_arr = np.array(outer)
u_arr = np.array(res_u)
p_arr = np.array(res_div)
q_arr = np.array(res_q)

worksheet = workbook.add_worksheet()

row = 0
array = np.array([t_arr, u_arr, p_arr, q_arr])
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

p = p.reshape((Nx, Ny, 1), order = 'F')
vx = u[0].reshape((Nx, Ny, 1), order = 'F')
vy = u[1].reshape((Nx, Ny, 1), order = 'F')
vz = np.zeros(g.num_cells).reshape((Nx, Ny, 1), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, filename)
