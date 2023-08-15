"""

********* Steady flow for Xie benchmark *********

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
filename = 'xie.vtk'
#workbook = xlsxwriter.Workbook('outputFiles/residuals/xie.xlsx')
qfworkbook = xlsxwriter.Workbook('xie.xlsx')

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

#Boundary conditions 
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > L - 1e-10))

bnd_faces = np.concatenate((left_faces, right_faces))

bnd_cond_faces = ['pres'] * bnd_faces.size

bound_cond = BoundaryCondition(
    g, bnd_faces, bnd_cond_faces
    )

#Input data
rho = 1000 # kg/m3
mu = 1e-3 # Pa s
p_in = 70
p_out = 0

# Assign boundary conditions values
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

p_bound[right_faces] = p_out
p_bound[left_faces] = p_in

# Store flow data in a dictionary
dataFlow = {
    "fluid density": rho,
    "fluid viscosity": mu,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    "kozeny-carman": 'xie',
    }

# Initialize porosity field (in a microcontinuum framework)
solid_porosity = 0.35
initial_permeability = 1.186e-11 #[m2]
phi = solid_porosity * np.ones(g.num_cells)

dataFlow["initial porosity field"] = phi
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
            
qmax = np.amax(q)
print('max velocity', qmax)

_1d = 24 * 60 * 60
print('outflux in m3/day', q[g.num_faces-1] *_1d)

"""
# Postprocess and write monitor results
t_arr = np.array(outer)
u_arr = np.array(res_u)
div_arr = np.array(res_div)
q_arr = np.array(res_q)

worksheet = workbook.add_worksheet()

array = np.array([t_arr, u_arr, div_arr, q_arr])
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()
"""

qfworksheet = qfworkbook.add_worksheet()

row = 0

array = np.array([np.arange(g.num_faces), q])
for col, data in enumerate(array):
    qfworksheet.write_column(row, col, data)
qfworkbook.close()

p = p.reshape((N, 1, 1), order = 'F')
vx = u[0].reshape((N, 1, 1), order = 'F')
vy = np.zeros(g.num_cells).reshape((N, 1, 1), order = 'F')
vz = np.zeros(g.num_cells).reshape((N, 1, 1), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, filename)


