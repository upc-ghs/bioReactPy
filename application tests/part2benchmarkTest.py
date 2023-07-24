"""

********* Molins dissolution benchmark *********

This example solves the time evolution of flow and concentration fields
around one disolving mineral grain using the microcontinuum approach.

This is the runscript to obtain the results presented in Section 3 of

Starnoni and Sanchez-Vila (2023)
Pore-scale modelling of microbially enhanced carbon mineralization
(submitted)

For geometry and flow, see molinsBenchmarkFlow.py

Numerical setup taken from Part II of the benchmark paper

Molins et al. (2021). Simulation of mineral dissolution at the pore scale
with evolving fluid-solid interfaces: Review of approaches and benchmark
problem set. Computational Geosciences, 25, 1285-1318.

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

A single acid species is injected at the left boundary, and a single
dissolution reaction is considered.
"""

import numpy as np
import xlsxwriter

from geometry import structured
from geometry.bc import BoundaryCondition, BoundaryConditionTransport
from models import molinsBenchmark
from importExport import printf
import time
import math
            
# Input Files
flowfile = 'outputFiles/plots/molinsBenchmarkN8.vtk'
fluxfile = 'outputFiles/fluxes/molinsBenchmarkN8.xls'

# Output files
outflowfile = 'outputFiles/plots/part2benchmarkN8.vtk'
cfile = 'outputFiles/concentrations/part2benchmarkN8.vtk'
workbook = xlsxwriter.Workbook('outputFiles/monitors/part2benchmarkN8.xlsx')
evolutionFile = 'outputFiles/plots/part2benchmarkN8Time{0}.vtk'

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

# Boundary conditions 
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))

xc = g.cell_centers[0]-domain[0]/2
yc = g.cell_centers[1]-domain[1]/2
r = np.sqrt(xc**2 + yc**2)
calcite_cells = np.ravel(np.argwhere(r <= a))

# BC for flow
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
rhof = 1
ni = 1e-2
mu = rhof * ni
u_in = 0.12
p_out = 0
solid_porosity = 1e-2
initial_permeability = 1e-11

diff = 1e-5
c_in = 1e-5
kappa = math.pow(10, -4.05)
gamma = 1000
Vm = 36.9
xi = -1

UpdateFlow = 0.01

t_end = 600
t_print = 600

# Assign boundary conditions values
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))
c_bound = np.zeros(g.num_faces)

p_bound[right_faces] = p_out
u_bound[0, left_faces] = u_in
c_bound[left_faces] = c_in

# Initialize porosity field
phi = np.ones(g.num_cells)
phi[calcite_cells] = solid_porosity

# Store data for flow in a dictionary
dataFlow = {
    "fluid density": rhof,
    "fluid viscosity": mu,
    "inflow concentration": c_in,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    "boundary concentration": c_bound,
    "initial porosity field": phi,
    "initial permeability": initial_permeability,
    "molar volume": Vm,
    "stoichiometric coefficient": xi,
    "diffusion coefficient": diff,
    "rate constant": kappa * gamma,
    "end time": t_end,
    "flow filename": flowfile,
    "flux filename": fluxfile,
    "evolution filename": evolutionFile,
    "update flow criterion": UpdateFlow,
    "print geometry": t_print,
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
tol_trasport = 1e-5
if g.num_cells < 1000:
    tol_trasport = 1e-4

cfl = 1
u_max = 0.295
dt = cfl * g.dx / u_max
print('umax-based time step', dt)
dt = cfl * g.dx / u_in
print('u_in-based time step', dt)

# default minimum time step for N = 32
dt = 0.001
if N == 16:
    dt = 0.002
if N == 8:
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
    "tolerance steady-state": tol_steady_state,
    "tolerance bicg transport": tol_trasport,
    "time step": dt,
    "Flow model": FlowModel,
    }

# Solve Coupled Flow & Reactive Transport
start = time.time()
p, u, c, phi, t, R, S, V = molinsBenchmark.solveRT(
    g, dataFlow, bc_flow, bc_transport, dataSolver
    )
end = time.time()
elapsed_time = end - start
hours, mod = divmod(elapsed_time, 3600)
mins, secs = divmod(mod, 60)
print(
    'elapsed time in hours', int(hours), 'h', int(mins), 'm', int(secs), 's'
    ) 

#Postprocess
t_arr = np.array(t)
r_arr = np.array(R)
s_arr = np.array(S)
v_arr = np.array(V)

worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0

array = np.array([t_arr, r_arr, s_arr, v_arr])
# Iterate over the data and write it out row by row.

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

p = p.reshape((Nx, Ny, 1), order = 'F')
vx = u[0].reshape((Nx, Ny, 1), order = 'F')
vy = u[1].reshape((Nx, Ny, 1), order = 'F')
vz = np.zeros(g.num_cells).reshape((Nx, Ny, 1), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, outflowfile)

c_dict = {
    "c": c.reshape((g.Nx, g.Ny, 1), order = 'F'),
    }
printf.write_outFile(g, c_dict, cfile)

