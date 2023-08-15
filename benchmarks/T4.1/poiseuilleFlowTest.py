"""

********* Poiseuille flow in a circular tube *********

This example solves one the fundamental benchmark
in low Reynolds-number hydrodynamics: the Poiseuille solution
for steady flow in a circular tube.

The computational domain has size (2R, 2R, L)
centered at the tube centre in the (x,y) plane,
where R is the radius of the tube and L its length.

Taking advantage of the axis-symmetry of the problem,
only the upper-right quarter of the domain is simulated.

The grid is a structured grid of size Nx, Ny, Nz.
In the future, there is a concrete plan to extend
the formulation to general quadrilateral grids.

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

Boundary conditions consist of Pressure at the inflow and outflow,
no-slip conditions at the tube solid boundaries, and symmetry
at the bottom and left faces.
Alternatively, velocity at the inflow boundary can also be tested.

The numerical flow field (pressure and velocity)
is compared against the anlytical solution.

R: radius of the sphere
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
import math

# Output files
filename = 'outputFiles/plots/poiseuilleN32.vtk'
workbook = xlsxwriter.Workbook('outputFiles/residuals/poiseuilleN32.xlsx')

FlowModel = 'NoSlip'
FlowSolver = 'Simple'

# Geometry                  
L = 0.001
R = 0.0005
N = np.array([2, 4, 8, 16, 32])

listp = []
listv = []

for j in N:
    dx = R / j
    Lx = R
    Ly = Lx
    Nx = int(Lx / dx)
    Ny = Nx
    Nz = int(L / dx)

    domain = np.array([Lx, Ly, L])
    basedim = np.array([Nx, Ny, Nz])

    # Create grid
    g = structured.CartGrid(basedim, domain)
    g.compute_geometry()

    print('Nx', g.Nx, 'Ny', g.Ny, 'Nz', g.Nz, 'dx', g.dx)
    print('dimensions', g.dim, 'Number of cells', g.num_cells)

    # Boundary conditions
    r = np.sqrt(g.cell_centers[0]**2 + g.cell_centers[1]**2)
    solid_cells = np.ravel(np.argwhere(r > R))

    left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
    right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
    bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
    top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))
    inflow_faces = np.ravel(np.argwhere(g.face_centers[2] < 1e-10))
    outflow_faces = np.ravel(np.argwhere(g.face_centers[2] > domain[2] - 1e-10))

    flow_faces = np.concatenate((inflow_faces, outflow_faces))
    wall_faces = np.concatenate((right_faces, top_faces))
    symm_faces = np.concatenate((left_faces, bot_faces))

    bnd_faces = np.concatenate((flow_faces, wall_faces))
    bnd_cond_faces = ['pres'] * inflow_faces.size + ['pres'] * outflow_faces.size + ['wall'] * wall_faces.size
    bnd_cond_symm_faces = ['x'] * left_faces.size + ['y'] * bot_faces.size

    bnd_wall_cells = []
    if FlowModel == 'NoSlip':
        bnd_wall_cells = solid_cells

    bound_cond = BoundaryCondition(
        g, bnd_faces, bnd_cond_faces, symm_faces, bnd_cond_symm_faces,
        wall_cells = bnd_wall_cells
        )

    # Input data
    rho = 1000
    mu = 0.001
    p_left = 2
    p_right = 1
    m = (p_right - p_left) / L

    # Assign boundary conditions values
    p_bound = np.zeros(g.num_faces)
    u_bound = np.zeros((g.dim, g.num_faces))

    p_bound[outflow_faces] = p_right
    p_bound[inflow_faces] = p_left

    xf = g.face_centers[0, inflow_faces]
    yf = g.face_centers[1, inflow_faces]
    rf = np.sqrt(xf**2 + yf**2)
    uf = -m / (4*mu) * (R**2 - rf**2)
    #u_bound[2, inflow_faces] = uf.clip(0)

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
        
    # Linear Solver data
    FlowSolver = 'Simple' # choose flow solver algorithm (currently, 'Simple' or 'Piso')
    omega_u = 0.5 # under-relaxation velocity
    if FlowSolver == 'Piso':
        omega_u = 1    
    omega_p = 0.1 # under-relaxation pressure correction
    iter_u = 25 # inner iterations velocity
    tol_u = 0.2 # inner tolerance velocity
    tol_p = 1e-2 # inner tolerance pressure correction
    convergence_criterion = 1e-4 # establish when flow solution has converged (for SIMPLE)
    tol_continuity = 1e-10 # Absolute tolerance for continuity imbalance
    tol_discharge = 1e-5 # Absolute tolerance for discharge
    outer_iterations = 2000 # max number of iterations
    tol_steady_state = 1e-4 # establish when flow solution has reached steady-state (for PISO)

    cflmax = 0.5
    umax = -m / (4*mu) * R**2
    dt = cflmax * g.dx / umax

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

    start = time.time()
    p, u, q, outer, res_u, res_div, res_q = flowSolver.StokesSolver(
        g, dataFlow, bound_cond, dataSolver, p_guess, u_guess, return_residuals
        )
    end = time.time()
    print('elapsed time', end - start)

    # Analytical solution
    u_ex = -m / (4*mu) * (R**2 - r**2)
    u_ex[solid_cells] = 0
    p_ex = p_left + m * g.cell_centers[2]
    p_ex[solid_cells] = 0
    q_ex = -m * math.pi * pow(R,4) / (8*mu)
    print('exact discharge', q_ex/4)

    u_diff=u_ex-u[2]
    eu=np.sqrt(np.sum(g.cell_volumes * u_diff**2))/np.sqrt(np.sum(g.cell_volumes * u_ex**2))
    print('velocity error', eu)

    p_diff=p_ex-p
    ep=np.sqrt(np.sum(g.cell_volumes * p_diff**2))/np.sqrt(np.sum(g.cell_volumes * p_ex**2))
    print('pressure error', ep)

    # Internal cells
    internal_cells = np.ravel(np.argwhere(r < R))

    u_diff_int = u_ex[internal_cells] - u[2, internal_cells]
    eu=np.sqrt(
        np.sum(g.cell_volumes[internal_cells] * u_diff_int**2)
        ) / np.sqrt(
            np.sum(g.cell_volumes[internal_cells] * u_ex[internal_cells]**2)
            )
    print('Internal velocity error', eu)

    p_diff_int = p_ex[internal_cells] - p[internal_cells]
    ep=np.sqrt(
        np.sum(g.cell_volumes[internal_cells] * p_diff_int**2)
        ) / np.sqrt(
            np.sum(g.cell_volumes[internal_cells] * p_ex[internal_cells]**2)
            )
    print('Internal pressure error', ep)

    listp.append(ep)
    listv.append(eu)

print('listp', listp)
print('listv', listv)

"""
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
"""

# Export flow field
p = p.reshape((Nx, Ny, Nz), order = 'F')
vx = u[0].reshape((Nx, Ny, Nz), order = 'F')
vy = u[1].reshape((Nx, Ny, Nz), order = 'F')
vz = u[2].reshape((Nx, Ny, Nz), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, filename)


