"""

********* Stokes flow around a sphere *********

This example solves one the fundamental benchmark
in low Reynolds-number hydrodynamics: the Stokes solution
for steady flow past a small sphere.

The computational domain is a cube of size (2L)^3
centered at the sphere centre, where L is the distance
of the boundary from the centre of the sphere.

Taking advantage of the symmetry of the problem,
only the upper right quarter of the domain is simulated.

The grid is a structured grid of size Nx, Ny, Nz.
In the future, there is a concrete plan to extend
the formulation to general quadrilateral grids.

***********************************
***  The grid is Non-staggered  ***
***  i.e. all variables (p, u)  ***
*** are defined at cell-centres ***
***********************************

Boundary conditions consist of exact pressure at the outflow,
symmetry at the lower and left faces, and velocity everywhere else.

The numerical flow field (pressure and velocity)
is compared against the anlytical solution.

a: radius of the sphere
N: number of cells along the radius
Tests:
L = {10, 15, 20, 25} * a
N = {2, 4, 8}

********* Notes on the numerical results *********

Microcontinuum
The velocity field is calculated with a global error below 0.4% when L=15
even with the coarser mesh (N=2). This is a very good result.
However Pressure is inaccurate globally (error about 50%).
This means that pressure gradients are correct but pressure values are not.
However, when considering internal cells around the sphere,
the pressure error goes down to less than 10% in proximity of the sphere
(r < 5a), which is within the limit of the engineering good approximation.
Also, this error tends to decrease with increasing mesh resolution,
e.g. the Internal pressure error reduces from 10% (N=2) to 6% (N=4).

NoSlip
L=15, N=2, global velocity error < 0.3%, global pressure error = 12%,
internal pressure error < 8%

This pressure error appears to be an irreducible inconsistency
due to the complexity of the flow field and numerical approximation
of the Stokes equations of flow.
"""

import numpy as np
import xlsxwriter

from geometry import structured
from geometry.bc import BoundaryCondition
from models import flowSolver
from importExport import printf
import time
                   
def cart2cyl(q, th, fi):

    """
    Function to convert the velocity field
    from cartesian to cylindrical coordinates

    Inputs:
    q: velocity field (dimension: number of cells)
    th: the in-plane angle, theta = atan2(y/x)
    fi: the angle from the positive flow axis, fi = arccos(z,r)

    Returns:
    qr: radial component of velocity
    qt: polar component of velocity
    """

    qr = q[0]*np.cos(th)*np.sin(fi)+q[1]*np.sin(th)*np.sin(fi)+q[2]*np.cos(fi)
    qt = q[0]*np.cos(th)*np.cos(fi)+q[1]*np.sin(th)*np.cos(fi)-q[2]*np.sin(fi)

    return qr, qt

def cyl2cart(g, th, fi, vr, vt):

    """
    Function to convert the velocity field
    from cylindrical to cartesian coordinates

    Inputs:
    g: the grid
    th: the in-plane angle, theta = atan2(y/x)
    fi: the angle from the positive flow axis, fi = arccos(z,r)
    vr: radial component of velocity
    vt: polar component of velocity
    
    Returns:
    q: velocity vector field (dimensions: 3 x size of vr)
    """

    q = np.zeros((g.dim, vr.size))

    q[0] = vr*np.cos(th)*np.sin(fi)+vt*np.cos(th)*np.cos(fi)
    q[1] = vr*np.sin(th)*np.sin(fi)+vt*np.sin(th)*np.cos(fi)
    q[2] = vr*np.cos(fi)-vt*np.sin(fi)

    return q

# Output files
filename = 'outputFiles/plots/sphere15aN4.vtk'
workbook = xlsxwriter.Workbook('outputFiles/residuals/sphere15aN4.xlsx')

FlowModel = 'NoSlip'
FlowSolver = 'Simple'

# Geometry
a = 0.001 # radius of the sphere
d2b = 15 # distance to the boundary
N = 4 # number of cells along the radius 
dx = a / N # grid size

Lx = d2b * a
Ly = Lx
Lz = 2 * d2b * a

Nx = round(Lx / dx) # number of elements
Ny = Nx
Nz = round(Lz / dx)
domain = np.array([Lx, Ly, Lz])
basedim = np.array([Nx, Ny, Nz])

# Create grid
g = structured.CartGrid(basedim, domain)
g.compute_geometry()

print('Nx', g.Nx, 'Ny', g.Ny, 'Nz', g.Nz, 'dx', g.dx)
print('dimensions', g.dim, 'Number of cells', g.num_cells)

# Cylindrical coordinates
# from (x, y, z) to (r, phi, theta)
xc = g.cell_centers[0]
yc = g.cell_centers[1]
zc = g.cell_centers[2] - domain[2]/2
r = np.sqrt(xc**2 + yc**2 + zc**2)
phiangle = np.arctan2(yc, xc)
thetaangle = np.arccos(zc / r)

# Boundary conditions
sphere_cells = np.ravel(np.argwhere(r <= a))
   
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))
inflow_faces = np.ravel(np.argwhere(g.face_centers[2] < 1e-10))
outflow_faces = np.ravel(np.argwhere(g.face_centers[2] > domain[2] - 1e-10))

vel_faces = np.concatenate((inflow_faces, right_faces, top_faces))
pres_faces = outflow_faces

flow_faces = np.concatenate((vel_faces, pres_faces))
bnd_cond_flow_faces = ['vel'] * vel_faces.size + ['pres'] * pres_faces.size

symm_faces = np.concatenate((left_faces, bot_faces))
bnd_cond_symm_faces = ['x'] * left_faces.size + ['y'] * bot_faces.size

bnd_wall_cells = []
if FlowModel == 'NoSlip':
    bnd_wall_cells = sphere_cells
    
bound_cond = BoundaryCondition(
    g, flow_faces, bnd_cond_flow_faces, symm_faces, bnd_cond_symm_faces,
    wall_cells = bnd_wall_cells
    )

# Input data
rho = 1000 # fluid density
mu = 0.001 # fluid viscosity
u0 = 0.01 # freestream velocity

# Assign boundary conditions values
p_bound = np.zeros(g.num_faces)
u_bound = np.zeros((g.dim, g.num_faces))

# First, calculate analytical solution on the faces
xf = g.face_centers[0]
yf = g.face_centers[1]
zf = g.face_centers[2] - domain[2]/2
rf = np.sqrt(xf**2 + yf**2 + zf**2)
phif = np.arctan2(yf, xf)
thetaf = np.arccos(zf / rf)

vrf = u0 * np.cos(thetaf) * (1 - 3/2 * a/rf + 1/2 * pow(a/rf, 3))
vtf = - u0 * np.sin(thetaf) * (1 - 3/4 * a/rf - 1/4 * pow(a/rf, 3))

pf = - 3/2 * mu * u0 * np.cos(thetaf) * a / rf**2

# Assign outflow pressure
p_bound[outflow_faces] = pf[outflow_faces]

# Assign boundary velocity
uf = cyl2cart(g, phif, thetaf, vrf, vtf)
u_bound[:, vel_faces] = uf[:, vel_faces]

# Store data for flow in a dictionary
dataFlow = {
    "fluid density": rho,
    "fluid viscosity": mu,
    "boundary velocity": u_bound,
    "boundary pressure": p_bound,
    }

# Initialize porosity field (in a microcontinuum framework)
if FlowModel == 'microcontinuum':
    solid_porosity = 1e-4
    initial_permeability = 1e-11
    phi = np.ones(g.num_cells)
    phi[sphere_cells] = solid_porosity

    dataFlow["porosity field"] = phi
    dataFlow["initial permeability"] = initial_permeability
    
# Linear Solver data
omega_u = 0.4 # under-relaxation velocity
omega_p = 0.08 # under-relaxation pressure correction
iter_u = 35 # inner iterations velocity
tol_u = 0.25 # inner tolerance velocity
tol_p = 0.05 # inner tolerance pressure correction
convergence_criterion = 1e-4 # establish when flow solution has converged (for SIMPLE)
tol_continuity = 1e-10 # Absolute tolerance for continuity imbalance
tol_discharge = 1e-1 # Absolute tolerance for discharge
outer_iterations = 2000 # max number of iterations
tol_steady_state = 1e-3 # establish when flow solution has reached steady-state (for PISO)

cflmax = 1
dt = cflmax * g.dx / u0 # time step (for PISO)

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

# Initialize variables
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

# Convert the numerical velocity from cartesian to cylindrical coordinates
vr, vt = cart2cyl(u, phiangle, thetaangle)

# Calculate analytical solution
vr_ex = u0 * np.cos(thetaangle) * (1 - 3/2 * a/r + 1/2 * pow(a/r, 3))
vt_ex = - u0 * np.sin(thetaangle) * (1 - 3/4 * a/r - 1/4 * pow(a/r, 3))
p_ex = - 3/2 * mu * u0 * np.cos(thetaangle) * a / r**2
p_ex[sphere_cells] = 0
vr_ex[sphere_cells] = 0
vt_ex[sphere_cells] = 0

# Calculate global errors
print('GLOBAL ERRORS')
vr_diff = vr_ex - vr
vt_diff = vt_ex - vt
p_diff = p_ex - p
er=np.sqrt(np.sum(g.cell_volumes * vr_diff**2))/np.sqrt(np.sum(g.cell_volumes * vr_ex**2))
print('Radial velocity error', er)
et=np.sqrt(np.sum(g.cell_volumes * vt_diff**2))/np.sqrt(np.sum(g.cell_volumes * vt_ex**2))
print('Polar velocity error', et)

ep=np.sqrt(np.sum(g.cell_volumes * p_diff**2))/np.sqrt(np.sum(g.cell_volumes * p_ex**2))
print('Pressure error', ep)

# Internal cells
print('INTERNAL ERRORS')
epList = []
evrList = []
evtList = []
for j in range (2, d2b+1):
    internal_cells = np.logical_and(r > a, r < j*a)
    p_diff_int = p_ex[internal_cells] - p[internal_cells]
    vr_diff = vr_ex[internal_cells] - vr[internal_cells]
    vt_diff = vt_ex[internal_cells] - vt[internal_cells]
    ep=np.sqrt(
        np.sum(g.cell_volumes[internal_cells] * p_diff_int**2)
        ) / np.sqrt(
            np.sum(g.cell_volumes[internal_cells] * p_ex[internal_cells]**2)
            )
    er=np.sqrt(
        np.sum(g.cell_volumes[internal_cells] * vr_diff**2)
        ) / np.sqrt(
            np.sum(g.cell_volumes[internal_cells] * vr_ex[internal_cells]**2)
            )
    et=np.sqrt(
        np.sum(g.cell_volumes[internal_cells] * vt_diff**2)
        ) / np.sqrt(
            np.sum(g.cell_volumes[internal_cells] * vt_ex[internal_cells]**2)
            )
    epList.append(ep)
    evrList.append(er)
    evtList.append(et)

evrInternal = np.array(evrList)
evtInternal = np.array(evtList)
epInternal = np.array(epList)
print('Internal radial velocity error', evrInternal)
print('Internal polar velocity error', evtInternal)
print('Internal pressure error', epInternal)

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

# Export flow field
p = p.reshape((Nx, Ny, Nz), order = 'F')
p_ex = p_ex.reshape((Nx, Ny, Nz), order = 'F')
vx = u[0].reshape((Nx, Ny, Nz), order = 'F')
vy = u[1].reshape((Nx, Ny, Nz), order = 'F')
vz = u[2].reshape((Nx, Ny, Nz), order = 'F')
printf.write_outFlow(g, p, vx, vy, vz, filename)
