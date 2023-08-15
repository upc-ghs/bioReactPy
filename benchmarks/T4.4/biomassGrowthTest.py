"""

********* Biomass growth Benchmark *********

This example solves biomass growth at the pore-scale.

This is the runscript to obtain the results presented in Section 4.4 of

Starnoni (2023)
bioReactPy
(Submitted)

"""

import numpy as np
import xlsxwriter
import math

from geometry import structured
from geometry.bc import BoundaryCondition, BoundaryConditionTransport
from models import biomassGrowth
from importExport import readf, printf
import time

thresh = 1e-16

# Output files
workbook = xlsxwriter.Workbook('outputFiles/monitors/bioGrowthB1t400.xlsx')

cB0 = 0.01 #mg/l

t_end = 500

Lx = 500 #cm
Ly = 20 #cm
mesh_size = 0.5
Nx = round(Lx / mesh_size)
Ny = round(Ly / mesh_size)
domain = np.array([Lx, Ly])
basedim = np.array([Nx, Ny])

# Create grid
g = structured.CartGrid(basedim, domain)
g.compute_geometry()

print('Nx', g.Nx, 'Ny', g.Ny, 'dx', g.dx)
print('Number of cells', g.num_cells)
print('Initial biomass', cB0)

# BC
aTop = 1.25*(Ly/2)
aLow = 0.75*(Ly/2)

# Boundary conditions 
left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))

ed_faces = np.ravel(np.argwhere(
    (g.face_centers[0] < 1e-10) &
    (g.face_centers[1] > aLow) & (g.face_centers[1] < aTop)
    ))
ea_facesTop = np.ravel(np.argwhere(
    (g.face_centers[0] < 1e-10) &
    (g.face_centers[1] > aTop) & (g.face_centers[1] < domain[1] - 1e-10)
    ))
ea_facesLow = np.ravel(np.argwhere(
    (g.face_centers[0] < 1e-10) &
    (g.face_centers[1] < aLow) & (g.face_centers[1] > - 1e-10)
    ))

ea_faces = np.concatenate((ea_facesLow, ea_facesTop))

# BC for transport
bnd_cond_faces = ['dir'] * left_faces.size

bc = BoundaryConditionTransport(
    g, left_faces, bnd_cond_faces
    )

# Input data
u0 = 100 # cm/d
mu_max = 1 #d^(-1)
diff = 2.5 # cm2/d
_lambda = 0.1 #d^(-1)

Ked = 8.33e-2 #mmol/l
Kea = 3.13e-2 #mmol/l

Ced_in = 0.33 #mmol/l
Cea_in = 0.25 #mmol/l

Y = 1 #mg/mmol

# Boundary conditions
ed_bound = np.zeros(g.num_faces)
ea_bound = np.zeros(g.num_faces)
bio_bound = np.zeros(g.num_faces)

ed_bound[ed_faces] = Ced_in
ea_bound[ea_faces] = Cea_in

# Store data for flow in a dictionary
dataFlow = {
    "boundary concentration ED": ed_bound,
    "boundary concentration EA": ea_bound,
    "boundary concentration biomass": bio_bound,
    "diffusion coefficient": diff,
    "velocity": u0,
    "mu max": mu_max,
    "lambda": _lambda,
    "Ked": Ked,
    "Kea": Kea,
    "yield coefficient": Y,
    "end time": t_end,
    "initial biomass": cB0,
    }

dt = 0.25 * g.dx / u0
dt_shao = 2/24/60 #2 min
print('time step', dt)
print('time step Shao', dt_shao)
dataFlow["time step"] = dt

# Solve Coupled Flow & Reactive Transport
start = time.time()
c, l = biomassGrowth.solveRT(
    g, dataFlow, bc
    )
end = time.time()
elapsed_time = end - start
hours, mod = divmod(elapsed_time, 3600)
mins, secs = divmod(mod, 60)
print(
    'elapsed time in hours', int(hours), 'h', int(mins), 'm', int(secs), 's'
    ) 

#Postprocess
ced = c["concentration ED"]
cea = c["concentration EA"]
prd = c["concentration product"]
bio = c["concentration biomass"]

faces = np.ravel(np.argwhere(
    (g.face_centers[0] < 200 + 1e-10) & (g.face_centers[0] > 200 - 1e-10)
    ))
ind_cells=g.cell_face_as_dense()[:,faces]
assert faces.size == Ny
yc = np.zeros(Ny)
cd = np.zeros(Ny)
ca = np.zeros(Ny)
cp = np.zeros(Ny)
cb = np.zeros(Ny)
for i in np.arange(Ny):
    f = faces[i]
    yc[i] = g.face_centers[1, f]
    cd[i] = 0.5 * (ced[ind_cells[0,i]]+ced[ind_cells[1,i]])
    ca[i] = 0.5 * (cea[ind_cells[0,i]]+cea[ind_cells[1,i]])
    cp[i] = 0.5 * (prd[ind_cells[0,i]]+prd[ind_cells[1,i]])
    cb[i] = 0.5 * (bio[ind_cells[0,i]]+bio[ind_cells[1,i]])

# 1 Unpack and print monitor quantities
t = l["time"]
ced = l["average ED"]
cea = l["average EA"]
prd = l["average product"]
bio = l["average biomass"]

t_arr = np.array(t)
d_arr = np.array(ced)
a_arr = np.array(cea)
p_arr = np.array(prd)
b_arr = np.array(bio)

worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0

array = np.array(
    [t_arr, d_arr, a_arr, p_arr, b_arr, yc, cd, ca, cp, cb], dtype=object
    )

# Iterate over the data and write it out row by row.

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()
    
