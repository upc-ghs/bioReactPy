"""
Various functions used to read input data
"""

import numpy as np
import xlrd

thresh = 1e-16
eps = 1e-4
               
def read_flowfield(g, filename):

    """
    Function to read the flow field from a
    vtk legacy file for inizialitation of variables

    Inputs:
    g: the grid
    filename: path to the vtk file

    Returns:
    p: np.array(g.num_cells) the scalar pressure field 
    u: np.ndarray(g.dim, g.num_cells) the velocity vector
    """
    
    p = np.zeros(g.num_cells)
    u = np.zeros((g.dim, g.num_cells))

    a1=[]
    a2=[]
    a3=[]

    with open(filename) as fp:
        for _ in range(10):
            next(fp)
        for j in np.arange(g.num_cells):
            line = fp.readline()
            p[j] = float(line.strip('\n'))
        next(fp)
        for j in np.arange(g.num_cells):
            line = fp.readline()
            data = line.split()
            a1.append(float(data[0]))
            a2.append(float(data[1]))
            a3.append(float(data[2]))

    u[0]=np.array(a1)
    if g.dim > 1:
        u[1]=np.array(a2)
    if g.dim > 2:
        u[2]=np.array(a3)

    return p, u

def read_fluxfield(g, filename):

    """
    Function to read the flux field from a
    vtk legacy file for inizialitation of the advection term

    Inputs:
    g: the grid
    filename: path to the vtk file

    Returns:
    q: np.array(g.num_faces) the flux field on the faces 
    """
    
    q = np.zeros(g.num_faces)

    workbook = xlrd.open_workbook(filename)

    sheet = workbook.sheet_by_name('Sheet1')

    i = 0

    for value in sheet.col_values(1):
        q[i] = value
        i += 1


    return q
