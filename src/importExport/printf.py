"""
Various functions used to print output data
"""

import numpy as np
import xlrd

thresh = 1e-16
eps = 1e-4
                
def write_outFlow(g, p, u, v, w, filename):
    
    """
    Function to print the flow field into a
    vtk legacy file for vizualization in Paraview

    Inputs:
    g: the grid
    p: the scalar pressure field
    u, v, w: components of the velocity vector field
    filename: path to the vtk file
    """

    #default 1D geometry
    Nx = g.Nx
    Ny = 1
    Nz = 1
    if g.dim > 1:
        Ny = g.Ny
    if g.dim > 2:
        Nz = g.Nz

    with open(filename, 'w') as outfile:
        outfile.write('# vtk DataFile Version 2.0\n')
        outfile.write('flow field\n')
        outfile.write('ASCII\n')
        outfile.write('DATASET STRUCTURED_POINTS\n')
        outfile.write('DIMENSIONS ')
        outfile.write('{}'.format(Nx))
        outfile.write(' ')
        outfile.write('{}'.format(Ny))
        outfile.write(' ')
        outfile.write('{}\n'.format(Nz))
        outfile.write('ASPECT_RATIO 1 1 1\n')
        outfile.write('ORIGIN 0 0 0\n')
        outfile.write('POINT_DATA ')
        outfile.write('{:03d}\n'.format(g.num_cells))
        outfile.write('SCALARS pressure float 1\n')
        outfile.write('LOOKUP_TABLE default\n')
        for k in np.arange(Nz):
            for j in np.arange(Ny):
                for i in np.arange(Nx):
                    outfile.write(f'{float(p[i,j,k]):.5e}\n')
        outfile.write('VECTORS velocity float\n')
        for k in np.arange(Nz):
            for j in np.arange(Ny):
                for i in np.arange(Nx):
                    outfile.write(f'{float(u[i,j,k]):.5e}')
                    outfile.write(' ')
                    outfile.write(f'{float(v[i,j,k]):.5e}')
                    outfile.write(' ')
                    outfile.write(f'{float(w[i,j,k]):.5e}\n')

def write_outFile(g, c_dict, filename):
    
    """
    Function to print multiple scalar fields into a
    vtk legacy file for vizualization in Paraview

    Inputs:
    g: the grid
    c_dict: dictionary containing the fields to write
    filename: path to the vtk file
    """

    #default 1D geometry
    Nx = g.Nx
    Ny = 1
    Nz = 1
    if g.dim > 1:
        Ny = g.Ny
    if g.dim > 2:
        Nz = g.Nz
        
    with open(filename, 'w') as outfile:
        outfile.write('# vtk DataFile Version 2.0\n')
        outfile.write('flow field\n')
        outfile.write('ASCII\n')
        outfile.write('DATASET STRUCTURED_POINTS\n')
        outfile.write('DIMENSIONS ')
        outfile.write('{}'.format(Nx))
        outfile.write(' ')
        outfile.write('{}'.format(Ny))
        outfile.write(' ')
        outfile.write('{}\n'.format(Nz))
        outfile.write('ASPECT_RATIO 1 1 1\n')
        outfile.write('ORIGIN 0 0 0\n')
        outfile.write('POINT_DATA ')
        outfile.write('{:03d}\n'.format(g.num_cells))
        for key in c_dict:
            c = c_dict[key]
            outfile.write('SCALARS ')
            outfile.write('{}'.format(key))
            outfile.write(' float 1\n')
            outfile.write('LOOKUP_TABLE default\n')
            for k in np.arange(Nz):
                for j in np.arange(Ny):
                    for i in np.arange(Nx):
                        outfile.write(f'{float(c[i,j,0]):.5e}\n')
                

