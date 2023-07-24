""" Module containing class for boundary conditions.

Acknowledgements:

    This implementation is largely inspired from PorePy,
    an open-source simulation tool for fractured and deformable
    porous media developed by the University of Bergen,
    see https://github.com/pmgbergen/porepy
    which is released under the terms of the GNU General Public License  
"""

import numpy as np
import scipy.sparse as sps

class BoundaryCondition(object):

    """ Class to store information on boundary conditions for flow.

    The BCs are specified by face number, and can have type Velocity, Pressure,
    Symmetry or Wall.

    Attributes:
        num_faces (int): Number of faces in the grid
        dim (int): Dimension of the boundary. One less than the dimension of
            the grid.
        is_vel (np.ndarray boolean, size g.num_faces): Element i is true if
            face i has been assigned a Velocity condition. Tacitly assumes that
            the face is on the boundary.
        is_pres (np.ndarary, boolean, size g.num_faces): Element i is true if
            face i has been assigned a Pressure condition.
        is_wall (np.ndarray, boolean, size g.num_faces): Element i is true if
            face i has been assigned a wall condition.
        is_sym (np.ndarray, boolean, size g.num_faces): Element i is true if
            face i has been assigned a symmetry condition.
        is_sym_x, is_sym_y, is_sym_z (np.ndarray, boolean, size g.num_faces):
            Element i is true if face i has been assigned a symmetry condition
            in the x, y, z direction, respectively.
    """

    def __init__(
        self, g, faces=None, cond=None, faces_sym=None, cond_sym=None,
        wall_cells = None,
        ):
        """Constructor for BoundaryCondition.

        The conditions are specified by face numbers. Faces that do not get an
        explicit condition will have a pressure condition assigned.

        Parameters:
            g (grid): For which boundary conditions are set.
            faces (np.ndarray): Faces for which conditions are assigned.
            cond (list of str): Conditions on the faces, in the same order as
                used in faces. Should be as long as faces. The list elements
                should be one of "vel", "pres", "wall".
            faces_sym (np.ndarray): Faces for which symmetry conditions are assigned.
            cond_sym (list of str): Conditions on the symmetry faces,
                in the same order as used in faces_sym. Should be as long as faces_sym.
                The list elements should be one of "x", "y", "z".

            Optional for the No-Slip implementation
            wall_cells (np.ndarray): Cells which are Solid 
        """

        self.num_faces = g.num_faces
        self.num_cells = g.num_cells
        self.dim = g.dim - 1

        self.list_wall = wall_cells
        if wall_cells is None:
            self.list_wall = []

        # Find boundary faces
        bf = g.get_boundary_faces()

        self.is_vel = np.zeros(self.num_faces, dtype=bool)
        self.is_pres = np.zeros(self.num_faces, dtype=bool)
        self.is_wall = np.zeros(self.num_faces, dtype=bool)

        self.is_sym = np.zeros(self.num_faces, dtype=bool)
        self.is_sym_x = np.zeros(self.num_faces, dtype=bool)
        self.is_sym_y = np.zeros(self.num_faces, dtype=bool)
        self.is_sym_z = np.zeros(self.num_faces, dtype=bool)

        # By default, all boundary faces are pres - zero gradient
        self.is_pres[bf] = True
        
        if faces is not None:
            # Validate arguments
            assert cond is not None
            if faces.dtype == bool:
                if faces.size != self.num_faces:
                    raise ValueError(
                        """When giving logical faces, the size of
                                        array must match number of faces"""
                    )
                faces = np.argwhere(faces)
            if not np.all(np.in1d(faces, bf)):
                raise ValueError(
                    "Give boundary condition only on the \
                                 boundary"
                )

            if isinstance(cond, str):
                cond = [cond] * faces.size
            if faces.size != len(cond):
                raise ValueError("One BC per face")

            for l in np.arange(faces.size):
                s = cond[l]
                if s.lower() == "pres":
                    pass  # Neumann is already default
                else:
                    if s.lower() == "vel":
                        self.is_vel[faces[l]] = True
                        self.is_pres[faces[l]] = False
                        self.is_sym[faces[l]] = False
                        self.is_sym_x[faces[l]] = False
                        self.is_sym_y[faces[l]] = False
                        self.is_sym_z[faces[l]] = False
                        self.is_wall[faces[l]] = False
                    elif s.lower() == "wall":
                        self.is_vel[faces[l]] = False
                        self.is_pres[faces[l]] = False
                        self.is_sym[faces[l]] = False
                        self.is_sym_x[faces[l]] = False
                        self.is_sym_y[faces[l]] = False
                        self.is_sym_y[faces[l]] = False
                        self.is_wall[faces[l]] = True
                    else:
                        raise ValueError("error other boundary condition")                        

        if faces_sym is not None:
            for l in np.arange(faces_sym.size):
                s = cond_sym[l]
                self.is_sym[faces_sym[l]] = True
                self.is_vel[faces_sym[l]] = False
                self.is_pres[faces_sym[l]] = False
                self.is_wall[faces_sym[l]] = False
                if s.lower() == "x":
                    self.is_sym_x[faces_sym[l]] = True
                    self.is_sym_y[faces_sym[l]] = False
                    self.is_sym_z[faces_sym[l]] = False
                elif s.lower() == "y":
                    self.is_sym_x[faces_sym[l]] = False
                    self.is_sym_y[faces_sym[l]] = True
                    self.is_sym_z[faces_sym[l]] = False
                elif s.lower() == "z":
                    self.is_sym_x[faces_sym[l]] = False
                    self.is_sym_y[faces_sym[l]] = False
                    self.is_sym_z[faces_sym[l]] = True
                else:
                    raise ValueError("symmetry must be x, y or z")

        if wall_cells is not None:
            for l in wall_cells:
                fi, _, _ = sps.find(g.cell_faces[:,l])
                for j in np.arange(fi.size):
                    self.is_vel[fi[j]] = False
                    self.is_pres[fi[j]] = False
                    self.is_sym[fi[j]] = False
                    self.is_wall[fi[j]] = True

class BoundaryConditionTransport(object):

    """ Class to store information on boundary conditions for transport.

    The BCs are specified by face number,
    and can have type Dirichlet or Neumann.

    Attributes:
        num_faces (int): Number of faces in the grid
        dim (int): Dimension of the boundary. One less than the dimension of
            the grid.
        is_dir (np.ndarray boolean, size g.num_faces): Element i is true if
            face i has been assigned a Dirichlet condition. Tacitly assumes that
            the face is on the boundary.
        is_neu (np.ndarary, boolean, size g.num_faces): Element i is true if
            face i has been assigned a Neumann condition.
    """

    def __init__(
        self, g, faces=None, cond=None
        ):
        """Constructor for BoundaryCondition.

        The conditions are specified by face numbers. Faces that do not get an
        explicit condition will have a pressure condition assigned.

        Parameters:
            g (grid): For which boundary conditions are set.
            faces (np.ndarray): Faces for which conditions are assigned.
            cond (list of str): Conditions on the faces, in the same order as
                used in faces. Should be as long as faces. The list elements
                should be one of "dir" or "neu".
        """

        self.num_faces = g.num_faces
        self.num_cells = g.num_cells
        self.dim = g.dim - 1

        # Find boundary faces
        bf = g.get_boundary_faces()

        self.is_neu = np.zeros(self.num_faces, dtype=bool)
        self.is_dir = np.zeros(self.num_faces, dtype=bool)

        # By default, all faces are Neumann.
        self.is_neu[bf] = True
        
        if faces is not None:
            # Validate arguments
            assert cond is not None
            if faces.dtype == bool:
                if faces.size != self.num_faces:
                    raise ValueError(
                        """When giving logical faces, the size of
                                        array must match number of faces"""
                    )
                faces = np.argwhere(faces)
            if not np.all(np.in1d(faces, bf)):
                raise ValueError(
                    "Give boundary condition only on the \
                                 boundary"
                )

            if isinstance(cond, str):
                cond = [cond] * faces.size
            if faces.size != len(cond):
                raise ValueError("One BC per face")

            for l in np.arange(faces.size):
                s = cond[l]
                if s.lower() == 'neu':
                    pass  # Neumann is already default
                elif s.lower() == 'dir':
                    self.is_dir[faces[l]] = True
                    self.is_neu[faces[l]] = False
                else:
                    raise ValueError('Boundary should be Dirichlet or Neumann')
