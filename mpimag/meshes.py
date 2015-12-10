"""
=========================================================================
Meshes
=========================================================================

-------------------------------------------------------------------------
0D meshes
-------------------------------------------------------------------------

The 0D mesh is a simply a single spin at a point. It has the following
properties associated with it:

=========================================================================
Property    | Description                       | Usage
=========================================================================
dims        | The number of dimensions, which   |
            | is 0.                             |
=========================================================================


-------------------------------------------------------------------------
1D meshes
-------------------------------------------------------------------------

The 1D mesh object has the following properties associated with it:

=========================================================================
Property    | Description                       | Usage
=========================================================================
x0          | The position of the first node    | int or float
-------------------------------------------------------------------------
x1          | The position of the final node    | int or float
-------------------------------------------------------------------------
nx          | The number of nodes               | int
-------------------------------------------------------------------------
cells       | The coordinates of the centres of | list [c0, c1,..., cn-1]
            | cells                             |
-------------------------------------------------------------------------
dims        | The number of dimensions, which   |
            | is 1.                             |
=========================================================================

Usage:
    mesh = FDmesh1D(x0, x1, nx)

"""
from __future__ import division
import numpy as np
from mpi4py import MPI


class FDmesh0D(object):
    """
    0D mesh

    NOTE: Seems a bit pointless having a class for this. It has been
    implemented so that there is consistency when meshes of different
    dimensions are used. This will allow the Macrospin and future
    Simulation classes to be more elegant.

    """
    def __init__(self):
        self._dims = 0

    def _get_dims(self):
        return self._dims

    dims = property(fget=_get_dims)

class FDmesh1D(object):
    """
    1D meshes
    """
    def __init__(self, x0, x1, nx):

        # Setting up of process ranks
        self._comm = MPI.COMM_WORLD#comm
        self._size = self._comm.size
        self._rank = self._comm.rank

        if x0 >= x1:
            raise ValueError("Attempting to set x0 as greater than or\
                              equal to x1. x0 should be less that x1.")
        self._x0 = x0
        self._x1 = x1
        self._nx = nx
        self._ncells = nx - 1 # number of cells
        self._dims = 1 # mesh dimension


        self._node_spacing = (self._x1 - self._x0) / self._ncells
        # the number of cells on a 'full' process
        # self._ncells_local_full = np.ceil(self._ncells / self._size)
        self._ncells_local_full = self._ncells // self._size
        # the number of cells on a 'partially full' process.
        # This happens when (ncells % size != 0).
        self._ncells_local_partial = self._ncells - ((self._size - 1) * self._ncells_local_full)

        # coordinate of the local x0 node
        self._x0_local = self._x0 + (self._rank * self._node_spacing * self._ncells_local_full)

        # calculated the local values, x0_local, x1_local and x_local
        # The final process takes the remaining mesh, not taken up by the other prcoesses
        if self._rank == (self._size - 1):
            # coordinate of the local x1 node
            self._x1_local = self._x1
            # number of local nodes
            self._nx_local = self._ncells_local_partial + 1#self._nx - ((self._size - 1) * self._ncells_local_full)
        else:
            self._x1_local = self._x0 + (self._rank + 1) * self._node_spacing * (self._ncells_local_full)
            self._nx_local = self._ncells_local_full + 1

        # calculate the local node coords held on each process
        self._nodes_local = np.linspace(self._x0_local, self._x1_local, self._nx_local)
        # calculate the coords of the cell centres on each process.
        self._cells_local = self._calculate_cells_local()

    def _get_nodes_local(self):
        return self._nodes_local

    nodes_local = property(fget=_get_nodes_local)

    def _calculate_cells_local(self):
        cells_local = self._nodes_local[0:-1] + 0.5 * self._node_spacing
        return cells_local

    def _get_cells_local(self):
        return self._cells_local

    cells_local = property(fget=_get_cells_local)

    def _get_cells_global(self):
        """
        Calculates the x-coordinates of the cells.
        """
        # Create array in which global cell data is gathered into.
        # It is gathered onto process rank 0, thus this the array
        # on this process is initialised as the same length as the
        # global number of cells
        if self._rank == 0:
            self._cellsGathered = np.empty(self._ncells)
        else:
            self._cellsGathered = None
        self._comm.Barrier()
        # set up sendcounts tuple for comm.Gather. This is a tuple containing
        # the length of the array which each process send. Thus it is in the format
        # (ncells_process0, ncells_process1, ..., ncell_processn-1)
        # sendcounts = tuple([int(self._ncells_local_full)] * (self._size - 1) + [int(self._ncells_local_partial)])
        sendcounts = tuple([self._ncells_local_full] * (self._size - 1) + [int(self._ncells_local_partial)])
        # Set up the displacements tuple for comm.Gather. This is a tuple containing
        # the indicies where each set of local data should be placed from in the global
        # array (cellsGathered).
        # displacements = tuple([rank * int(self._ncells_local_full) for rank in range(self._size)])
        displacements = tuple([rank * self._ncells_local_full for rank in range(self._size)])
        self._comm.Barrier()
        # Gatherv is used instead of comm.gather as the length of _cells_local
        # is not neccessarily the same on each process. comm.gather only works if
        # the local data arrays being gathered is all the same legnth.
        self._comm.Gatherv(self._cells_local, [self._cellsGathered, sendcounts, displacements, MPI.DOUBLE])#self._comm.gather(self._cells_local.tolist(), cells, root=0)
        self._comm.Barrier()
        # return self._cellsGathered
        if self._rank != 0:
            # TODO: would be good to raise an attribute error here when trying to access
            # global cell data from process other than 0th.
            self._cellsGathered = "Global cell data only available from process 0."
        return self._cellsGathered

    cells = property(fget=_get_cells_global)

    def _get_dims(self):
        return self._dims

    dims = property(fget=_get_dims)

    def _get_ncells(self):
        """
        Returns a tuple, (a,b,c,d) where a,b,c is:
            a) the total number of cells
            b) the number of local cells on the process
            c) the number of local cells on a 'full' process
            d) the number of local cells on the final process
        """
        return (self._ncells, len(self._cells_local), self._ncells_local_full, self._ncells_local_partial)
    
    ncells = property(fget=_get_ncells)