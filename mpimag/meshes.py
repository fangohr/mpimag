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

import numpy as np

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
		if x0 >= x1:
			raise ValueError("Attempting to set x0 as greater than or\
							  equal to x1. x0 should be less that x1. ")
		self._x0 = x0
		self._x1 = x1
		self._nx = nx
		self._cells = self._calculate_cells()
		self._dims = 1

	def _calculate_cells(self):
		"""
		Calculates the x-coordinates of the cells.
		"""
		nodes = np.linspace(self._x0, self._x1, self._nx)
		cells = nodes[0:-1] + 0.5 * nodes[1] - nodes[0]
		return cells

	def _get_cells(self):
		return self._cells

	cells = property(fget=_get_cells)

	def _get_dims(self):
		return self._dims

	dims = property(fget=_get_dims)
