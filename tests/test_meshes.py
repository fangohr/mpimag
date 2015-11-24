"""
=========================================================================
Mesh Tests
=========================================================================
Tests for FD meshes 

-------------------------------------------------------------------------
0D meshes
-------------------------------------------------------------------------
parameters:
	- none

properties:
	- dims - number of dimensions, 0

tests:
	- dims should equal 0

-------------------------------------------------------------------------
1D meshes
-------------------------------------------------------------------------
parameters:
	- x0 - position of first node
	- x1 - position of final node
	- nx - number of nodes. The nodes are assumed to be equally spaced.

properties:
	- cells - x-coordinates of the centre of the cells
	- dims - number of dimensions, 1

tests:
	- x1 should be greater that x0. Exception should be raised if not.
	- correct number of cells created (should be nx-1)
	- calculation of the cell coordinates.
	- dims should equal 1
"""
import numpy as np
import pytest

# ----------------------------------------------------------------------
# 0D mesh tests
# ----------------------------------------------------------------------

def test_mesh0d():
	from mpimag import FDmesh0D
	mesh = FDmesh0D()
	assert mesh.dims == 0

# ----------------------------------------------------------------------
# 1D mesh tests
# ----------------------------------------------------------------------

def setup_1d():
	"""
	Define the mesh parameters.
	Create a 1D mesh with the relevant parameters
	Return the mesh.

	A mesh of length 10 is created, with 6 nodes at:
		x = 0, 2, 4, 6, 8, 10
	The centres of the cells are thus located at:
		x = 1, 3, 5, 7, 9
	"""
	from mpimag import FDmesh1D
	x0 = 0
	x1 = 10
	xn = 6

	mesh = FDmesh1D(x0, x1, xn)
	return mesh


def test_mesh1d_creation():
	"""
	The value of x0 should not be greater than x1.
	Test that exception is raised if attempt is made to create mesh with
	x0 > x1.

	value of dims should be 1
	"""
	from mpimag import FDmesh1D
	with pytest.raises(ValueError):
		FDmesh1D(0, -10, 6)

	mesh = setup_1d()

	assert mesh.dims == 1


def test_mesh1d_cells():
	"""
	Test that the correct number of cells are created
	Test that the coordinates of the cells are correct.
	"""
	mesh = setup_1d()

	assert len(mesh.cells == 5)
	assert (mesh.cells == [1, 3, 5, 7, 9]).all()
