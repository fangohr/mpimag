"""
=========================================================================
Mesh Tests
=========================================================================
Tests for FD meshes 

-------------------------------------------------------------------------
1D meshes
-------------------------------------------------------------------------
parameters:
	- x0 - position of first node
	- x1 - position of final node
	- nx - number of nodes. The nodes are assumed to be equally spaced.

properties:
	- cells - x-coordinates of the centre of the cells

tests:
	- x1 should be greater that x0. Exception should be raised if not.
	- correct number of cells created (should be nx-1)
	- calculation of the cell coordinates.
"""
import numpy as np
import pytest

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

@pytest.mark.xfail
def test_mesh_creation():
	"""
	The value of x0 should not be greater than x1.
	Test that exception is raised if attempt is made to create mesh with
	x0 > x1.
	"""
	from mpimag import FDmesh1D
	with pytest.raises(ValueError):
		FDmesh1D(0, -10, 6)

@pytest.mark.xfail
def test_cells():
	"""
	Test that the correct number of cells are created
	Test that the coordinates of the cells are correct.
	"""
	mesh = setup_1d()

	assert len(sim.cells == 5)
	assert sim.cells == [1, 3, 5, 7, 9]
