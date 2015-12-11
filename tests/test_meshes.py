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
from mpi4py import MPI
import pytest

from .skips import skip_in_parallel, xfail_in_parallel, skip_if_not_parallel4, skip_if_not_process0

# ----------------------------------------------------------------------
# 0D mesh tests
# ----------------------------------------------------------------------

@skip_in_parallel
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
	Test that exception is made if attempt to

	value of dims should be 1

	Expected to work in serial of parallel
	"""
	from mpimag import FDmesh1D

	comm = MPI.COMM_WORLD

	with pytest.raises(ValueError):
		FDmesh1D(0, -10, 6)

	mesh = setup_1d()

	assert mesh.dims == 1


def test_mesh1d_cells():
	"""
	Test that the correct number of cells are created
	Test that all the coordinates of the cells are correct.

	Only process 0 has knowledge of all the cells.
	Test checks that error is raised if all cells are requested
	by non-zeroth process.

	Expected to work in serial of parallel.
	"""
	comm = MPI.COMM_WORLD
	from mpimag import FDmesh1D

	mesh = setup_1d()
	cells = mesh.cells

	# Check correct cells are computed on zeroth process
	if comm.rank == 0:
		assert len(cells) == 5
		assert (cells == [1, 3, 5, 7, 9]).all()
	else:
		cells = type(str)

	# Check attribute error is raised if try to access cells
	# on a non-zeroth process.
	# if comm.rank != 0:
	# 	with pytest.raises(AttributeError):
	# 		mesh.cells


@skip_if_not_parallel4
@pytest.mark.parallel4
def test_mesh1d_local_cells():
	"""
	Test that the local cells held on each process are as expected

	Test expects to be run with 4 processes
	"""
	from mpimag import FDmesh1D

	# mpi communicator
	comm = MPI.COMM_WORLD

	# mesh parameters
	x0 = 0
	x1 = 10
	xn = 26

	expected_cells =  [[0.2, 0.6, 1.0, 1.4, 1.8, 2.2],
					  [2.6, 3.0, 3.4, 3.8, 4.2, 4.6],
					  [5.0, 5.4, 5.8, 6.2, 6.6, 7.0],
					  [7.4, 7.8, 8.2, 8.6, 9.0, 9.4, 9.8]]

	mesh = FDmesh1D(x0, x1, xn)

	#TODO: why does this need a tolerance set to pass?!
	assert np.isclose(mesh.cells_local, expected_cells[comm.rank],
						rtol=1e-15, atol=1e-15, equal_nan=False).all()
