"""
Mpi4py example of a 1D mesh.

The global coordinates of the mesh are:
	- x0_global: the left most x coordinate
	- x1_global: the right most x coordinate
	- xn_global: the total number of modes

Each process has knowledge of these global parameters. The processes then create a local
mesh based on the knowledge of these global parameters. The distribution of the number of
total number of nodes each process holds is:
 	= ceil(xn_global / total no. of processes)
Except for the final process, which takes the remaining nodes

E.g. the structure of the mesh with 3 processes, p0, p1, p2
with 11 nodes (where '*' marks the nodes) is

  x0_global          x_left(p1)   x0_local(p1)         x1_local(p1)   x_right(p1)  x1_global
     |                     \         |                       |         /             |
     V                      \        V                       V        /              V
     *       *       *       *       *       *       *       *       *       *       *
 |-------------------------------|-------------------------------|-----------------------|
 |       local p0 mesh           |        local p1 mesh          |     local p2 mesh     |
                                          

The mesh needs to have knowledge of the coordinates either side of it which are stored on
difference processes. To get this data, a data swap occurs between the processes, using
sendrecv.

Periodic boundary conditions are assumed.

In this example, x0_global = 0, x1_global = 1 and xn_global = 11.

The nodes spacing is therefore = 0.1

On process 0:
	- There should be 4 nodes
	- The coordinates of the nodes should be: 0.0, 0.1, 0.2, 0.3
	- The coordinate to the left should be 1.0
	- The coordinate to the right should be 0.4

On process 1:
	- There should be 4 nodes
	- the coordinates of the nodes should be: 0.4, 0.5, 0.6, 0.7
	- The coordinate to the left should be 0.3
	- The coordinate to the right should be 0.8

On process 2:
	- There should be 3 nodes
	- the coordinates of the nodes should be: 0.8, 0.9, 1.0
	- The coordinate to the left should be 0.7
	- The coordinate to the right should be 0.0
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

size = comm.size
rank = comm.rank

# global mesh parameters
x0_global = 0
x1_global = 1
xn_global = 11

#---------------------------------------------------------------------------------------------------
# Calculation of the local mesh coordinates
#---------------------------------------------------------------------------------------------------

# spacing between the nodes
cell_spacing = (x1_global - x0_global) / float(xn_global - 1)

# calculated the local values, x0_local, x1_local and x_local
# The final process takes the remaining mesh, not taken up by the other prcoesses
if rank == (size - 1):
	xn_local_full = np.ceil(xn_global / np.float(size)) # the number of nodes on the other processes
	xn_local = xn_global - ((size - 1) * xn_local_full)
	x0_local = x0_global + (rank * cell_spacing * xn_local_full)
	x1_local = x1_global
else:
	xn_local = np.ceil(xn_global / np.float(size))
	x0_local = x0_global + (rank * cell_spacing * xn_local)
	x1_local = x0_local + (cell_spacing * (xn_local - 1))

# compute the x coordinates of the cells that each process has
x_values_local = np.linspace(x0_local, x1_local, xn_local)

#---------------------------------------------------------------------------------------------------
# Finding out data held on neighbouring mesh segements (on 'neighbouring' processes)
#---------------------------------------------------------------------------------------------------

# Determine which values to send to the other processes.
# E.g. the first x value and final x value of the local mesh
x0_send = x_values_local[0]
x1_send = x_values_local[-1]

# x0_send pairs with x_right:
#	x0_send from rank swaps with x1_send of rank_left 
# x1 send pairs with x_left:
#	x1_send from rank swaps with x0_send of rank_right

# Determine the rank numbers of the process to left and right (assuming periodic
# boundary conditions).
rank_left = (size + rank -1) % size
rank_right = (size + rank + 1) % size

# # Example of data swap for a single process
# if rank == 2:
# 	x_left = comm.sendrecv(x0_send, source=rank_left, dest=rank_left)
# 	print "I am process {}. The value on my left is {}".format(rank, x_left)
# if rank == 1:
# 	x_right = comm.sendrecv(x1_send, source=rank_right, dest=rank_right)
# 	print "I am process {}. The value on my right is {}".format(rank, x_right)

# Data swapping for all processes
# need to loop to prevent deadlock
for r in range(size):
	r_right = (size + r + 1) % size

	if rank == r:
		x_left = comm.sendrecv(x0_send, source=rank_left, dest=rank_left)
	if rank == r_right:
		x_right = comm.sendrecv(x1_send, source=rank_right, dest=rank_right)

print "I am process {}. my x-coords = {}. Left x-coord = {}. Right x-coord = {}".format(rank, x_values_local, x_left, x_right)
