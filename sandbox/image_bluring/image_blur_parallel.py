from __future__ import division
import matplotlib.pyplot as plt
from skimage import io
from mpi4py import MPI
import numpy as np

# Setting up of process ranks
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

# read image in on rank 0
# note: could read the image on all processes,
# but reading in on one and then scattering the
# array is a good exercise in itself.
if rank == 0:
	from skimage.data import astronaut
	# need to store image with dtype=float64 (this 
	# corresponds to MPI.DOUBLE), as there are issues
	# with scattering data with dtype=uint8
	image_full = astronaut().astype('float64')
else:
	image_full = None

# broadcast image dimensions to other processes
if rank == 0:
	dims = image_full.shape[0:2]
else:
	dims = np.empty(2, dtype='i')

dims = comm.bcast(dims, root = 0)
x,y = dims

# print "rank = {}, x,y = {},{}".format(rank, x, y) 

#-----------------------------------------------------
# Scattering the image
#-----------------------------------------------------
# scatter sections of the image to different processes
# The image is divided in the simplest way:
#	the image is divided into p strips where 'p' is the
#	number of processes.

# Each process takes an equal number of rows. If the
# total number of rows, r cannot be split equally over all
# the processes, the first (r%p) processes take (r//p + 1)
# and the remaining processes take (r//p) rows.
# e.g if r = 10 and p = 3, the distributions of rows would
# be (4,3,3)

# array detailing how many rows each process owns, where
# the index of the array, corresponds to the process no.
x_locals = np.ones(size, dtype = 'i') * (x // size)
# distribute the n remaining rows across the first n
# processes
remaining_rows = x % size
x_locals[0:remaining_rows] += 1

# array which details how much data is sent to each process
sendcounts = x_locals * y * 3
# array which details the displacement along image_full, from
# which the data for each process if scattered from
displacements = np.concatenate(([0], np.cumsum([sendcounts])[0:-1]),
								axis=0)

# create an empty array on each process for which to scatter
# the section of the image into
x_local = x_locals[rank]
image_local = np.empty((x_local, y, 3))

# Scatter the data
comm.Scatterv([image_full, sendcounts, displacements, MPI.DOUBLE], image_local, root=0)

# print "rank = {}, size = {}".format(rank, image_local.shape)

# plot to show successful scatter
plt.imshow(image_local.astype('uint8'))
plt.show()
