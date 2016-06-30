from __future__ import division
import matplotlib.pyplot as plt
from skimage import io
from mpi4py import MPI
import numpy as np

from blur import blur

# Setting up of process ranks
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
blur_factor = 4

#-----------------------------------------------------------------------------
# read image in on rank 0
#-----------------------------------------------------------------------------
# note: could read the image on all processes, but reading in on one and then
# cattering the array is a good exercise in itself.
if rank == 0:
    # from skimage.data import astronaut
    # need to store image with dtype=float64 (this corresponds to MPI.DOUBLE),
    # as there are issues with scattering data with dtype=uint8

    # image_full = astronaut().astype('float64')
    image_full = io.imread("start.png").astype('float64')
else:
    image_full = None

# broadcast image dimensions to other processes
if rank == 0:
    dims = image_full.shape[0:2]
else:
    dims = np.empty(2, dtype='i')

dims = comm.bcast(dims, root = 0)
x,y = dims

#-----------------------------------------------------------------------------
# Scattering the image
#-----------------------------------------------------------------------------
# scatter sections of the image to different processes
# The image is divided in the simplest way:
#   the image is divided into p strips where 'p' is the
#   number of processes.

# Each process takes an equal number of rows. If the total number of rows, r
# cannot be split equally over all the processes, the first (r%p) processes
# take (r//p + 1) and the remaining processes take (r//p) rows.
# e.g if r = 10 and p = 3, the distributions of rows would be (4,3,3)

# array detailing how many rows each process owns, where the index of the
# array, corresponds to the process no.
x_locals = np.ones(size, dtype = 'i') * (x // size)
# distribute the n remaining rows across the first n processes
remaining_rows = x % size
x_locals[0:remaining_rows] += 1

# array which details how much data is sent to each process
sendcounts = x_locals * y * 3
# array which details the displacement along image_full, from which the data
# for each process if scattered from
displacements = np.concatenate(([0], np.cumsum([sendcounts])[0:-1]),
                                axis=0)

# create an empty array on each process for which to scatter the section of
# the image into
x_local = x_locals[rank]
image_local = np.empty((x_local, y, 3))

# Scatter the data
comm.Scatterv([image_full, sendcounts, displacements, MPI.DOUBLE],
               image_local,
               root=0)

#-----------------------------------------------------------------------------
# Get required data held on other processes
#-----------------------------------------------------------------------------
# create empty arrays ghosts points 'above' and 'below' the local_image

# There are no possible ghost points above image process zero as this is the
# top of the original image.
if rank > 0:
    ghost_above = np.empty((blur_factor, y, 3))
    num_ghosts_above = blur_factor
if rank == 0:
    ghost_above = np.empty((0, y, 3))
    num_ghosts_above = 0
# likewise, no ghost points below image stored on highest numbered process
if rank < (size - 1):
    ghost_below = np.empty((blur_factor, y, 3))
if rank == size - 1:
    ghost_below = np.empty((0, y, 3))


# # 'Rebecca' method for data swapping
# #-----------------------------------
# # Send and recv data swaps
# # TODO: explain how this works!
# if rank % 2 == 0 and rank != (size-1):
#     comm.Sendrecv(sendbuf=image_local[-blur_factor:],
#                     dest=rank+1,
#                     sendtag=rank,
#                     recvbuf=ghost_below,
#                     source=rank+1,
#                     recvtag=rank+1)

# if rank % 2 == 1 and rank != 0:
#     comm.Sendrecv(sendbuf=image_local[0:blur_factor],
#                     dest=rank-1,
#                     sendtag=rank,
#                     recvbuf=ghost_above,
#                     source=rank-1,
#                     recvtag=rank-1)

# comm.Barrier()

# if rank % 2 == 1 and rank != (size-1):
#     comm.Sendrecv(sendbuf=image_local[-blur_factor:],
#                     dest=rank+1,
#                     sendtag=rank,
#                     recvbuf=ghost_below,
#                     source=rank+1,
#                     recvtag=rank+1)

# if rank % 2 == 0 and rank != 0:
#     comm.Sendrecv(sendbuf=image_local[0:blur_factor],
#                     dest=rank-1,
#                     sendtag=rank,
#                     recvbuf=ghost_above,
#                     source=rank-1,
#                     recvtag=rank-1)

# Halo method for data swapping
#------------------------------

# define the ranks which lie 'above' and 'below'. Assume there is no
# periodicity
cart_comm = comm.Create_cart([size], periods=[0])
above_rank, below_rank = cart_comm.Shift(0, 1)

# Halo part one. Send the bottom part of image to rank 'below'. The rank below
# will use this data a ghost pixels for the top of it's image.
cart_comm.Sendrecv(sendbuf=image_local[-blur_factor:],
                   dest=below_rank,
                   recvbuf=ghost_above,
                   source=above_rank)

# Halo part two: and repeat in opposite direction
cart_comm.Sendrecv(sendbuf=image_local[0:blur_factor],
                   dest=above_rank,
                   recvbuf=ghost_below,
                   source=below_rank)


# create array containing local image plus the surrounding ghost pixels
image_local_ghosts = np.concatenate((ghost_above, image_local, ghost_below))

#-----------------------------------------------------------------------------
# Blur, plot and save local data
#-----------------------------------------------------------------------------

# blurring including blurred ghosts
image_local_ghosts_blurred = blur(image_local_ghosts, blur_factor=blur_factor)

# blurred image without ghosts
image_local_blurred = \
    image_local_ghosts_blurred[num_ghosts_above: x_local + num_ghosts_above
                               :,
                               :]

f, axarr = plt.subplots(2)

axarr[0].imshow(image_local.astype('uint8'))
axarr[0].set_title('Original Image, rank {}'.format(rank))

axarr[1].imshow(image_local_blurred.astype('uint8'))
axarr[1].set_title('Blurred Image, rank {}'.format(rank))

# turn off axis
axarr[0].get_xaxis().set_visible(False)
axarr[0].get_yaxis().set_visible(False)
axarr[1].get_xaxis().set_visible(False)
axarr[1].get_yaxis().set_visible(False)

f.savefig("image_blur_parallel_{}".format(rank))

#-----------------------------------------------------------------------------
# Test!
#-----------------------------------------------------------------------------
# first gather data onto process 0.

# define array to gather the blurred image data into
if rank == 0:
    image_blurred_gathered = np.empty((x, y, 3))
else:
    image_blurred_gathered = None

# sendcounts and displacements already defined from before

# Gather the data
comm.Gatherv(image_local_blurred, [image_blurred_gathered,
                                   sendcounts,
                                   displacements,
                                   MPI.DOUBLE], root=0)
# compare the two arrays: the image blurred in serial and the image blurred
# in parallel then gathered onto process 0.
if rank == 0:
    # blur the original image in serial
    image_blurred_serial = blur(image_full, blur_factor=blur_factor)
    # check it is the same as the image blurred in parallel
    assert((image_blurred_serial == image_blurred_gathered).all())

# plot complete images (original and blurred)
if rank == 0:
    f, axarr = plt.subplots(2)

    axarr[0].imshow(image_full.astype('uint8'))
    axarr[0].set_title('Original Image, rank {}'.format(rank))

    axarr[1].imshow(image_blurred_gathered.astype('uint8'))
    axarr[1].set_title('Blurred Image, rank {}'.format(rank))

    # turn off axis
    axarr[0].get_xaxis().set_visible(False)
    axarr[0].get_yaxis().set_visible(False)
    axarr[1].get_xaxis().set_visible(False)
    axarr[1].get_yaxis().set_visible(False)

    f.savefig("image_blurred_gathered.png".format(rank))
