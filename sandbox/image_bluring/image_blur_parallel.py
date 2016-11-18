from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpi4py import MPI
import numpy as np
import sys

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
imageFilename = sys.argv[1]

if rank == 0:
    # from skimage.data import astronaut
    # need to store image with dtype=float64 (this corresponds to MPI.DOUBLE),
    # as there are issues with scattering data with dtype=uint8

    # img = astronaut().astype('float64')
    img = mpimg.imread("images/{}.png".format(imageFilename)).astype('float64')
else:
    img = None

# broadcast image dimensions to other processes
if rank == 0:
    dims = img.shape[0:2]
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
xLocals = np.ones(size, dtype = 'i') * (x // size)
# distribute the n remaining rows across the first n processes
remaining_rows = x % size
xLocals[0:remaining_rows] += 1

# array which details how much data is sent to each process
sendcounts = xLocals * y * 3
# array which details the displacement along img, from which the data
# for each process if scattered from
displacements = np.concatenate(([0], np.cumsum([sendcounts])[0:-1]),
                                axis=0)

#-----------------------------------------------------------------------------
# Define number of ghost points
#-----------------------------------------------------------------------------
# Define number or rows of ghosts above and below local image.

# There are no possible ghost points above image process zero as this is the
# top of the original image.

if (rank == 0):
    xGhostsAbove = 0
    xGhostsBelow = blur_factor

elif (rank == (size - 1)):
    xGhostsAbove = blur_factor
    xGhostsBelow = 0

else:
    xGhostsAbove = blur_factor
    xGhostsBelow = blur_factor

#-----------------------------------------------------------------------------
# Create local image array (and populate with relevant data)
#-----------------------------------------------------------------------------
# create an empty array on each process for which to scatter the section of
# the image into.
# This array also will include the ghost points
xLocal = xLocals[rank]
imgLocal = np.empty((xGhostsAbove + xLocal + xGhostsBelow, y, 3))

# Scatter the data
comm.Scatterv([img, sendcounts, displacements, MPI.DOUBLE],
               imgLocal[xGhostsAbove:],
               root=0)

#-----------------------------------------------------------------------------
# Halo method for data swapping
#-----------------------------------------------------------------------------
# define the ranks which lie 'above' and 'below'. Assume there is no
# periodicity
cart_comm = comm.Create_cart([size], periods=[0])
above_rank, below_rank = cart_comm.Shift(0, 1)
# Halo part one. Send the bottom part of image to rank 'below'. The rank below
# will use this data a ghost pixels for the top of it's image.

cart_comm.Sendrecv(sendbuf=imgLocal[xGhostsAbove + xLocal - blur_factor : xGhostsAbove + xLocal],
                   dest=below_rank,
                   recvbuf=imgLocal[0:xGhostsAbove],
                   source=above_rank)

# Halo part two: and repeat in opposite direction
cart_comm.Sendrecv(sendbuf=imgLocal[xGhostsAbove : xGhostsAbove + blur_factor],
                   dest=above_rank,
                   recvbuf=imgLocal[xGhostsAbove + xLocal:],
                   source=below_rank)

#-----------------------------------------------------------------------------
# Blur, plot and save local data
#-----------------------------------------------------------------------------

# blurring including blurred ghosts
imgLocalBlurred = blur(imgLocal, blur_factor=blur_factor)

f, axarr = plt.subplots(2)

axarr[0].imshow(imgLocal)
axarr[0].set_title('Original Image, rank {}'.format(rank))

axarr[1].imshow(imgLocalBlurred)
axarr[1].set_title('Blurred Image, rank {}'.format(rank))

# turn off axis
axarr[0].get_xaxis().set_visible(False)
axarr[0].get_yaxis().set_visible(False)
axarr[1].get_xaxis().set_visible(False)
axarr[1].get_yaxis().set_visible(False)

f.savefig("images/python/image_blur_parallel_{}".format(rank))

#-----------------------------------------------------------------------------
# Test!
#-----------------------------------------------------------------------------
# first gather data onto process 0.

# define array to gather the blurred image data into
if rank == 0:
    imgBlurred = np.empty((x, y, 3))
else:
    imgBlurred = None

# sendcounts and displacements already defined from before

# Gather the data
comm.Gatherv(imgLocalBlurred[xGhostsAbove : xGhostsAbove + xLocal], [imgBlurred,
                                                sendcounts,
                                                displacements,
                                                MPI.DOUBLE], root=0)

# compare the two arrays: the image blurred in serial and the image blurred
# in parallel then gathered onto process 0.
if rank == 0:
    # blur the original image in serial
    imBlurredSerial = blur(img, blur_factor=blur_factor)
    # check it is the same as the image blurred in parallel
    assert((imBlurredSerial == imgBlurred).all())

# plot complete images (original and blurred)
if rank == 0:
    f, axarr = plt.subplots(2)

    axarr[0].imshow(img)
    axarr[0].set_title('Original Image, rank {}'.format(rank))

    axarr[1].imshow(imgBlurred)
    axarr[1].set_title('Blurred Image, rank {}'.format(rank))

    # turn off axis
    axarr[0].get_xaxis().set_visible(False)
    axarr[0].get_yaxis().set_visible(False)
    axarr[1].get_xaxis().set_visible(False)
    axarr[1].get_yaxis().set_visible(False)

    f.savefig("images/python/image_blurred_gathered.png".format(rank))
