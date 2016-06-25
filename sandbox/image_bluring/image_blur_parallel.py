from __future__ import division
import matplotlib.pyplot as plt
from skimage import io
from mpi4py import MPI
import numpy as np

# Setting up of process ranks
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
blur_factor = 4

def blur(image_local_ghosts, ghosts, x_local, y, blur_factor=1):
    """
    Image blurring function.
    
    Takes an RGB image, with shape (x_local,y,3) and blurs it according to the blur_factor, n.
    x and y are the dimensions (in pixels of the image). 
    
    Blurring is done as follows:
        The 'blurred' value at (xi_local, yi) is the average of each pixel at (xi_local, yi) and all
        the pixels surrounding it between xi_local-n, xi_local+n and yi-n,yi+n.
        
        Thus for n=1, the average is taken of the pixel and the 8 closest neighbouring pixels.

    ghosts: number of upper ghost rows
    x_local: number of rows on local image (no ghosts included)
    """
    blurred = np.zeros((x_local, y, 3))

    # loop over all pixels
    for xi in range(ghosts, x_local + ghosts):
        # calculate the lower and up values of pixel x indices to average
        xi_lower = xi - blur_factor
        xi_upper = xi + blur_factor + 1

        #  correct x indices if out of range.
        if xi_lower < 0:
            xi_lower = 0
        # if xi_upper > x_local + ghosts:
        #     xi_upper = x_local + ghosts

        for yi in range(y):
            # calculate the lower and up values of pixel y indices to average
            yi_lower = yi - blur_factor
            yi_upper = yi + blur_factor + 1

            #  correct y indices if out of range.
            if yi_lower < 0:
                yi_lower = 0
            if yi_upper > y:
                yi_upper = y

            # calculate the average of the pixel and all surrounding pixels for
            # each RGB colour
            # red channel
            red_channel_pixels = image_local_ghosts[xi_lower:xi_upper, yi_lower:yi_upper, 0]
            blurred[xi-ghosts, yi, 0] = np.mean(red_channel_pixels.flatten())
            # green channel
            green_channel_pixels = image_local_ghosts[xi_lower:xi_upper, yi_lower:yi_upper, 1]
            blurred[xi-ghosts, yi, 1] = np.mean(green_channel_pixels.flatten())
            # blue channel
            blue_channel_pixels = image_local_ghosts[xi_lower:xi_upper, yi_lower:yi_upper, 2]
            blurred[xi-ghosts, yi, 2] = np.mean(blue_channel_pixels.flatten())

    return blurred

#-----------------------------------------------------
# read image in on rank 0
#-----------------------------------------------------
# note: could read the image on all processes,
# but reading in on one and then scattering the
# array is a good exercise in itself.
if rank == 0:
	from skimage.data import astronaut
	# need to store image with dtype=float64 (this 
	# corresponds to MPI.DOUBLE), as there are issues
	# with scattering data with dtype=uint8

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

# # plot to show successful scatter
# plt.imshow(image_local.astype('uint8'))
# plt.title("rank = {}".format(rank))
# plt.show()

#-----------------------------------------------------
# Get required data held on other processes
#----------------------------------------------------
# create empty arrays ghosts points 'above' and 'below'
# the local_image

# There are no possible ghost points above image process zero
# as this is the top of the original image.
if rank > 0:
	ghost_above = np.empty((blur_factor, y, 3))
# likewise, no ghost points below image stored on highest
# numbered process
if rank < (size - 1):
	ghost_below = np.empty((blur_factor, y, 3))

if rank == 0:
	ghost_above = np.empty((0, y, 3))

if rank == size - 1:
	ghost_below = np.empty((0, y, 3))

ghosts = [ghost_above.shape[0], ghost_below.shape[0]]

# Send and recv data swaps
# TODO: explain how this works!
if rank % 2 == 0 and rank != (size-1):
	comm.Sendrecv(sendbuf=image_local[-blur_factor:],
					dest=rank+1,
					sendtag=rank,
					recvbuf=ghost_below,
					source=rank+1,
					recvtag=rank+1)

if rank % 2 == 1 and rank != 0:
	comm.Sendrecv(sendbuf=image_local[0:blur_factor],
					dest=rank-1,
					sendtag=rank,
					recvbuf=ghost_above,
					source=rank-1,
					recvtag=rank-1)

comm.Barrier()

if rank % 2 == 1 and rank != (size-1):
	comm.Sendrecv(sendbuf=image_local[-blur_factor:],
					dest=rank+1,
					sendtag=rank,
					recvbuf=ghost_below,
					source=rank+1,
					recvtag=rank+1)

if rank % 2 == 0 and rank != 0:
	comm.Sendrecv(sendbuf=image_local[0:blur_factor],
					dest=rank-1,
					sendtag=rank,
					recvbuf=ghost_above,
					source=rank-1,
					recvtag=rank-1)


# create array with ghost pixels
image_local_ghosts = np.concatenate((ghost_above, image_local, ghost_below))

# blur, plot and save
blurred = blur(image_local_ghosts, ghosts[0], x_local, y, blur_factor=blur_factor)

f, axarr = plt.subplots(2)

axarr[0].imshow(image_local.astype('uint8'))
axarr[0].set_title('Original Image, rank {}'.format(rank))

axarr[1].imshow(blurred.astype('uint8'))
axarr[1].set_title('Blurred Image, rank {}'.format(rank))

# turn off axis
axarr[0].get_xaxis().set_visible(False)
axarr[0].get_yaxis().set_visible(False)
axarr[1].get_xaxis().set_visible(False)
axarr[1].get_yaxis().set_visible(False)

f.savefig("image_blur_parallel_{}".format(rank))

#-----------------------------------------------------
# Test
#----------------------------------------------------
# TODO