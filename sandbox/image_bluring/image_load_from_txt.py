import numpy as np
import matplotlib.pyplot as plt
import sys

# define image name and image dimensions, x,y,z
imageFilename = sys.argv[1] + "_out.txt"
x = int(sys.argv[2])
y = int(sys.argv[3])
z = int(sys.argv[4])

image = np.loadtxt(imageFilename, dtype='float64')

# reshape 1d image array in 2D RGB values (3D array)
image = np.reshape(image, (x, y, z), order='C')

#plot
plt.imshow(image)
plt.savefig('{}_blurred_c.png'.format(sys.argv[1]))