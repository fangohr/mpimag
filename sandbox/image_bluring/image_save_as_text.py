import matplotlib.image as mpimg
import numpy as np
import sys

imageFilename = sys.argv[1]

image = mpimg.imread(imageFilename + '.png').astype('float64')

# if the image is greyscale without individual values for RGB.
# create an array with individual RGB values (the RGB are all
# equal).
if len(image.shape) == 2:
	imageRGB = np.empty((image.shape[0], image.shape[1], 3))
	imageRGB[:,:,0] = image
	imageRGB[:,:,1] = image
	imageRGB[:,:,2] = image
	image = imageRGB

# Remove the alpha value if the image has one
if image.shape[2] == 4:
	imageRGB = np.empty((image.shape[0], image.shape[1], 3))
	imageRGB[:,:,:] = image[:,:,0:3]
	image = imageRGB

# find the image dimensions
shape = image.shape
print '{} {} {}'.format(*shape)

# flatten the image for c processing and save as a txt file
image = image.flatten()
np.savetxt(imageFilename + '.txt', image)
