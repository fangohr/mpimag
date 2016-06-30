import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from blur import blur

#---------------------------------------------
# greyscale image blur
#---------------------------------------------

blur_factor = 3

image = io.imread("start.png")
image_blurred = blur(image, blur_factor=blur_factor)

# create plot comparing original and blurred image
f, axarr = plt.subplots(2)

axarr[0].imshow(image, cmap=plt.cm.Greys_r)
axarr[0].set_title('Original Image')

axarr[1].imshow(image_blurred, cmap=plt.cm.Greys_r)
axarr[1].set_title('Blurred Image')

# turn off axis
axarr[0].get_xaxis().set_visible(False)
axarr[0].get_yaxis().set_visible(False)
axarr[1].get_xaxis().set_visible(False)
axarr[1].get_yaxis().set_visible(False)

f.savefig('image_blur_compare_greyscale.png')


#---------------------------------------------
# colour image blur
#---------------------------------------------

from skimage.data import astronaut

blur_factor = 9

image = astronaut()
image_blurred = blur(image, blur_factor=blur_factor)

# create plot comparing original and blurred image
f, axarr = plt.subplots(2)

axarr[0].imshow(image)
axarr[0].set_title('Original Image')

axarr[1].imshow(image_blurred)
axarr[1].set_title('Blurred Image')

# turn off axis
axarr[0].get_xaxis().set_visible(False)
axarr[0].get_yaxis().set_visible(False)
axarr[1].get_xaxis().set_visible(False)
axarr[1].get_yaxis().set_visible(False)

f.savefig('image_blur_compare_colour.png')
