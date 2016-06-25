import matplotlib.pyplot as plt
import numpy as np
from skimage import io

def blur(image, blur_factor=1):
    """
    Image blurring function.
    
    Takes an RGB image, with shape (x,y,3) and blurs it according to the blur_factor, n.
    x and y are the dimensions (in pixels of the image). 
    
    Blurring is done as follows:
        The 'blurred' value at (xi, yi) is the average of each pixel at (xi, yi) and all
        the pixels surrounding it between xi-n, xi+n and yi-n,yi+n.
        
        Thus for n=1, the average is taken of the pixel and the 8 closest neighbouring pixels.
    """
    x,y = image.shape[0:2]
    blurred = np.zeros_like(image)
    
    # loop over all pixels
    for xi in range(x):
        # calculate the lower and up values of pixel x indices to average
        xi_lower = xi - blur_factor
        xi_upper = xi + blur_factor + 1

        #  correct x indices if out of range.
        if xi_lower < 0:
            xi_lower = 0
        if xi_upper > x:
            xi_upper = x

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
            red_channel_pixels = image[xi_lower:xi_upper, yi_lower:yi_upper, 0]
            blurred[xi, yi, 0] = np.mean(red_channel_pixels.flatten())
            # green channel
            green_channel_pixels = image[xi_lower:xi_upper, yi_lower:yi_upper, 1]
            blurred[xi, yi, 1] = np.mean(green_channel_pixels.flatten())
            # blue channel
            blue_channel_pixels = image[xi_lower:xi_upper, yi_lower:yi_upper, 2]
            blurred[xi, yi, 2] = np.mean(blue_channel_pixels.flatten())

    return blurred


if __name__ == "__main__":

    #---------------------------------------------
    # greyscale image blur
    #---------------------------------------------

    blur_factor = 3

    image = io.imread("start.png")
    blurred = blur(image, blur_factor=blur_factor)

    # create plot comparing original and blurred image
    f, axarr = plt.subplots(2)

    axarr[0].imshow(image, cmap=plt.cm.Greys_r)
    axarr[0].set_title('Original Image')

    axarr[1].imshow(blurred, cmap=plt.cm.Greys_r)
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
    blurred = blur(image, blur_factor=blur_factor)

    # create plot comparing original and blurred image
    f, axarr = plt.subplots(2)

    axarr[0].imshow(image)
    axarr[0].set_title('Original Image')

    axarr[1].imshow(blurred)
    axarr[1].set_title('Blurred Image')

    # turn off axis
    axarr[0].get_xaxis().set_visible(False)
    axarr[0].get_yaxis().set_visible(False)
    axarr[1].get_xaxis().set_visible(False)
    axarr[1].get_yaxis().set_visible(False)

    f.savefig('image_blur_compare_colour.png')
