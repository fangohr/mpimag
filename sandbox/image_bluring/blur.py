import numpy as np

def blur(image, blur_factor=1):
    """Image blurring function.
    
    Takes an RGB image, with shape (x, y, 3) and blurs it according to the
    blur_factor, n, where x and y are the dimensions (in pixels of the image).

    Returns as blurred image as a numpy array with shape (x, y, 3) 
    
    Blurring is done as follows:
        The 'blurred' value at (xi, yi) is the average of each pixel at
        (xi, yi) and all the pixels surrounding it between xi-n, xi+n and
        yi-n, yi+n.
        
        Thus for n=1, the average is taken of the pixel and the 8 closest
        neighbouring pixels.
    """
    x, y = image.shape[0:2]
    blurred_image = np.zeros_like(image)
    
    # loop over all pixels
    for xi in range(x):
        # calculate the lower and up values of pixel x indices to average
        xi_lower = xi - blur_factor
        xi_upper = xi + blur_factor + 1

        #  correct x indices if out of range.
        if xi_lower < 0:
            xi_lower = 0
        # if xi_upper > x:
        #     xi_upper = x

        for yi in range(y):
            # calculate the lower and up values of pixel y indices to average
            yi_lower = yi - blur_factor
            yi_upper = yi + blur_factor + 1

            #  correct y indices if out of range.
            if yi_lower < 0:
                yi_lower = 0
            # if yi_upper > y:
            #     yi_upper = y

            # calculate the average of the pixel and all surrounding pixels
            # for each RGB colour

            # red channel
            red_channel_pixels = image[xi_lower:xi_upper,
                                       yi_lower:yi_upper,
                                       0]
            blurred_image[xi, yi, 0] = np.mean(red_channel_pixels.flatten())

            # green channel
            green_channel_pixels = image[xi_lower:xi_upper,
                                         yi_lower:yi_upper,
                                         1]
            blurred_image[xi, yi, 1] = np.mean(green_channel_pixels.flatten())

            # blue channel
            blue_channel_pixels = image[xi_lower:xi_upper,
                                        yi_lower:yi_upper,
                                        2]
            blurred_image[xi, yi, 2] = np.mean(blue_channel_pixels.flatten())

    return blurred_image