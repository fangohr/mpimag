/* To do
change z to c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define blur_factor 10

/*
 * Function: index 
 * ---------------
 * Helper function for accessing the correct indices.
 * x, y, z are the dimensions of the array.
 * i, j, k are the indices required to access.
 */
int indices(int x, int y, int z, int i, int j, int k){
	return k + (j * z) + (i * y * z);
}

/*
 * Function: read_file_to_array 
 * -----------------------------
 * Loads a txt file containing png image data and converts it to a 3 dimensional
 * array in c in the format (x,y,c), where x, y are dimensions (in pixels) of
 * the image and c are the number of RGB(A) colours (3 for RGB and 4 for RGBA).
 * The data is assumed to be stored file in the order,
 * (r(x0, y0), g(x0, y0), b(x0, y0), r(x0, y1), g(x0, y1), b(x0, y1)...)
 * The above example only has RGB values and no A (transparency values).
 *
 * Parameters
 * ----------
 * filename: string of the txt filename
 * imageArray: long double, the c array where the data will be written to.
 *
 */
void read_file_to_array(int x, int y, int z, char filename[], long double *imageArray)
{
	FILE *imagefile;

    int i, j, k;

    imagefile = fopen(filename, "r");

    // Loop through file and assign values to the image array.
    for(i = 0; i < x; i++){
    	for(j = 0; j < y; j++){
    		for(k = 0; k < z; k++){
    			fscanf(imagefile, "%Lf", &imageArray[indices(x, y, z, i, j, k)]);
    		}
    	}
    }

    fclose(imagefile);
}

/*
 * Function: blur_image 
 * --------------------
 * Takes a image and blurs in according to the defined blur_factor:
 *   - Each pixel is equal to the average value of that pixel and the nearest
 *     surrounding neighbours, which form a square around that particular pixel.
 *     The blur_factor defines how large the surrounding square is. A
 *     blur_factor of 1 are the 8 nearest neightbours.
 *
 * Parameters
 * ----------
 * imageArray: the array containing the RGB colour matrix of the image.
 * imageBlurred: the array which will contain the the blurred image data.
 *
 * Both arrays are 3 dimensional in the format (x,y,c), where x, y are
 * dimensions (in pixels) of the image and c are the number of RGB(A) colours
 * (3 for RGB and 4 for RGBA).
 */
void blur_image(int x, int y, int z, long double *imageArray, long double *imageBlurred)
{
	int xi, yi, zi;
	int xi_lower, xi_upper, yi_lower, yi_upper;
	long double sumR, sumG, sumB;
	int c, r;
	int numCells;

    for(xi = 0; xi < x; xi++){
    	// calculate the lower and upper values of pixel x indices
    	// to average
    	xi_lower = xi - blur_factor;
        xi_upper = xi + blur_factor + 1;

        // correct x indices if out of range.
        if(xi_lower < 0){
            xi_lower = 0;
        }
        if(xi_upper > x){
            xi_upper = x;
        }

    	for(yi = 0; yi < y; yi++){
	    	// calculate the lower and up values of pixel y
	    	// indices to average
	    	yi_lower = yi - blur_factor;
	        yi_upper = yi + blur_factor + 1;

	        // correct y indices if out of range.
	        if (yi_lower < 0){
	            yi_lower = 0;
	        }
	        if (yi_upper > y){
	            yi_upper = y;
	        }

	        // initialise sum on RGB values
	        sumR = 0.0;
	        sumG = 0.0;
	        sumB = 0.0;
	        // sum over RBG values for each pixel and the surrounding pixels.
	        for(c = xi_lower; c < xi_upper; c++){
	        	for (r = yi_lower; r < yi_upper; r++){
	        		sumR += imageArray[indices(x, y, z, c, r, 0)];
	        		sumG += imageArray[indices(x, y, z, c, r, 1)];
	        		sumB += imageArray[indices(x, y, z, c, r, 2)];
	        	}
	        }

	        // calculate the number of pixels from which the average is being
	        // calculated from
	        numCells = (xi_upper - xi_lower) * (yi_upper - yi_lower);
	        // calculate average RGB values for each pixel
	        imageBlurred[indices(x, y, z, xi, yi, 0)] = sumR / (long double) numCells;
	        imageBlurred[indices(x, y, z, xi, yi, 1)] = sumG / (long double) numCells;
	        imageBlurred[indices(x, y, z, xi, yi, 2)] = sumB / (long double) numCells;

    	}
    }
}

void print_array(int x, int y, int z, long double *imageArray)
{
    int i, j, k;

    for(i = 0; i < x; i++){
    	for(j = 0; j < y; j++){
    		for(k = 0; k < z; k++){
    			printf("%.18Lf ", imageArray[indices(x, y, z, i, j, k)]);
    		}
    	printf("\n");
    	}
    printf("\n");
    }
}

/*
 * Function: write_array_to_file 
 * -----------------------------
 * Writes a 3 dimensional array in c in the format (x,y,c) to a txt file. 
 * The x, y are dimensions, in pixels, of the image and c are the number of
 * RGB(A) colours (3 for RGB and 4 for RGBA).
 * Each pixel value is written on a new line, in c order (r(x0, y0), g(x0, y0)
 * b(x0, y0), r(x0, y1), g(x0, y1), b(x0, y1)...)
 * The above example only has RGB values and no A (transparency values).
 *
 * Parameters
 * ----------
 * filename: string of the txt filename
 * imageArray: long double, the c array to be written to the txt file. *
 */
void write_array_to_file(int x, int y, int z, char filename[], long double *imageArray)
{
	FILE *imagefile;
	int i, j, k;

	imagefile = fopen(filename, "w");

    for(i = 0; i < x; i++){
    	for(j = 0; j < y; j++){
    		for(k = 0; k < z; k++){
    			fprintf(imagefile, "%.18Lf\n", imageArray[indices(x, y, z, i, j, k)]);
    		}
    	}
    }

	fclose(imagefile);
}

int main(int argc, char *argv[])
{
	char *filename = malloc(strlen(argv[1]) + 4 + 1);
	char *filenameWrite = malloc(strlen(argv[1]) + 8 + 1);
	long double *imageArray;
	long double *imageBlurred;
	int x, y, z;

    sprintf(filename, "%s.txt", argv[1]);
    sprintf(filenameWrite, "%s_out.txt", argv[1]);

	x = strtol(argv[2], NULL, 10);
	y = strtol(argv[3], NULL, 10);
	z = strtol(argv[4], NULL, 10);

    imageArray = malloc(sizeof(long double) * x * y * z);
    imageBlurred = malloc(sizeof(long double) * x * y * z);

    read_file_to_array(x, y, z, filename, imageArray);
    blur_image(x, y, z, imageArray, imageBlurred);
    write_array_to_file(x, y, z, filenameWrite, imageBlurred);

    free(imageArray);
    free(imageBlurred);

    free(filename);
    free(filenameWrite);

	return 0;
}
