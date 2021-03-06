/* To do
change z to c
comments!
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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
 * img: long double, the c array where the data will be written to.
 *
 */
void read_file_to_array(int x, int y, int z, char filename[], long double *img)
{
	FILE *imagefile;

    int i, j, k;

    imagefile = fopen(filename, "r");

    // Loop through file and assign values to the image array.
    for(i = 0; i < x; i++){
    	for(j = 0; j < y; j++){
    		for(k = 0; k < z; k++){
    			fscanf(imagefile, "%Lf", &img[indices(x, y, z, i, j, k)]);
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
 * img: the array containing the RGB colour matrix of the image.
 * imageBlurred: the array which will contain the the blurred image data.
 *
 * Both arrays are 3 dimensional in the format (x,y,c), where x, y are
 * dimensions (in pixels) of the image and c are the number of RGB(A) colours
 * (3 for RGB and 4 for RGBA).
 */
void blur_image(int x, int y, int z, long double *img, long double *imageBlurred)
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
	        		sumR += img[indices(x, y, z, c, r, 0)];
	        		sumG += img[indices(x, y, z, c, r, 1)];
	        		sumB += img[indices(x, y, z, c, r, 2)];
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

void print_array(int x, int y, int z, long double *img)
{
    int i, j, k;

    for(i = 0; i < x; i++){
    	for(j = 0; j < y; j++){
    		for(k = 0; k < z; k++){
    			printf("%.18Lf ", img[indices(x, y, z, i, j, k)]);
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
 * img: long double, the c array to be written to the txt file. *
 */
void write_array_to_file(int x, int y, int z, char filename[], long double *img)
{
	FILE *imagefile;
	int i, j, k;

	imagefile = fopen(filename, "w");

    for(i = 0; i < x; i++){
    	for(j = 0; j < y; j++){
    		for(k = 0; k < z; k++){
    			fprintf(imagefile, "%.18Lf\n", img[indices(x, y, z, i, j, k)]);
    		}
    	}
    }

	fclose(imagefile);
}

/*
 * Function: calculate_scatterv_variables 
 * -------------------------------------
 * Calculate variables required for MPI_Scatterv for an image that is
 * is to be divided into horizontal strips, with each process taking
 * a strip. 
 */
void calculate_scatterv_variables(int* xLocals, int x, int y, int z, int* sendcounts, int* displacements){
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int p;
    int dispSum = 0;

    for (p = 0; p < size; p++){
        // first determine the number of rows (strip size), each
        // process takes (e.g. find the local values of x of the 
        // local images held by each process)
        xLocals[p] = x / size; // floored integer division required!
        if (p < x % size){
            if (rank == 0){
            }
            xLocals[p] += 1;
        }
        // Calculate the sendcounts and displacements variables for scatterv
        sendcounts[p] = xLocals[p] * y * z;
        displacements[p] = dispSum;
        dispSum += sendcounts[p];
    }
}

/*
 * Function: define_ghost_variables
 * --------------------------------
 * Function to determine the number of ghost cells above and below the local image.

 * Parameters
 * ----------
 * ghostsAboveSize: length of array of all ghosts above the local image.
 * xGhostsAbove: x value of the ghosts cells above the local image (e.g. the blur_factor or 0
 * if the local image is from the top of the global image).
 */
void define_ghost_variables(int y, int z, int* ghostsAboveSize, int* ghostsBelowSize, int* xGhostsAbove, int* xGhostsBelow) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (rank == 0){
        *xGhostsAbove = 0;
        *xGhostsBelow = blur_factor;
    }
    else if (rank == (size - 1)){
        *xGhostsAbove = blur_factor;
        *xGhostsBelow = 0;
    }
    else {
        *xGhostsAbove = blur_factor;
        *xGhostsBelow = blur_factor;
    }

    *ghostsAboveSize = *xGhostsAbove * y * z;
    *ghostsBelowSize = *xGhostsBelow * y * z;
}

/*
 * Function: get_ghosts
 * --------------------
 * Function to get all ghost data from neighbouring processes
 * (Halo SendRecv)
 */
void get_ghosts(long double* imgLocal, int imgLocalSize, int ghostsAboveSize, int blurFactorSize) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int ndims = 1;
    int dim[ndims];
    int period[ndims];
    int reorder;
    MPI_Comm cart_comm;
    MPI_Status status;

    int above_rank, below_rank;

    dim[0] = size;
    period[0] = 0; //bool
    reorder = 0; //bool

    // Create new Cart with topological information of
    // which processes need to speak to which other
    // processes
    MPI_Cart_create(MPI_COMM_WORLD,
                    ndims,
                    dim,
                    period,
                    reorder,
                    &cart_comm);
    MPI_Cart_shift(cart_comm, 0, 1, &above_rank, &below_rank);

    // Get ghost cell data from neighbouring processes via Halo Exchange
    MPI_Sendrecv(imgLocal + ghostsAboveSize + imgLocalSize - blurFactorSize,
                 blurFactorSize,
                 MPI_LONG_DOUBLE,
                 below_rank,
                 0,
                 imgLocal,
                 blurFactorSize,
                 MPI_LONG_DOUBLE,
                 above_rank,
                 0,
                 cart_comm,
                 &status);

    MPI_Sendrecv(imgLocal + ghostsAboveSize,
                 blurFactorSize,
                 MPI_LONG_DOUBLE,
                 above_rank,
                 1,
                 imgLocal + ghostsAboveSize + imgLocalSize,
                 blurFactorSize,
                 MPI_LONG_DOUBLE,
                 below_rank,
                 1,
                 cart_comm,
                 &status);

}

// void halo(){}

int main(int argc, char *argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char *filename = malloc(strlen(argv[1]) + 4 + 1);
	char *filenameWrite = malloc(strlen(argv[1]) + 8 + 1);

    int x, y, z, xLocal, xGhostsAbove, xGhostsBelow;
    long double *img, *imgBlurred, *imgLocal, *imgLocalBlurred;
    int imgSize, imgLocalSize, ghostsAboveSize, ghostsBelowSize, blurFactorSize;

    int *xLocals = (int*) malloc(sizeof(int) * size); // array containing sizes of x on
                                                      // each process. Array index
                                                      // corresponds to process number.
    int *sendcounts = (int*) malloc(sizeof(int) * size);
    int *displacements = (int*) malloc(sizeof(int) * size);

    // Determine variables from command line arguments.
    // Filenames
    sprintf(filename, "%s.txt", argv[1]);
    sprintf(filenameWrite, "%s_out.txt", argv[1]);
    // image dimensions
	x = strtol(argv[2], NULL, 10);
	y = strtol(argv[3], NULL, 10);
	z = strtol(argv[4], NULL, 10);

    imgSize = x * y * z; // total length of image array
    blurFactorSize = blur_factor * y * z; // total length of array containing
                                          // data which is used for blurring

    // Allocate memory for original and blurred images.
    img = (long double*) malloc(sizeof(long double) * imgSize);
    imgBlurred = (long double*) malloc(sizeof(long double) * imgSize);

    // read image in on process 0.    
    if (rank == 0){
        read_file_to_array(x, y, z, filename, img);
    }

    // calculate data required for scatterv of entire image to other processes.
    calculate_scatterv_variables(xLocals, x, y, z, sendcounts, displacements);
    xLocal = xLocals[rank];
    imgLocalSize = sendcounts[rank];

    // Define the number of ghosts cells (and amount of data) above and below each process
    define_ghost_variables(y, z, &ghostsAboveSize, &ghostsBelowSize, &xGhostsAbove, &xGhostsBelow);

    // Allocate memory for Local images (normal and blurred) (ghost cell data also contained in these arrays)
    imgLocal = (long double*) malloc(sizeof(long double) * (imgLocalSize + ghostsBelowSize + ghostsAboveSize));
    imgLocalBlurred = (long double*) malloc(sizeof(long double) * (imgLocalSize + ghostsBelowSize + ghostsAboveSize));

    // Scatter parts of full image on process 0 to all processes
    MPI_Scatterv(img,
                 sendcounts,
                 displacements,
                 MPI_LONG_DOUBLE,
                 imgLocal + ghostsAboveSize,
                 imgLocalSize,
                 MPI_LONG_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // Get ghost cell data
    get_ghosts(imgLocal, imgLocalSize, ghostsAboveSize, blurFactorSize);

    // Blur the image
    blur_image(xLocal + xGhostsBelow + xGhostsAbove, y, z,
               imgLocal,
               imgLocalBlurred);

    // Gather all local blurred images onto process 0 to form full blurred image
    MPI_Gatherv(imgLocalBlurred + ghostsAboveSize,
               imgLocalSize,
               MPI_LONG_DOUBLE,
               imgBlurred,
               sendcounts,
               displacements,
               MPI_LONG_DOUBLE,
               0,
               MPI_COMM_WORLD);

    // Write blurred image data to file
    if (rank == 0){
        write_array_to_file(x, y, z, filenameWrite, imgBlurred);
        printf("saved\n");
    }

    free(filename);
    free(filenameWrite);
    free(img);
    free(imgBlurred);
    free(imgLocal);
    free(imgLocalBlurred);
    free(xLocals);
    free(sendcounts);
    free(displacements);

    MPI_Finalize();

	return 0;
}
