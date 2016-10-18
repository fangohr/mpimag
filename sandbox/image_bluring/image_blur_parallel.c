/* To do
change z to c
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

    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

	char *filename = malloc(strlen(argv[1]) + 4 + 1);
	char *filenameWrite = malloc(strlen(argv[1]) + 8 + 1);

    int x, y, z, p;
	long double *imageArray;
	// long double *imageBlurred;
    int *x_locals = (int*) malloc(sizeof(int) * size);
    int *sendcounts = (int*) malloc(sizeof(int) * size);
    int *displacements = (int*) malloc(sizeof(int) * size);
    int dispSum = 0;
    int x_local, recvCount;
    long double *image_local;
    long double *ghosts_above;
    long double *ghosts_below;
    // int ghostsBelowCount;
    // int ghostsAboveCount;
    long double *image_local_ghosts;
    long double *image_local_blurred;
    long double *image_blurred;
    int x_local_ghosts;
    int ghostsAboveCount;
    int blurCount;

    // variables for car
    int ndims = 1;
    int dim[ndims];
    int period[ndims];
    int reorder;
    MPI_Comm cart_comm;
    int above_rank, below_rank;

    sprintf(filename, "%s.txt", argv[1]);
    sprintf(filenameWrite, "%s_out.txt", argv[1]);

	x = strtol(argv[2], NULL, 10);
	y = strtol(argv[3], NULL, 10);
	z = strtol(argv[4], NULL, 10);

    sprintf(filename, "%s.txt", argv[1]);

    // read image in on process 0.  
    imageArray = (long double*) malloc(sizeof(long double) * x * y * z);
    image_blurred = (long double*) malloc(sizeof(long double) * x * y * z);
  
    if (rank == 0){
        read_file_to_array(x, y, z, filename, imageArray);
    }

    for (p = 0; p < size; p++){ 
        x_locals[p] = x / size; // floored integer division required!
        if (p < x % size){
            if (rank == 0){
            }
            x_locals[p] += 1;
        }
        sendcounts[p] = x_locals[p] * y * z;
        displacements[p] = dispSum;
        dispSum += sendcounts[p];
    }

    x_local = x_locals[rank];
    recvCount = sendcounts[rank];
    blurCount = blur_factor * y * z;
    // printf("%d ", sendcounts[rank]);

    image_local = (long double*) malloc(sizeof(long double) * recvCount);

    MPI_Scatterv(imageArray,
                 sendcounts,
                 displacements,
                 MPI_LONG_DOUBLE,
                 image_local,
                 recvCount,
                 MPI_LONG_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // if (rank == 2){
    //     write_array_to_file(x_local, y, z, filenameWrite, image_local);
    //     printf("saved\n");
    // }

    if (rank > 0){
        ghosts_above = (long double*) malloc(sizeof(long double) * blurCount);
    }
    if (rank < (size - 1)){
        ghosts_below = (long double*) malloc(sizeof(long double) * blurCount);
        // ghosts_below = NULL;
    }

    ndims = 1;
    dim[0] = size;
    period[0] = 0; //bool
    reorder = 0; //bool
    
    MPI_Cart_create(MPI_COMM_WORLD,
                    ndims,
                    dim,
                    period,
                    reorder,
                    &cart_comm);

    MPI_Cart_shift(cart_comm, 0, 1, &above_rank, &below_rank);

    MPI_Sendrecv(&image_local[indices(x_local, y, z, x_local - blur_factor, 0, 0)],
                 blurCount,
                 MPI_LONG_DOUBLE,
                 below_rank,
                 0,
                 ghosts_above,
                 blurCount,
                 MPI_LONG_DOUBLE,
                 above_rank,
                 0,
                 cart_comm,
                 &status);

    MPI_Sendrecv(image_local,
                 blurCount,
                 MPI_LONG_DOUBLE,
                 above_rank,
                 1,
                 ghosts_below,
                 blurCount,
                 MPI_LONG_DOUBLE,
                 below_rank,
                 1,
                 cart_comm,
                 &status);

    if (rank == 0){
        image_local_ghosts = (long double*) malloc(sizeof(long double) * (blurCount + recvCount));
        image_local_blurred = (long double*) malloc(sizeof(long double) * (blurCount + recvCount));
        x_local_ghosts = x_local + blur_factor;
        ghostsAboveCount = 0;
        memcpy(image_local_ghosts,
               image_local,
               recvCount * sizeof(long double));
        memcpy(image_local_ghosts + recvCount,
               ghosts_below,
               blurCount * sizeof(long double));
    }
    else if (rank == (size - 1)){
        image_local_ghosts = (long double*) malloc(sizeof(long double) * (blurCount + recvCount));
        image_local_blurred = (long double*) malloc(sizeof(long double) * (blurCount + recvCount));
        x_local_ghosts = x_local + blur_factor;
        ghostsAboveCount = blur_factor * y * z;
        memcpy(image_local_ghosts,
               ghosts_above,
               blurCount * sizeof(long double));
        memcpy(image_local_ghosts + blurCount,
               image_local,
               recvCount * sizeof(long double));
    }
    else {
        image_local_ghosts = (long double*) malloc(sizeof(long double) * (blurCount + blurCount + recvCount));
        image_local_blurred = (long double*) malloc(sizeof(long double) * (blurCount + blurCount + recvCount));
        x_local_ghosts = x_local + blur_factor + blur_factor;
        ghostsAboveCount = blur_factor * y * z;
        memcpy(image_local_ghosts,
               ghosts_above,
               blurCount * sizeof(long double));
        memcpy(image_local_ghosts + blurCount,
               image_local,
               recvCount * sizeof(long double));
        memcpy(image_local_ghosts + blurCount + recvCount,
               ghosts_below,
               blurCount * sizeof(long double));
    }
    // if (rank == 1){
    //     write_array_to_file(blur_factor, y, z, filenameWrite, ghosts_above);
    //     printf("saved\n");
    // }

    blur_image(x_local_ghosts, y, z,
               image_local_ghosts,
               image_local_blurred);

    // if (rank == 1){
    //     write_array_to_file(x_local_ghosts, y, z, filenameWrite, image_local_blurred);
    //     printf("saved\n");
    // }

    MPI_Gatherv(image_local_blurred + ghostsAboveCount,
               recvCount,
               MPI_LONG_DOUBLE,
               image_blurred,
               sendcounts,
               displacements,
               MPI_LONG_DOUBLE,
               0,
               MPI_COMM_WORLD);

    if (rank == 0){
        write_array_to_file(x, y, z, filenameWrite, image_blurred);
        printf("saved\n");
    }

    // ghosts_below = NULL;
    // free(ghosts_below);

    free(imageArray);
    free(filename);
    free(filenameWrite);
    free(x_locals);
    free(sendcounts);
    free(displacements);
    free(image_local);
    free(image_local_ghosts);

    MPI_Finalize();

	return 0;
}
