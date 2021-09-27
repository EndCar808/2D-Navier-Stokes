#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <fftw3.h>
// #include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>



int main(int argc, char** argv) {

	// Initalize params
	const long int N_0 = 64;
	const long int N_1 = 64;
	const long int N[2] = {N_0, N_1};


	// MPI vars
	int num_procs;		// Number of processes
	int calling_rank;   // rank of master process

	// Local FFTW transform array variables
	ptrdiff_t local_N;			// this is the size of the first local dimension
	ptrdiff_t local_N_start;	// this is the position for each local array


	// Initialize MPI section
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);      // Get the number of processes in the communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &calling_rank);   // Get the rank of the processes
	if ( !(calling_rank) ) {
		printf("\nTotal number of MPI tasks running: %d\nCalled from process %d\n\n", num_procs, calling_rank) ;
	}

	// Initialize FFTW MPI interface
	fftw_mpi_init();


	// Get local array sizes for the transform
	ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N[0], (N[1] / 2 + 1), MPI_COMM_WORLD, &local_N, &local_N_start);

	// Allocate local arrays
	double* input        = (double* )malloc(sizeof(double) * 2 * alloc_local);
	fftw_complex* output = (fftw_complex*)malloc(sizeof(fftw_complex) * alloc_local);

	// Allocate Real and Fourier space variables
	double* x[2];
	int* k[2];
	x[0] = (double* )malloc(sizeof(double) * local_N);
	x[1] = (double* )malloc(sizeof(double) * N[1]);
	k[0] = (int* )malloc(sizeof(int) * local_N);
	k[1] = (int* )malloc(sizeof(int) * N[1] / 2 + 1);

	// Real and Fourier space arrays
	// Fill the first dirction 
	int j = 0;
	for (int i = 0; i < N[0]; ++i) {
		if((i >= local_N_start) && ( i < local_N_start + local_N)) { // Ensure each process only writes to its local array slice
			x[0][j] = (double) i * 2.0 * M_PI / (double) N[0];
			j++;
		}
	}
	j = 0;
	for (int i = 0; i < local_N; ++i) {
		if (local_N_start + i <= N[0] / 2) {
			k[0][j] = local_N_start + i;
			j++;
		}
		else if (local_N_start + i > N[0] / 2) {
			k[0][j] = local_N_start + i - N[0];
			j++;
		}
	}
	// Fill the second direction
	for (int i = 0; i < N[1]; ++i) {
		if (i < N[1]/2 + 1) {
			k[1][i] = i;
		}
		x[1][i] = (double) i * 2.0 * M_PI / (double) N[0];
	}


	// Gather data to see if allocated correctly
	int* k0    = (int* )malloc(sizeof(int) * N[0]);
	double* x0 = (double* )malloc(sizeof(double) * N[0]);
	MPI_Gather(k[0], local_N, MPI_INT, k0, local_N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(x[0], local_N, MPI_DOUBLE, x0, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(!(calling_rank)) {
		for (int i = 0; i < N[0]; ++i)
		{
			printf("x[%d]: %5.16lf\txMPI[%d]: %5.16lf \t k[%d]: %d\n", i,  (double) i * 2.0 * M_PI / (double) N[0], i, x0[i], i, k0[i]);
		}
		printf("\n");
		for (int i = 0; i < N[0]; ++i)
		{
			printf("x[%d]: %5.16lf\n", i,  x[1][i]);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);



	// Create FFTW plan
	fftw_plan fftw_2d_dft_r2c, fftw_2d_dft_c2r;
	fftw_2d_dft_r2c = fftw_mpi_plan_dft_r2c_2d(N[0], (N[1]/2 + 1), input, output, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	fftw_2d_dft_c2r = fftw_mpi_plan_dft_c2r_2d(N[0], (N[1]/2 + 1), output, input, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);


	// Fill arrays with data
	for (int i = 0; i < local_N; ++i) {
		for (int j = 0; j < N[1]; ++j) {
			input[i * N[1] + j] = cos(x[0][i]) * sin(x[1][j]);
			if (j < N[1]/ 2 + 1){
				output[i * (N[1]/ 2 + 1) + j] = 0.0 + 0.0 * I;
			} 
		}
	}
	
	// Execute transform
	fftw_mpi_execute_dft_r2c(fftw_2d_dft_r2c, input, output);

	// Cleanup FFTW MPI interface
	fftw_mpi_cleanup();    // Calls the serial fftw_cleanup function also

	// Exit MPI scetion
	MPI_Finalize();


	return 0;
}