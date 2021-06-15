#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <fftw3.h>
// #include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>




void fill_space_arrays(double** x, int** k, const long int* N, int local_N, int local_N_start) {

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
}



void print_space_arrays(double** x, int** k, const long int* N, int local_N, int calling_rank, MPI_Comm Comm) {

	// Gather data to see if allocated correctly
	int* k0    = (int* )malloc(sizeof(int) * N[0]);
	double* x0 = (double* )malloc(sizeof(double) * N[0]);
	for (int i = 0; i < N[0]; ++i) {
		k0[i] = 0;
		x0[i] = 0.0;
	}
	MPI_Gather(k[0], local_N, MPI_INT, k0, local_N, MPI_INT, 0, Comm);
	MPI_Gather(x[0], local_N, MPI_DOUBLE, x0, local_N, MPI_DOUBLE, 0, Comm);

	if(!(calling_rank)) {
		for (int i = 0; i < N[0]; ++i) {
			printf("x[%d]: %5.16lf\txMPI[%d]: %5.16lf \t k[%d]: %d\n", i,  (double) i * 2.0 * M_PI / (double) N[0], i, x0[i], i, k0[i]);
		}
		printf("\n\n");
	}

	MPI_Barrier(Comm);


	free(k0);
	free(x0);
}





int main(int argc, char** argv) {


	// Initalize params
	const long int N_0 = 4;
	const long int N_1 = 4;
	const long int N[2] = {N_0, N_1};

	int tmp;
	int indx;

	int* kx = (int* )malloc(sizeof(int) * N[0]);
	int* ky = (int* )malloc(sizeof(int) * (N[1] / 2 + 1));

	printf("\nN / 3: %d\n\n", (int)ceil(N[0] / 3));

	for (int i = 0; i < N[0]; ++i)
	{
		if (i < N[0] / 2 + 1) {
			kx[i] = i;
			ky[i] = i;  
			printf("kx[%d]: %+d\t ky[%d]: %+d\n", i, kx[i], i, ky[i]);
		}
		else {
			kx[i] = i - N[0];
			printf("kx[%d]: %+d\n", i, kx[i]);
		}
	}
	printf("\n\n");


	for (int i = 0; i < N[0]; ++i) {
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			// if ((abs(kx[i]) < ceil((double)N[0] / 3.0)) && (abs(ky[i]) < ceil((double)N[1] / 3.0))) {
				printf("(%d, %d)\t", kx[i], kx[j]);
			// }
			// else {
			// 	printf("(%d, %d)\t", 0, 0);
			// }
		}
		printf("\n");
	}
	printf("\n\n");

	for (int i = 0; i < N[0]; ++i) {
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			if ((abs(kx[i]) < (int) ceil(N[0] / 3)) && (abs(ky[j]) < (int) ceil(N[1] / 3))) {
				printf("(%d, %d)\t", kx[i], kx[j]);
			}
			else {
				printf("(%d, %d)\t", 0, 0);
			}
		}
		printf("\n");
	}
	printf("\n\n");








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


	double* x[2];
	int* k[2];
	x[0] = (double* )malloc(sizeof(double) * local_N);
	x[1] = (double* )malloc(sizeof(double) * N[1]);
	k[0] = (int* )malloc(sizeof(int) * local_N);
	k[1] = (int* )malloc(sizeof(int) * N[1] / 2 + 1);
	// fill_space_arrays(x, k, N, local_N, local_N_start);
	// print_space_arrays(x, k, N, local_N, calling_rank, MPI_COMM_WORLD);



	


	return 0;
}