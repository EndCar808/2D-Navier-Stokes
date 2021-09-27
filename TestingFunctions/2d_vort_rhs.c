#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <fftw3.h>
// #include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>



void nonlinear_rhs(fftw_complex* w_hat, fftw_complex* dw_hat_dt, fftw_plan* fftw_plan_r2c, fftw_plan* fftw_plan_c2r, double* u, double*v, double* w, const long int* N, ptrdiff_t alloc_local, ptrdiff_t local_N, int** k) {

	// Initialize variables
	int tmp;
	int indx;
	fftw_complex k_sqr;


	// Allocate memory
	double* dw_dx = (double *)malloc(sizeof(double) * 2 * alloc_local);
	double* dw_dy = (double *)malloc(sizeof(double) * 2 * alloc_local);
	double* dw_dt = (double *)malloc(sizeof(double) * 2 * alloc_local);
	fftw_complex* u_hat    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) *  alloc_local);
	fftw_complex* v_hat    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) *  alloc_local);
	fftw_complex* dwhat_dx = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) *  alloc_local);
	fftw_complex* dwhat_dy = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) *  alloc_local);
	// Initialize the real arrays
	for (int i = 0; i < local_N; ++i) {
		for (int j = 0; j < N[1]; ++j) {
			dw_dx[i * (N[1] + 2) + j] = 0.0;
			dw_dy[i * (N[1] + 2) + j] = 0.0;
			dw_dt[i * (N[1] + 2) + j] = 0.0;
		}
	}

	// Compute (-\Delta)^-1 \omega - i.e., u_hat = -I kx/|k|^2 \omegahat_k, v_hat = -I ky/|k|^2 \omegahat_k
	for (int i = 0; i < local_N; ++i) {	
		tmp = i * (N[1] / 2 + 1);
		for (int j = 0; j < (N[1] / 2 + 1); ++j) {	
			indx = tmp + j;

			// Prefactor = I / |k|^2 denominator
			k_sqr = I / (double)(k[0][i] * k[0][i] + k[1][j] * k[1][j] + (double) 1E-50);

			// u_hat and v_hat
			u_hat[indx] = k_sqr * ((double)k[1][j]) * w_hat[indx];
			v_hat[indx] = - k_sqr * ((double)k[0][i]) * w_hat[indx];
		}
	}


	// Transform Fourier velocities to Real space
	fftw_mpi_execute_dft_c2r((*fftw_plan_c2r), u_hat, u);
	fftw_mpi_execute_dft_c2r((*fftw_plan_c2r), v_hat, v);


	// Compute \nabla\omega - i.e., d\omegahat_dx = -I kx \omegahat_k, d\omegahat_dy = -I ky \omegahat_k
	for (int i = 0; i < local_N; ++i){	
		tmp = i * (N[1] / 2 + 1);
		for (int j = 0; j < (N[1] / 2 + 1); ++j) {	
			indx = tmp + j;

			// u_hat and v_hat
			dwhat_dx[indx] = -I * (k[0][i]) * w_hat[indx];
			dwhat_dy[indx] = -I * (k[1][j]) * w_hat[indx];
		}
	}



	// Transform Fourier velocities to Real space
	fftw_mpi_execute_dft_c2r((*fftw_plan_c2r), dwhat_dx, dw_dx);
	fftw_mpi_execute_dft_c2r((*fftw_plan_c2r), dwhat_dy, dw_dy);



	// Perform the multiplication in real space
	for (int i = 0; i < local_N; ++i) {
		tmp = i * (N[1] + 2);
		for (int j = 0; j < N[1]; ++j) {
			indx = tmp + j;

			// compute the nonlinear term in realspace
			dw_dt[indx] = -(u[indx] * dw_dx[indx] + v[indx] * dw_dy[indx]); 
		}
	}


	// Transform Fourier velocities to Real space
	fftw_mpi_execute_dft_r2c((*fftw_plan_r2c), dw_dt, dw_hat_dt);


	// Apply dealiasing


	// Free tmp memory
	free(dw_dt);
	free(dw_dy);
	free(dw_dx);
	fftw_free(dwhat_dx);
	fftw_free(dwhat_dy);
	fftw_free(u_hat);
	fftw_free(v_hat);
}



void nonlinear_rhs_batch(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u, double* w, int** k, fftw_plan* fftw_batch_r2c, fftw_plan* fftw_batch_c2r, const long int* N, int local_N) {


	int tmp, indx;
	fftw_complex k_sqr;

	double* vel = (double* )malloc(sizeof(double) * local_N * (N[1] + 2));


	// Compute (-\Delta)^-1 \omega - i.e., u_hat = -I kx/|k|^2 \omegahat_k, v_hat = -I ky/|k|^2 \omegahat_k
	for (int i = 0; i < local_N; ++i) {
		tmp = i * (N[1] / 2 + 1);
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			indx = tmp + j;

			// denominator
			k_sqr = I / (double) (k[0][i] * k[0][i] + k[1][j] * k[1][j] + (double)1E-50);		

			// Fill array
			dw_hat_dt[2 * (indx) + 0] = k_sqr * ((double) k[1][j]) * w_hat[indx];
			dw_hat_dt[2 * (indx) + 1] = -k_sqr * ((double) k[0][j]) * w_hat[indx];
		}
	}


	// Batch transform both fourier velocites to real space
	fftw_mpi_execute_dft_c2r((*fftw_batch_c2r), dw_hat_dt, u);


	// Compute \nabla\omega - i.e., d\omegahat_dx = -I kx \omegahat_k, d\omegahat_dy = -I ky \omegahat_k
	for (int i = 0; i < local_N; ++i) {
		tmp = i * (N[1] / 2 + 1);
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			indx = tmp + j;

			// Fill array
			dw_hat_dt[2 * indx + 0] = - I * ((double) k[0][i]) * w_hat[indx];
			dw_hat_dt[2 * indx + 1] = - I * ((double) k[1][j]) * w_hat[indx]; 
		}
	}


	// Batch transform both fourier velocites to real space
	fftw_mpi_execute_dft_c2r((*fftw_batch_c2r), dw_hat_dt, w);


	// Perform the multiplication in real space
	for (int i = 0; i < local_N; ++i) {
		tmp = i * (N[1] + 2);
		for (int j = 0; j < N[1]; ++j) {
			indx = tmp + j; 
 			
 			// Perform multiplication
 			vel[indx] = - (u[2 * indx + 0] * w[2 * indx + 0] + u[2 * indx + 1] * w[2 * indx + 1]);
 		}
 	}


 	// Transform Fourier velocities to Real space
 	fftw_mpi_execute_dft_r2c((*fftw_batch_r2c), vel, dw_hat_dt);


 	// Apply dealiasing

 	free(vel);

}


void fill_w_hat(double* u, double* v, double** x, int** k, fftw_complex* u_hat, fftw_complex* v_hat, fftw_complex* w_hat, fftw_plan* fftw_2d_dft_r2c, fftw_plan* fftw_2d_dft_c2r, const long int* N, int local_N, int calling_rank) {

	int tmp, indx;

	// Fill arrays Real Space arrays with Taylor Green IC
	for (int i = 0; i < local_N; ++i) {
		// tmp indx
		tmp = i * (N[1] + 2);
		for (int j = 0; j < N[1]; ++j) {
			// actual indx
			indx = tmp + j;

			// Fill arrays
			u[indx] = cos(x[0][i]) * sin(x[1][j]);
			v[indx] = -sin(x[0][i]) * cos(x[1][j]);
		}
	}
	// print_veloc(u, v, N, calling_rank, local_N, MPI_COMM_WORLD);

	
	// Transform to Fourier Space
	fftw_mpi_execute_dft_r2c((*fftw_2d_dft_r2c), u, u_hat);
	fftw_mpi_execute_dft_r2c((*fftw_2d_dft_r2c), v, v_hat);
	// print_fourier_veloc(u_hat, v_hat, N, calling_rank, local_N, MPI_COMM_WORLD);


	// fftw_mpi_execute_dft_c2r((*fftw_2d_dft_c2r), u_hat, u);
	// fftw_mpi_execute_dft_c2r((*fftw_2d_dft_c2r), v_hat, v);
	// for (int i = 0; i < local_N; ++i) {
	// 	for (int j = 0; j < N[1]; ++j) {
	// 		u[i * (N[1] + 2) + j] /= (N[0] * N[1]);
	// 		v[i * (N[1] + 2) + j] /= (N[0] * N[1]); 
	// 	}
	// }
	// print_veloc(u, v, N, calling_rank, local_N, MPI_COMM_WORLD);


	// Fill w_hat
	for (int i = 0; i < local_N; ++i) {
		// tmp indx
		tmp = i * (N[1] / 2 + 1);
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			// actual indx
			indx = tmp + j;

			// Fill array
			w_hat[indx] = I * (k[0][i] * v_hat[indx] - k[1][j] * u_hat[indx]);
		}
	}

	// Print w_hat
	fftw_complex* w_hat0 = (fftw_complex* )malloc(sizeof(fftw_complex) * N[0] * (N[1] / 2 + 1));
	MPI_Gather(w_hat, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, w_hat0, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	if ( !(calling_rank) ) {
		for (int i = 0; i < N[0]; ++i)	{
			for (int j = 0; j < N[1] / 2 + 1; ++j) {
				printf("what[%ld]: %5.16lf %5.16lf I\t", i * (N[1] / 2 + 1) + j, creal(w_hat0[i * (N[1] / 2 + 1) + j]), cimag(w_hat0[i * (N[1] / 2 + 1) + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	free(w_hat0);

}

void fill_w_hat_batch(fftw_complex* w_hat, double* u_batch, fftw_complex* u_hat_batch, fftw_plan* fftw_plan_batch_r2c, fftw_plan* fftw_plan_batch_c2r, double** x, int** k, const long int* N, int local_N, int calling_rank) {

	int tmp, indx;

	// Fill the velocity array with the Taylor Green vortex IC
	for (int i = 0; i < local_N; ++i) {
		tmp = i * (N[1] + 2);
		for (int j = 0; j < N[1]; ++j) {
			indx = 2 * (tmp + j);

			// Fill the velocities
			u_batch[indx + 0] = cos(x[0][i]) * sin(x[1][j]);
			u_batch[indx + 1] = -sin(x[0][i]) * cos(x[1][j]);		
		}
	}


	// Transform velocities to Fourier space
	fftw_mpi_execute_dft_r2c((*fftw_plan_batch_r2c), u_batch, u_hat_batch);



	// Fill the Fourier space vorticity array
	for (int i = 0; i < local_N; ++i) {	
		tmp = i * (N[1] / 2 + 1);
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			indx = tmp + j;

			// Fill vorticity
			w_hat[indx] = I * (k[0][i] * u_hat_batch[2 * (indx) + 1] - k[1][j] * u_hat_batch[2 * (indx) + 0]);
		}
	}
	// Print w_hat
	fftw_complex* w_hat0 = (fftw_complex* )malloc(sizeof(fftw_complex) * N[0] * (N[1] / 2 + 1));
	MPI_Gather(w_hat, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, w_hat0, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	if ( !(calling_rank) ) {
		for (int i = 0; i < N[0]; ++i)	{
			for (int j = 0; j < N[1] / 2 + 1; ++j) {
				printf("what[%ld]: %5.16lf %5.16lf I\t", i * (N[1] / 2 + 1) + j, creal(w_hat0[i * (N[1] / 2 + 1) + j]), cimag(w_hat0[i * (N[1] / 2 + 1) + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}
	free(w_hat0);
}


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

void print_veloc(double* u, double* v, const long int* N, int calling_rank, int local_N, MPI_Comm Comm) {

	double* u0 = (double* )fftw_malloc(sizeof(double) * N[0] * (N[1] + 2));
	double* v0 = (double* )fftw_malloc(sizeof(double) * N[0] * (N[1] + 2));
	for (int i = 0; i < N[0]; ++i) {
		for (int j = 0; j < N[1] + 2; ++j) {
			u0[i * (N[1] + 2) + j] = 0.0;
			v0[i * (N[1] + 2) + j] = 0.0;
		}
	}
	MPI_Gather(u, local_N * (N[1] + 2), MPI_DOUBLE, u0, local_N * (N[1] + 2), MPI_DOUBLE, 0, Comm);
	MPI_Gather(v, local_N * (N[1] + 2), MPI_DOUBLE, v0, local_N * (N[1] + 2), MPI_DOUBLE, 0, Comm);

	if (!(calling_rank)) {
		for (int i = 0; i < N[0]; ++i)
		{
			for (int j = 0; j < N[1]; ++j)
			{
				printf("u0[%ld]: %+5.16lf\t", i * (N[1]) + j, u0[i * (N[1] + 2) + j]);
			}
			printf("\n");
		}
		printf("\n\n");


		for (int i = 0; i < N[0]; ++i)
		{
			for (int j = 0; j < N[1]; ++j)
			{
				printf("v0[%ld]: %+5.16lf\t", i * (N[1]) + j, v0[i * (N[1] + 2) + j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	MPI_Barrier(Comm);

	fftw_free(u0);
	fftw_free(v0);
}

void print_fourier_veloc(fftw_complex* u_hat, fftw_complex* v_hat, const long int* N, int calling_rank, int local_N, MPI_Comm Comm) {

	fftw_complex* u_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * N[0] * (N[1] / 2 + 1));
	fftw_complex* v_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * N[0] * (N[1] / 2 + 1));
	for (int i = 0; i < N[0]; ++i) {
		for (int j = 0; j < N[1] / 2 + 1; ++j) {
			u_hat0[i * (N[1] / 2 + 1) + j] = 0.0 + 0.0 * I;
			v_hat0[i * (N[1] / 2 + 1) + j] = 0.0 + 0.0 * I;
		}
	}
	MPI_Gather(u_hat, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, u_hat0, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, 0, Comm);
	MPI_Gather(v_hat, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, v_hat0, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, 0, Comm);

	if (!(calling_rank)) {
		for (int i = 0; i < N[0]; ++i)
		{
			for (int j = 0; j < (N[1] / 2 + 1); ++j)
			{
				printf("u0[%ld]: %+5.16lf %+5.16lfI\t", i * ((N[1] / 2 + 1)) + j, creal(u_hat0[i * (N[1] / 2 + 1) + j]), cimag(u_hat0[i * (N[1] / 2 + 1) + j])); // /(N[0]*N[1])
			}
			printf("\n");
		}
		printf("\n\n");


		for (int i = 0; i < N[0]; ++i)
		{
			for (int j = 0; j < (N[1] / 2 + 1); ++j)
			{
				printf("v0[%ld]: %+5.16lf %+5.16lfI\t", i * (N[1] / 2 + 1) + j, creal(v_hat0[i * (N[1] / 2 + 1) + j]), cimag(v_hat0[i * (N[1] / 2 + 1) + j])); // /(N[0]*N[1])
			}
			printf("\n");
		}
		printf("\n\n");
	}
	MPI_Barrier(Comm);

	fftw_free(u_hat0);
	fftw_free(v_hat0);
}


int main(int argc, char** argv) {


	// Initalize params
	const long int N_0 = 8;
	const long int N_1 = 8;
	const long int N[2] = {N_0, N_1};

	int tmp;
	int indx;


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


	// Allocate memory
	double* u     = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* v     = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* w     = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* dw_dt = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* dw_dx = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* dw_dy = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	fftw_complex* u_hat     = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* v_hat     = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* w_hat     = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* dw_hat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* dw_hat_dx = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* dw_hat_dy = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);

	double* x[2];
	int* k[2];
	x[0] = (double* )malloc(sizeof(double) * local_N);
	x[1] = (double* )malloc(sizeof(double) * N[1]);
	k[0] = (int* )malloc(sizeof(int) * local_N);
	k[1] = (int* )malloc(sizeof(int) * N[1] / 2 + 1);
	fill_space_arrays(x, k, N, local_N, local_N_start);
	print_space_arrays(x, k, N, local_N, calling_rank, MPI_COMM_WORLD);



	// Create FFTW Plans
	fftw_plan fftw_2d_dft_r2c, fftw_2d_dft_c2r;
	fftw_2d_dft_r2c = fftw_mpi_plan_dft_r2c_2d(N[0], N[1], w, w_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	fftw_2d_dft_c2r = fftw_mpi_plan_dft_c2r_2d(N[0], N[1], w_hat, w, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);



	// Fill w_hat with Taylor Green Vortex IC
	fill_w_hat(u, v, x, k, u_hat, v_hat, w_hat, &fftw_2d_dft_r2c, &fftw_2d_dft_c2r, N, local_N, calling_rank);
	

	// Call the RHS
	nonlinear_rhs(w_hat, dw_hat_dt, &fftw_2d_dft_r2c, &fftw_2d_dft_c2r, u, v, w, N, alloc_local, local_N, k);


	fftw_complex* dw_hat_dt_0 = (fftw_complex* )malloc(sizeof(fftw_complex) * N[0] * (N[1] / 2 + 1));
	MPI_Gather(dw_hat_dt, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, dw_hat_dt_0, local_N * (N[1] / 2 + 1), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	if ( !(calling_rank) ) {
		for (int i = 0; i < N[0]; ++i)	{
			for (int j = 0; j < N[1] / 2 + 1; ++j) {
				printf("Solv[%ld]: %5.16lf %5.16lf I\t", i * (N[1] / 2 + 1) + j, creal(dw_hat_dt_0[i * (N[1] / 2 + 1) + j]), cimag(dw_hat_dt_0[i * (N[1] / 2 + 1) + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}


	///	
	///	
	///	BATCH VERSION
	///	
	///	
	
	// Define dimensions for batch local function
	const ptrdiff_t NBatch[2] = {N_0, N_1 / 2 + 1};

	// Define a batch local scheme that performs 2 2d transforms of size N_0 * N_1 using default block dims
	ptrdiff_t alloc_local_batch = fftw_mpi_local_size_many(2, NBatch, (ptrdiff_t) 2, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &local_N, &local_N_start);
			

	// Allocate batch memory arrays
	double* u_batch     = (double *)malloc(sizeof(double) * 2 * alloc_local_batch);
	double* w_batch     = (double *)malloc(sizeof(double) * 2 * alloc_local_batch);
	fftw_complex* u_hat_batch = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local_batch);
	fftw_complex* w_hat_batch = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local_batch);


	// FFTW batch plans
	fftw_plan fftw_2d_dft_batch_c2r, fftw_2d_dft_batch_r2c;
	fftw_2d_dft_batch_r2c = fftw_mpi_plan_many_dft_r2c(2, N, (ptrdiff_t) 2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, u_batch, u_hat_batch, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	fftw_2d_dft_batch_c2r = fftw_mpi_plan_many_dft_c2r(2, N, (ptrdiff_t) 2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, u_hat_batch, u_batch, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	


	// Fill w_hat
	fill_w_hat_batch(w_hat, u_batch, u_hat_batch, &fftw_2d_dft_batch_r2c, &fftw_2d_dft_batch_c2r, x, k, N, local_N, calling_rank);


	// Call the nonlinear term
	nonlinear_rhs_batch(w_hat, w_hat_batch, u_batch, w_batch, k, &fftw_2d_dft_batch_r2c, &fftw_2d_dft_batch_c2r, N, local_N);



	// Free memory
	fftw_free(u);
	fftw_free(v);
	fftw_free(w);
	for (int i = 0; i < 2; ++i)
	{
		free(k[i]);
		free(x[i]);
	}
	fftw_free(u_hat);
	fftw_free(v_hat);
	fftw_free(dw_hat_dt);
	fftw_free(w_hat);

	// Destroy FFTW plans
	fftw_destroy_plan(fftw_2d_dft_r2c);
	fftw_destroy_plan(fftw_2d_dft_c2r);

	// Cleanup FFTW MPI interface
	fftw_mpi_cleanup();    // Calls the serial fftw_cleanup function also

	// Exit MPI scetion
	MPI_Finalize();






	return 0;
}