#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>



int main(int argc, char **argv)
{
	// Initialize variables

	int rank, num_procs;

	//////////////////////////////////
	// Initialize MPI section
	MPI_Init(&argc, &argv);
	//////////////////////////////////
	
	// Get the number of active processes and their rank and print to screen
	MPI_Comm_size(MPI_COMM_WORLD, &(num_procs));      
	MPI_Comm_rank(MPI_COMM_WORLD, &(rank));  
	if ( !(rank) ) {
		printf("\nTotal number of MPI tasks running: %d\n\n", num_procs);
	}

	// Initialize FFTW MPI interface - must be called after MPI_Init but before anything else in FFTW
	fftw_mpi_init();




	////////////
	/// System Vars
	int tmp, indx;
	int Nx        = 8;
	int Ny        = 8;
	int Nyf       = (int) Ny / 2 + 1;
	const ptrdiff_t N[2]      = {Nx, Ny};
	const ptrdiff_t NBatch[2] = {Nx, Nyf};
	ptrdiff_t local_Nx, alloc_local, local_Nx_start;
	ptrdiff_t local_Nx_batch, alloc_local_batch, local_Nx_start_batch;

	///////////////
	/// Get Memory Size using FFTW
	alloc_local       = fftw_mpi_local_size_2d(Nx, Nyf, MPI_COMM_WORLD, &(local_Nx), &(local_Nx_start));
	alloc_local_batch = fftw_mpi_local_size_many((int)2, NBatch, (ptrdiff_t) 2, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &(local_Nx_batch), &(local_Nx_start_batch));
		
	// Check local and batch local parameters are the same
	printf("Rank: %d \t local_Nx: %d \t local_NxB: %d \t local_start: %d \t local_NxB: %d\n", rank, local_Nx, local_Nx_batch, local_Nx_start, local_Nx_start_batch);

	///////////////
	/// Allocate Memory
	double* u 		= (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* v 		= (double* )fftw_malloc(sizeof(double) * 2 * alloc_local);
	double* u_batch = (double* )fftw_malloc(sizeof(double) * 2 * alloc_local_batch);
	fftw_complex* u_h 		= (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* v_h 		= (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local);
	fftw_complex* u_h_batch = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * alloc_local_batch);
	int* k[2];
	k[0] = (int * )fftw_malloc(sizeof(int) * Nx);
	k[1] = (int * )fftw_malloc(sizeof(int) * Nyf);
	


	////////////////
	/// Setup FFTW Plans
	fftw_plan fftw_2d_dft_r2c = fftw_mpi_plan_dft_r2c_2d(Nx, Ny, u, u_h, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	fftw_plan fftw_2d_dft_c2r = fftw_mpi_plan_dft_c2r_2d(Nx, Ny, u_h, u, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	
	fftw_plan fftw_2d_dft_batch_r2c = fftw_mpi_plan_many_dft_r2c((int)2, N, (ptrdiff_t) 2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, u_batch, u_h_batch, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	fftw_plan fftw_2d_dft_batch_c2r = fftw_mpi_plan_many_dft_c2r((int)2, N, (ptrdiff_t) 2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, u_h_batch, u_batch, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	


	/////////////// 
	/// Fill Data
	double x, y;
	for (int i = 0; i < local_Nx; ++i)
	{
		tmp = i * (Ny + 2);
		x = ((double) i) * 2.0 * M_PI / Nx;
		for (int j = 0; j < Ny; ++j)
		{
			indx = tmp + j;
			y = ((double) j) * 2.0 * M_PI / Ny;
			
			u[indx] = - sin(M_PI * x) * cos(M_PI * y); 
			v[indx] = cos(M_PI * x) * sin(M_PI * y);

			u_batch[2 * indx + 0] = - sin(M_PI * x) * cos(M_PI * y);
			u_batch[2 * indx + 1] = cos(M_PI * x) * sin(M_PI * y);
			// printf("u[%d]: %1.16lf \tv[%d]: %1.16lf\n", i * Ny + j, u[indx], i * Ny + j, v[indx]);
		}
	}
	printf("\n\n");


	int j = 0;
	for (int i = 0; i < local_Nx; ++i) {
		if (local_Nx_start + i <= Nx / 2) {   // Set the first half of array to the positive k
			k[0][j] = local_Nx_start + i;
			j++;
		}
		else if (local_Nx_start + i > Nx / 2) { // Set the second half of array to the negative k
			k[0][j] = local_Nx_start + i - Nx;
			j++;
		}
	}
	for (int i = 0; i < Ny; ++i) {
		if (i < Nyf) {
			k[1][i] = i;
		}
	}


	///////////////
	/// Execute the transforms
	fftw_mpi_execute_dft_r2c(fftw_2d_dft_r2c, u, u_h);
	fftw_mpi_execute_dft_r2c(fftw_2d_dft_r2c, v, v_h);
	fftw_mpi_execute_dft_r2c(fftw_2d_dft_batch_r2c, u_batch, u_h_batch);

	///////////////
	/// Compare in Fourier Space
	fftw_complex u_h_err, v_h_err; 
	double err_u_h = 0.0; 
	double err_v_h = 0.0;
	for (int i = 0; i < local_Nx; ++i)
	{
		tmp = i * Nyf;
		for (int j = 0; j < Nyf; ++j)
		{
			indx = tmp + j;

			u_h_err = u_h[indx] - u_h_batch[2 * indx + 0];
			v_h_err = v_h[indx] - u_h_batch[2 * indx + 1];

			// printf("v_h[%d]: %1.16lf %1.16lf I \t\t\t v_h[%d]: %1.16lf %1.16lf I\n", i * Nyf + j, creal(v_h[indx]), cimag(v_h[indx]), indx, creal(u_h_batch[2 * indx + 1]), cimag(u_h_batch[2 * indx + 1]));
			// printf("uh[%d]: %1.16lf %1.16lf \t\t\t vh[%d]: %1.16lf %1.16lf\n", indx, creal(u_h[indx]), cimag(u_h[indx]), indx, creal(v_h[indx]), cimag(v_h[indx]));
			err_u_h += pow(cabs(u_h_err), 2.0);
			err_v_h += pow(cabs(v_h_err), 2.0);
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &err_u_h, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
	MPI_Allreduce(MPI_IN_PLACE, &err_v_h, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if (!rank) {
		printf("\n\nL2 Error for u_h in Fourier Space is: %1.16g \nL2 Error for v_h in Fourier Space is: %1.16g \n\n", sqrt(1.0 / (Nx * Ny) * err_u_h), sqrt(1.0 / (Nx * Ny) * err_v_h));
	}


	//////////////
	/// Compute the derivative
	for (int i = 0; i < local_Nx; ++i)
	{
		tmp = i * Nyf;
		for (int j = 0; j < Nyf; ++j)
		{
			indx = tmp + j;

			u_h[indx] = I * k[0][i] * u_h[indx]; // dudx
			v_h[indx] = I * k[1][j] * v_h[indx]; // dvdy

			u_h_batch[2 * indx + 0] = I * k[0][i] * u_h_batch[2 * indx + 0];
			u_h_batch[2 * indx + 1] = I * k[1][j] * u_h_batch[2 * indx + 1];
		}
	}


	///////////////
	/// Execute the transforms
	fftw_mpi_execute_dft_c2r(fftw_2d_dft_c2r, u_h, u);
	fftw_mpi_execute_dft_c2r(fftw_2d_dft_c2r, v_h, v);
	fftw_mpi_execute_dft_c2r(fftw_2d_dft_batch_c2r, u_h_batch, u_batch);


	////////////// 
	/// Compare results
	double du_err, dv_err;
	double err_du = 0.0;
	double err_dv = 0.0;
	for (int i = 0; i < local_Nx; ++i)
	{
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j)
		{
			indx = tmp + j;

			du_err = u[indx] - u_batch[2 * indx + 0];
			dv_err = v[indx] - u_batch[2 * indx + 1];

			printf("u[%d]: %1.16lf \t \t \t v[%d]: %1.16lf\n", i * Ny + j, u[indx]/ (Nx * Ny), i * Ny + j, v[indx]/ (Nx * Ny));
			err_du += pow(cabs(du_err), 2.0);
			err_dv += pow(cabs(dv_err), 2.0);
		}
	}
	printf("\n\n");
	MPI_Allreduce(MPI_IN_PLACE, &err_du, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
	MPI_Allreduce(MPI_IN_PLACE, &err_dv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if (!rank) {
		printf("\n\nL2 Error for dudx in Real Space is: %1.16g \nL2 Error for dvdy in Real Space is: %1.16g \n\n", sqrt(1.0 / (Nx * Ny) * err_du), sqrt(1.0 / (Nx * Ny) * err_dv));
	}


	/////////////////
	/// Compare to actual derivative
	double du_b_err, dv_b_err;
	double err_du_b = 0.0;
	double err_dv_b = 0.0;
	for (int i = 0; i < local_Nx; ++i)
	{
		tmp = i * (Ny + 2);
		x = ((double) i) * 2.0 * M_PI / Nx;
		for (int j = 0; j < Ny; ++j)
		{
			indx = tmp + j;
			y = ((double) j) * 2.0 * M_PI / Ny;

			du_err   = u[indx] / (Nx * Ny) - (-M_PI * cos(M_PI * x) * cos(M_PI * y));
			dv_err   = v[indx] / (Nx * Ny) - (M_PI * cos(M_PI * x) * cos(M_PI * y));
			du_b_err = u_batch[2 * indx + 0] / (Nx * Ny) - (-M_PI * cos(M_PI * x) * cos(M_PI * y));
			dv_b_err = u_batch[2 * indx + 1] / (Nx * Ny) - (M_PI * cos(M_PI * x) * cos(M_PI * y));	

			// printf("v[%d]: %1.16lf \t v_b[%d]: %1.16lf \t a[%d]: %1.16lf \n", indx, v[indx] / (Nx * Ny), indx, u_batch[2 * indx + 1]/ (Nx * Ny), indx, M_PI * cos(M_PI * x) * cos(M_PI * y));
			err_du += pow(cabs(du_err), 2.0);
			err_dv += pow(cabs(dv_err), 2.0);
			err_du_b += pow(cabs(du_err), 2.0);
			err_dv_b += pow(cabs(dv_err), 2.0);
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &err_du, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
	MPI_Allreduce(MPI_IN_PLACE, &err_dv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &err_du_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
	MPI_Allreduce(MPI_IN_PLACE, &err_dv_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if (!rank) {
		printf("\n\nL2 Error deriv err u: %1.16g \nL2 Error deriv err v: %1.16g \nL2 Error deriv err u_b: %1.16g \nL2 Error deriv err v_b: %1.16g \n\n", sqrt(1.0 / (Nx * Ny) * err_du), sqrt(1.0 / (Nx * Ny) * err_dv), sqrt(1.0 / (Nx * Ny) * err_du_b), sqrt(1.0 / (Nx * Ny) * err_dv_b));
	}


	fftw_destroy_plan(fftw_2d_dft_r2c);
	fftw_destroy_plan(fftw_2d_dft_c2r);
	fftw_destroy_plan(fftw_2d_dft_batch_r2c);
	fftw_destroy_plan(fftw_2d_dft_batch_c2r);

	fftw_free(u);
	fftw_free(v);
	fftw_free(u_batch);
	fftw_free(u_h);
	fftw_free(v_h);
	fftw_free(u_h_batch);
	fftw_free(k[0]);
	fftw_free(k[1]);

	// Cleanup FFTW MPI interface - Calls the serial fftw_cleanup function also
	fftw_mpi_cleanup();    

	//////////////////////////////////
	// Exit MPI scetion
	MPI_Finalize();



	return 0;
}