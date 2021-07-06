/**
* @file utils.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the utilities functions for the pseudospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Utility function that gathers all the local real space vorticity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVorticityReal(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* w0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2));

	// Gather all the local arrays into w0
	MPI_Gather(run_data->w, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, w0, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("w[%ld]: %+5.16lf\t", i * (Ny) + j, w0[i * (Ny + 2) + j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w0);
}
/**
 * Utility function that gathers all the local Fourier space vorticity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVorticityFourier(const long int* N) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* w_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * (Ny_Fourier));

	// Gather all the local arrays into w0
	MPI_Gather(run_data->w_hat, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, w_hat0, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("wh[%ld]: %+5.16lf %+5.16lfI\t", i * (Ny_Fourier) + j, creal(w_hat0[i * Ny_Fourier + j]), cimag(w_hat0[i * Ny_Fourier + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w_hat0);
}
/**
 * Utility function that gathers all the local real space velocity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVelocityReal(const long int* N) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* u0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2) * SYS_DIM);

	// Gather all the local arrays into u0 and v0
	MPI_Gather(run_data->u, sys_vars->local_Nx * (Ny + 2) * SYS_DIM, MPI_DOUBLE, u0, sys_vars->local_Nx * (Ny + 2)* SYS_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("u[%ld]: %+5.16lf\t", i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 0]);
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("v[%ld]: %+5.16lf\t", i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 1]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u0);
}
/**
 * Utility function that gathers all the local Fourier space velocity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVelocityFourier(const long int* N) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);

	// Gather all the local arrays into u_hat0 and v_hat0
	MPI_Gather(run_data->u, sys_vars->local_Nx * Ny_Fourier * SYS_DIM, MPI_DOUBLE, u_hat, sys_vars->local_Nx * Ny_Fourier* SYS_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("uh[%ld]: %+5.10lf %+5.10lfI\t", i * (Ny_Fourier) + j, creal(u_hat[SYS_DIM * (i * Ny_Fourier + j) + 0]), cimag(u_hat[SYS_DIM * (i * Ny_Fourier + j) + 0]));
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("vh[%ld]: %+5.10lf %+5.10lfI\t", i * (Ny_Fourier) + j, creal(u_hat[SYS_DIM * (i * Ny_Fourier + j) + 1]), cimag(u_hat[SYS_DIM * (i * Ny_Fourier + j) + 1]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u_hat);
}
/**
 * Function to print scalar array in Fourier space
 * @param data The data to be printed
 * @param N    The size of the dimensions
 */
void PrintScalarFourier(fftw_complex* data, const long int* N, char* arr_name) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* w_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * (Ny_Fourier));

	// Gather all the local arrays into w0
	MPI_Gather(data, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, w_hat0, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("%s[%ld]: %+5.16lf %+5.16lfI\t", arr_name, i * (Ny_Fourier) + j, creal(w_hat0[i * Ny_Fourier + j]), cimag(w_hat0[i * Ny_Fourier + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w_hat0);
}
/**
 * Function to print scalar array in Real space
 * @param data The data to be printed
 * @param N    The size of the dimensions
 */
void PrintScalarReal(double* data, const long int* N, char* arr_name) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* w_hat0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2));

	// Gather all the local arrays into w0
	MPI_Gather(data, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, w_hat0, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("%s[%ld]: %+5.16lf\t", arr_name, i * (Ny) + j, w_hat0[i * (Ny + 2) + j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w_hat0);
}
/**
 * Function for printing a vector array in Real space
 * @param data data to be printed
 * @param N    The size of the dimensions
 */
void PrintVectorReal(double* data, const long int* N, char* arr_name1, char* arr_name2) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* u0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2) * SYS_DIM);

	// Gather all the local arrays into u0 and v0
	MPI_Gather(data, sys_vars->local_Nx * (Ny + 2) * SYS_DIM, MPI_DOUBLE, u0, sys_vars->local_Nx * (Ny + 2)* SYS_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("%s[%ld]: %+5.16lf\t", arr_name1, i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 0]);
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("%s[%ld]: %+5.16lf\t", arr_name2, i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 1]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u0);
}
/**
 * Function to print the Real and Fourier space variables
 */
void PrintSpaceVariables(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1; 

	// Allocate global array memory
	double* x0 = (double* )fftw_malloc(sizeof(double) * Nx);
	int* k0    = (int* )fftw_malloc(sizeof(int) * Nx);

	// Gather the data from each process onto master rank for printing
	MPI_Gather(run_data->x[0], sys_vars->local_Nx, MPI_DOUBLE, x0, sys_vars->local_Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(run_data->k[0], sys_vars->local_Nx, MPI_INT, k0, sys_vars->local_Nx, MPI_INT, 0, MPI_COMM_WORLD);

	// Print results on the master process
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			if (i < Ny_Fourier) {
				printf("x[%d]: %5.16lf\ty[%d]: %5.16lf\tkx[%d]: %d\tky[%d]: %d\n", i, x0[i], i, run_data->x[1][i], i, k0[i], i, run_data->k[1][i]);
			}
			else {
				printf("x[%d]: %5.16lf\ty[%d]: %5.16lf\tkx[%d]: %d\n", i, x0[i], i, run_data->x[1][i], i, k0[i]);
			}
		}
		printf("\n\n");
	}

	// Free up memory
	fftw_free(x0);
	fftw_free(k0);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------