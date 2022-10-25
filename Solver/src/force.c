/**
* @file utils.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the forcing functions for the pseudospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h> 
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
 * Function to initialize the forcing variables and arrays, indentify the local forcing processes and the number of forced modes
 */
void InitializeForcing(void) {

	// Initialize variables
	int tmp, indx;
	double k_abs;
	double scale_fac_f0;
	int num_forced_modes         = 0;
	int force_mode_counter       = 0;
	double sum_k_pow             = 0.0;
	double scaling_exp           = 0.0;
	sys_vars->local_forcing_proc = 0;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -----------------------------------
	// Initialize Forcing Objects 
	// -----------------------------------
	//--------------------------------- Apply Kolmogorov forcing
	if(!(strcmp(sys_vars->forcing, "KOLM"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			if (run_data->k[0][i] == 0) {
				sys_vars->local_forcing_proc = 1;
				num_forced_modes++;
			}
		}

		// Get the number of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}

			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			// Get the forcing index
			run_data->forcing_indx[0] = sys_vars->force_k;

			// Get the forcing wavenumbers
			run_data->forcing_k[0][0] = 0;
			run_data->forcing_k[1][0] = run_data->k[1][sys_vars->force_k];

			// Get the forcing scaling 
			run_data->forcing_scaling[0] = sys_vars->force_scale_var;
		}
	}
	//--------------------------------- Apply Stochastic forcing
	else if(!(strcmp(sys_vars->forcing, "STOC"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced		
		double forc_spect = 0.0;
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				// Compute |k|
				k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				if (run_data->k[0][i] != 0 && run_data->k[1][j] != 0) {
					if (j == 0 || j == Ny_Fourier - 1) {
						forc_spect += exp(- pow(k_abs - STOC_FORC_K, 2.0) / (2.0 * STOC_FORC_DELTA_K * STOC_FORC_DELTA_K)) / (2.0 * pow(k_abs, 2.0));
					}
					else {
						forc_spect += 2.0 * exp(- pow(k_abs - STOC_FORC_K, 2.0) / (2.0 * STOC_FORC_DELTA_K * STOC_FORC_DELTA_K)) / (2.0 * pow(k_abs, 2.0));						
					}
					
					// Count the forced modes
					sys_vars->local_forcing_proc = 1;
					num_forced_modes++;
				}
			}
		}

		// Get count of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// Sync sum of forced wavenumbers
		MPI_Allreduce(MPI_IN_PLACE, &forc_spect, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		forc_spect *= 0.5 / pow(sys_vars->N[0] * sys_vars->N[1], 2.0);

		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}

			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			// Initialize variables
			scale_fac_f0       = sys_vars->force_scale_var / forc_spect;
			force_mode_counter = 0;
			for (int i = 0; i < sys_vars->local_Nx; ++i) {
				tmp = i * Ny_Fourier;
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;

					// Compute |k|
					k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

					// Record the data for the forced modes
					if (run_data->k[0][i] != 0 && run_data->k[1][j] != 0) {
						run_data->forcing_scaling[force_mode_counter] = scale_fac_f0 * exp(- pow(k_abs - STOC_FORC_K, 2.0) / (2.0 * STOC_FORC_DELTA_K * STOC_FORC_DELTA_K));
						run_data->forcing_indx[force_mode_counter]    = indx;
						run_data->forcing_k[0][force_mode_counter]    = run_data->k[0][i];
						run_data->forcing_k[1][force_mode_counter]    = run_data->k[1][j];
						force_mode_counter++;
					}
				}
			}
		}
	}
	//--------------------------------- If CONST_GAUSS modes forcing selected
	else if(!(strcmp(sys_vars->forcing, "CONST_GAUSS"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {

				// Compute |k|
				k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				// Count the forced modes
				if (k_abs >= CONST_GAUSS_K_MIN && k_abs < CONST_GAUSS_K_MAX) {
					// Count forced modes
					sys_vars->local_forcing_proc = 1;
					num_forced_modes++;
				}
			}
		}

		// Get the number of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// -----------------------------------
		// Generate the Gaussian Data
		// -----------------------------------
		// Get the max size of the number of forced modes
		MPI_Allreduce(MPI_IN_PLACE, &num_forced_modes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

		// Allocate memory for the guassian data and conjugacy arrays
		fftw_complex* gauss_data = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_forced_modes);
		fftw_complex* conj_data  = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->N[0]);
		double r1, r2;
		double re_f, im_f;

		// Generate the gaussian data and record the modes to conjugate later
		force_mode_counter = 0;
		scale_fac_f0 = 0.0;
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {

				// Compute |k|
				k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				// Count the forced modes
				if (k_abs >= CONST_GAUSS_K_MIN && k_abs < CONST_GAUSS_K_MAX) {
					// Generate two uniform random numbers
					r1 = (double) rand() / (double) RAND_MAX;
					r2 = (double) rand() / (double) RAND_MAX;

					// Convert to Gaussian using Box-Muller transform
					re_f = sqrt(-2.0 * log(r1)) * cos(r2 * 2.0 * M_PI);
					im_f = sqrt(-2.0 * log(r1)) * sin(r2 * 2.0 * M_PI);

					// Record the gaussian data
					gauss_data[force_mode_counter] = re_f + im_f * I;
					if(run_data->k[1][i] == 0 && (run_data->k[0][i] > 0)) {
						// Record the forcing for modes to conjugate
						conj_data[run_data->k[0][i] * Ny_Fourier + 0] = re_f + im_f * I;
					}
					force_mode_counter++;

					// Update the sum of the l2 norm of the forcing
					scale_fac_f0 += cabs(re_f + im_f * I) * cabs(re_f + im_f * I);
				}
			}
		}
		// Add up the l2 norm sum of the forcing and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &num_forced_modes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		
		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * sys_vars->num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * sys_vars->num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}

			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			force_mode_counter = 0;
			for (int i = 0; i < sys_vars->local_Nx; ++i) {
				tmp = i * Ny_Fourier;
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;

					// Compute |k|
					k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

					// Record the data for the forced modes
					if (k_abs >= CONST_GAUSS_K_MIN && k_abs < CONST_GAUSS_K_MAX) {
						// Fill the forcing data arrays with the normalized forcing
						run_data->forcing[force_mode_counter]         = (gauss_data[force_mode_counter] / sqrt(scale_fac_f0) ) * sqrt(0.0025);
						run_data->forcing_indx[force_mode_counter]    = indx;
						run_data->forcing_scaling[force_mode_counter] = sqrt(0.0025); /// run_data->nu;
						run_data->forcing_k[0][force_mode_counter]    = run_data->k[0][i];
						run_data->forcing_k[1][force_mode_counter]    = run_data->k[1][j];
						force_mode_counter++;						
					}
				}
			}
		}

		// Free temporary memory
		fftw_free(gauss_data);
		fftw_free(conj_data);
	} 
	//--------------------------------- If ZERO modes forcing selected
	else if(!(strcmp(sys_vars->forcing, "ZERO"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {

				// Count the forced modes
				if ((abs(run_data->k[0][i]) <= sys_vars->force_k) || (abs(run_data->k[1][j]) <= sys_vars->force_k)) {
					sys_vars->local_forcing_proc = 1;
					num_forced_modes++;		
				}
			}
		}

		// Get the number of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * sys_vars->num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * sys_vars->num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}

			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			force_mode_counter = 0;
			for (int i = 0; i < sys_vars->local_Nx; ++i) {
				tmp = i * Ny_Fourier;
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;

					// Record the data for the forced modes
					if ((abs(run_data->k[0][i]) <= sys_vars->force_k) || (abs(run_data->k[1][j]) <= sys_vars->force_k)) {
						run_data->forcing[force_mode_counter]         = 0.0 + 0.0 * I;
						run_data->forcing_scaling[force_mode_counter] = 0.0;
						run_data->forcing_indx[force_mode_counter]    = indx;
						run_data->forcing_k[0][force_mode_counter]    = run_data->k[0][i];
						run_data->forcing_k[1][force_mode_counter]    = run_data->k[1][j];
						force_mode_counter++;
					}
				}
			}
		}
	}
	//--------------------------------- No forcing selected
	else {
		// Set number of forced modes and local forcing processes
		sys_vars->num_forced_modes   = 0;
		sys_vars->local_forcing_proc = 0;
	}
}
/**
 * Function that comutes the forcing for the current timestep
 */
void ComputeForcing(double dt) {

	// Initialize variables
	double r1, r2;
	double re_f, im_f;
	
	// --------------------------------------------
	// Compute Forcing
	// --------------------------------------------
	// Compute the forcing on the local process containing the forced modes
	if (sys_vars->local_forcing_proc) {
		//---------------------------- Compute Zero forcing -> specified modes are killed/set to 0
		if(!(strcmp(sys_vars->forcing, "ZERO"))) {
			// Loop over the forced modes
			for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
				run_data->w_hat[run_data->forcing_indx[i]] = 0.0 + 0.0 * I;
			}
		}
		//---------------------------- Compute Kolmogorov forcing -> f(u) = (sin(n y), 0); f(w) = -n cos(n y) -> f_k = -1/2 * n \delta(n)
		else if(!(strcmp(sys_vars->forcing, "KOLM"))) {
			// Compute the Kolmogorov forcing
			run_data->forcing[0] = -0.5 * sys_vars->force_scale_var * (sys_vars->force_k + 0.0 * I);
		}
		//---------------------------- Compute Stochastic forcing
		else if(!(strcmp(sys_vars->forcing, "STOC"))) {
			// Loop over the forced modes
			for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
				// // Generate two uniform random numbers
				// r1 = (double) rand() / (double) RAND_MAX;
				// r2 = (double) rand() / (double) RAND_MAX;

				// // Convert to Gaussian using Box-Muller transform
				// re_f = sqrt(-2.0 * log(r1)) * cos(r2 * 2.0 * M_PI);
				// im_f = sqrt(-2.0 * log(r1)) * sin(r2 * 2.0 * M_PI);

				// // Now compute the forcing 
				// run_data->forcing[i] = run_data->forcing_scaling[i] * (re_f + im_f * I);
				
				r1 = (double) rand() / (double) RAND_MAX;

				// // Now compute the forcing 
				run_data->forcing[i] = sqrt(run_data->forcing_scaling[i]) * cexp(2.0 * M_PI * I * r1) / sqrt(dt);				
			}		
		}
		//---------------------------- Compute Constant (in time) Gaussian ring forcing -> f_k = n1 + n2 * I, n1, n2 ~ N(0, 1)
		else if(!(strcmp(sys_vars->forcing, "CONST_GAUSS"))) {
			// Compute the forcing -> scale by the Grashof number = |f|/nu^2
			for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
				run_data->forcing[i] *= run_data->forcing_scaling[i];
			}
		}
		//---------------------------- No forcing
		else {

		}
	}
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------