/**
* @file utils.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the field functions for the pseudospectral solver data
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
#include <time.h>
#include <sys/time.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"

// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function used to fill the full field arrays
 */
void FullFieldData(void) {

	// Initialize variables
	int tmp, tmp1, tmp2;
	int indx;
	double k_sqr, k_sqr_fac, phase, amp;
	const long int Ny 		  = sys_vars->N[0];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;


	// --------------------------------
	// Fill The Full Field Arrays
	// --------------------------------
	for (int i = 0; i < Ny; ++i) {
		if (abs(run_data->k[0][i]) <= sys_vars->kmax) {
			tmp  = i * Nx_Fourier;	
			tmp1 = (sys_vars->kmax - run_data->k[0][i]) * (2 * sys_vars->kmax + 1); // ky > 0 - are the first kmax rows hence the -kx
			tmp2 = (sys_vars->kmax + run_data->k[0][i]) * (2 * sys_vars->kmax + 1); // ky < 0 - are the next kmax rows hence the +kx
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;
				if (abs(run_data->k[1][j]) <= sys_vars->kmax) {

					// Compute |k|^2 and 1 / |k|^2
					k_sqr = (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]); 
					if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
						k_sqr_fac = 1.0 / k_sqr;
					}
					else {
						k_sqr_fac = 0.0;	
					}

					// Pre-compute data
					phase = fmod(carg(run_data->w_hat[indx]) + 2.0 * M_PI, 2.0 * M_PI);
					amp   = cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));

					// fill the full field phases and spectra
	 				if (k_sqr <= sys_vars->kmax_sqr) {
	 					// No conjugate for ky = 0
	 					if (run_data->k[1][j] == 0) {
	 						// proc_data->k_full[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 
	 						proc_data->phases[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])] = phase;
	 						proc_data->amps[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = cabs(run_data->w_hat[indx]);
	 						proc_data->enrg[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = amp * k_sqr_fac;
	 						proc_data->enst[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = amp;
	 					}
	 					else {
	 						// Fill data and its conjugate
							proc_data->phases[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])] = phase;
							proc_data->phases[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])] = fmod(-phase + 2.0 * M_PI, 2.0 * M_PI);
							proc_data->amps[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = cabs(run_data->w_hat[indx]);
							proc_data->amps[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])]   = cabs(run_data->w_hat[indx]);
							proc_data->enrg[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = amp * k_sqr_fac;
							proc_data->enrg[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])]   = amp * k_sqr_fac;
							proc_data->enst[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = amp;
							proc_data->enst[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])]   = amp;
	 					}
	 				}
	 				else {
	 					// All dealiased modes set to zero
						if (run_data->k[1][j] == 0) {
	 						proc_data->phases[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])] = -50.0;
	 						proc_data->amps[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = -50.0;
	 						proc_data->enrg[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = -50.0;
	 						proc_data->enst[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = -50.0;
	 					}
	 					else {	
	 						proc_data->phases[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])] = -50.0;
	 						proc_data->phases[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])] = -50.0;
	 						proc_data->amps[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = -50.0;
	 						proc_data->amps[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])]   = -50.0;
	 						proc_data->enrg[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = -50.0;
	 						proc_data->enrg[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])]   = -50.0;
	 						proc_data->enst[(int)(tmp1 + sys_vars->kmax + run_data->k[1][j])]   = -50.0;
	 						proc_data->enst[(int)(tmp2 + sys_vars->kmax - run_data->k[1][j])]   = -50.0;
	 					}
	 				}
				}	
			}						
		}
	}
}
/**
 * Function to apply the selected dealiasing filter to the input array. Can be Fourier vorticity or velocity
 * @param array    	The array containing the Fourier modes to dealiased
 * @param array_dim The extra array dimension -> will be 1 for scalar or 2 for vector
 * @param N        	Array containing the dimensions of the system
 */
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N) {

	// Initialize variables
	int tmp, indx;
	const long int Ny         = N[0];
	const long int Nx         = N[1];
	const long int Nx_Fourier = Nx / 2 + 1;
	double k_sqr;
	double kmax_sqr = pow((int) (Ny / 3.0), 2.0);
	#if defined(__DEALIAS_HOU_LI)
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter 
	// --------------------------------------------
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = array_dim * (tmp + j);

			// Get |k|^2
			k_sqr = (double) run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j];

			#if defined(__DEALIAS_23)
			if (k_sqr <= 0.0 || k_sqr > kmax_sqr) {
				for (int l = 0; l < array_dim; ++l) {
					// Set dealised modes to 0
					array[indx + l] = 0.0 + 0.0 * I;	
				}
			}
			else {
				for (int l = 0; l < array_dim; ++l) {
					// Apply DFT normaliztin to undealiased modes
					array[indx + l] = array[indx + l];	
				}				
			}
			#elif __DEALIAS_HOU_LI
			// Compute Hou-Li filter
			hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Ny / 2), 2.0) + pow(run_data->k[1][j] / (Nx / 2), 2.0))), 36.0));

			for (int l = 0; l < array_dim; ++l) {
				// Apply filter and DFT normaliztion
				array[indx + l] *= hou_li_filter;
			}
			#endif
		}
	}
}
/**
 * Function to force conjugacy of the initial condition
 * @param w_hat The Fourier space worticity field
 * @param N     The array containing the size of the system in each dimension
 * @param dim   Dimension of the array -> 1: for scalar, 2: for vector.
 */
void ForceConjugacy(fftw_complex* array, const long int* N, const int dim) {

	// Initialize variables
	int tmp;
	const long int Ny         = N[0];
	const long int Nx_Fourier = N[1] / 2 + 1;

	// Allocate tmp memory to hold the data to be conjugated
	fftw_complex* conj_data = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * dim);

	// Loop through local process and store data in appropriate location in conj_data
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int d = 0; d < dim; ++d) {
			conj_data[dim * i + d] = array[dim * tmp + d];
		}
	}

	// Now ensure the 
	for (int i = 0; i < Ny; ++i) {
		if (run_data->k[0][i] > 0) {
			tmp = i * Nx_Fourier;
			for (int d = 0; d < dim; ++d) {
				// Fill the conjugate modes with the conjugate of the postive k modes
				array[dim * tmp + d] = conj(conj_data[dim * (Ny - run_data->k[0][i]) + d]);
				// array[dim * tmp + d] = conj(array[dim * (Ny - run_data->k[0][i]) + d]);
			}
		}
	}

	// Free memory
	fftw_free(conj_data);
}
/**
 * Function to compute the nonlinear term of the equation motion for the Fourier space vorticity
 * @param w_hat      The Fourier space vorticity
 * @param dw_hat_dt  The result of the nonlinear term
 * @param nonlinterm The nonlinear term in real space
 * @param u          The real space velocity
 * @param nabla_w    The gradient of the real space velocity
 */
void NonlinearRHS(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u) {

	// Initialize variables
	int tmp, indx;
	const long int Ny         = sys_vars->N[0];
	const long int Nx         = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	double k_sqr;
	double norm_fac = 1.0 / (Ny * Nx);


	// ----------------------------------
	//  Compute Fourier Space Velocity
	// ----------------------------------
	for (int i = 0; i < Ny; ++i) {
		tmp = i * (Nx_Fourier);
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Write w_hat to temporary array for transform back to real space
			run_data->tmp_w_hat[indx] = w_hat[indx];

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				// Laplacian prefactor
				k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Fill fill fourier velocities array
				dw_hat_dt[SYS_DIM * indx + 0] = I * ((double) run_data->k[0][i]) * k_sqr * w_hat[indx];
				dw_hat_dt[SYS_DIM * indx + 1] = -1.0 * I * ((double) run_data->k[1][j]) * k_sqr * w_hat[indx];
			}
			else {
				dw_hat_dt[SYS_DIM * (indx) + 0] = 0.0 + 0.0 * I;
				dw_hat_dt[SYS_DIM * (indx) + 1] = 0.0 + 0.0 * I;
			}
			// printf("wh[%d,%d]: %5.16lf %5.16lf ", i, j, creal(run_data->tmp_w_hat[indx]), cimag(run_data->tmp_w_hat[indx]));
		}
		// printf("\n");
	}
	// printf("\n");

	// Ensure conjugacy in the kx = 0
    ForceConjugacy(dw_hat_dt, sys_vars->N, 2);
 //    for (int i = 0; i < Ny; ++i) {
	// 	for (int j = 0; j < Nx_Fourier; ++j) {
	// 		printf("%s[%d,%d]: %5.16lf %5.16lf ", "uh", run_data->k[0][i], run_data->k[1][j], creal(dw_hat_dt[SYS_DIM * (i * (Nx_Fourier) + j) + 0]), cimag(dw_hat_dt[SYS_DIM * (i * (Nx_Fourier) + j) + 0]));
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// for (int i = 0; i < Ny; ++i) {
	// 	for (int j = 0; j < Nx_Fourier; ++j) {
	// 		printf("%s[%d,%d]: %5.16lf %5.16lf ", "vh", run_data->k[0][i], run_data->k[1][j], creal(dw_hat_dt[SYS_DIM * (i * (Nx_Fourier) + j) + 1]), cimag(dw_hat_dt[SYS_DIM * (i * (Nx_Fourier) + j) + 1]));
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// ----------------------------------
	//  Transform to Real Space
	// ----------------------------------
	// Perform transformation on the Fourier velocities
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, dw_hat_dt, u);

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Transform Fourier vorticity to real space
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->tmp_w_hat, run_data->w);

	// -----------------------------------
	// Perform Convolution in Real Space
	// -----------------------------------
	// Perform the multiplication in real space
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx;
		for (int j = 0; j < Nx; ++j) {
			indx = tmp + j; 

			// Perform multiplication of the nonlinear term 
			u[SYS_DIM * indx + 0] *= run_data->w[indx];
			u[SYS_DIM * indx + 1] *= run_data->w[indx];
 		}
 	}

	// ----------------------------------
	//  Transform to Fourier Space
	// ----------------------------------
	// Perform Fourier transform
	fftw_execute_dft_r2c(sys_vars->fftw_2d_dft_batch_r2c, u, dw_hat_dt);

	// Ensure conjugacy in the kx = 0 modes
    ForceConjugacy(dw_hat_dt, sys_vars->N, 2);

 	for (int i = 0; i < Ny; ++i) {
 		tmp = i * (Nx_Fourier);
 		for (int j = 0; j < Nx_Fourier; ++j) {
 			indx = tmp + j;

 			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
 				dw_hat_dt[indx] = (-I * run_data->k[1][j] * dw_hat_dt[SYS_DIM * indx + 0] -I * run_data->k[0][i] * dw_hat_dt[SYS_DIM * indx + 1]) * pow(norm_fac, 1.0);
 			}
 			else {
 				dw_hat_dt[indx] = 0.0 + 0.0 * I;
 				dw_hat_dt[indx] = 0.0 + 0.0 * I;
 			}
		}
	}

	// ----------------------------------------
 	// Apply Dealiasing & Conjugacy
 	// ----------------------------------------
 	// Apply dealiasing 
 	ApplyDealiasing(dw_hat_dt, 1, sys_vars->N);

	// Ensure conjugacy in the ky = 0 modes of the intial condition
    ForceConjugacy(dw_hat_dt, sys_vars->N, 1);
}
/**
* Function to compute 1Denergy spectrum from the Fourier vorticity
*/
void EnergySpectrum(void) {
    
    // Initialize variables
    int tmp, indx;
	int spec_indx;
	const long int Ny 		  = sys_vars->N[0];
	const long int Nx 		  = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Ny * Nx, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0); 
    double k_sqr;
    
    // --------------------------------
	//  Initialize Spectrum
	// --------------------------------	
    for(int i = 0; i < sys_vars->n_spec; ++i) {
        proc_data->enrg_spec[i] = 0.0;
    }
    

    // --------------------------------
	//  Compute spectrum
	// --------------------------------	
    for(int i = 0; i < Ny; ++i) {
        tmp = i * Nx_Fourier;
        for(int j = 0; j < Nx_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int) round(sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j])));
            
            if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				proc_data->enrg_spec[spec_indx] += 0.0;
			}
			else {
                // Compute the normalization factor 1.0 / |k|^2
                k_sqr = 1.0 / ((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));
            
                if ((j == 0) || (j == Nx_Fourier - 1)) {
                    proc_data->enrg_spec[spec_indx] += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                    // proc_data->enrg_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                }
                else {
                    // proc_data->enrg_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                    proc_data->enrg_spec[spec_indx] += 2.0 *cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                }
            }
        }
    }
}
/**
 * Function to compute the 1D enstrophy spectrum from the Fourier vorticity
 */
void EnstrophySpectrum(void) {

	// Initialize variables
	int tmp, indx;
	int spec_indx;
	const long int Ny 		  = sys_vars->N[0];
	const long int Nx 		  = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Ny * Nx, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0); 

	// --------------------------------
	//  Initialize Spectrum
	// --------------------------------	
    for(int i = 0; i < sys_vars->n_spec; ++i) {
        proc_data->enst_spec[i] = 0.0;
    }


	// --------------------------------
	//  Compute spectrum
	// --------------------------------	
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Compute spectrum index/bin
			spec_indx = (int )round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				proc_data->enst_spec[spec_indx] += 0.0;
			}
			else {
				if ((j == 0) || (j == Nx_Fourier - 1)) {
					// proc_data->enst_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
					proc_data->enst_spec[spec_indx] += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					// proc_data->enst_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
					proc_data->enst_spec[spec_indx] += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
		}
	}
}
/**
 * Function to compute the energy & enstrophy flux spectra and energy and enstorphy flux and dissipation in C for the current snapshot of the simulation
 * The results are gathered on the master rank before being written to file
 * @param snap    The current snapshot of the simulation
 */
void FluxSpectra(int snap) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	const long int Ny         = sys_vars->N[0];
	const long int Nx         = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	double pre_fac   = 0.0;
	double norm_fac  = 0.5 / pow(Ny * Nx, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0);
	double tmp_deriv, tmp_diss;

	// ------------------------------------
	// Initialize Arrays
	// ------------------------------------
	// Initialize arrays
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		#if defined(__ENST_FLUX)
		proc_data->d_enst_dt_spec[i] = 0.0;
		proc_data->enst_diss_spec[i] = 0.0;
		proc_data->enst_flux_spec[i] = 0.0;
		#endif
		#if defined(__ENRG_FLUX)
		proc_data->d_enrg_dt_spec[i] = 0.0;
		proc_data->enrg_diss_spec[i] = 0.0;
		proc_data->enrg_flux_spec[i] = 0.0;
		#endif
	}

	// Initialize Collective phase order array
	#if defined(__SEC_PHASE_SYNC)
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
		proc_data->enst_flux_C_theta[i]   = 0.0;
		proc_data->enst_diss_C_theta[i]   = 0.0;
		proc_data->phase_order_C_theta[i] = 0.0 + 0.0 * I;
	}
	#endif

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Compute the nonlinear term
	NonlinearRHS(run_data->w_hat, proc_data->dw_hat_dt, proc_data->nabla_psi);


	// -------------------------------------
	// Compute the Energy Flux Spectrum
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// The |k|^2 prefactor
				k_sqr = (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {  
					// Both Hyperviscosity and Ekman drag
					if (round(sqrt(k_sqr)) <= EKMN_DRAG_K) {
						pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA_LOW_K * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
					}
					else {
						pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA_HIGH_K * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
					}
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// No hyperviscosity but we have Ekman drag
					if (round(sqrt(k_sqr)) <= EKMN_DRAG_K) {
						pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA_LOW_K * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
					}
					else {
						pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA_HIGH_K * pow(k_sqr, sys_vars->EKMN_DRAG_POW);						
					}
				}
				else if ((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
				 	// Hyperviscosity only
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW);
				}
				else { 
					// No hyper viscosity or no ekman drag -> just normal viscosity
					pre_fac = sys_vars->NU * k_sqr; 
				}

				// Get the spectrum index
				spec_indx = (int) round(sqrt(k_sqr));

				// Update spectrum bin
				if ((j == 0) || (j == Nx_Fourier - 1)) {
					#if defined(__ENRG_FLUX) || defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
					// Get temporary values
					tmp_deriv = creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * const_fac * norm_fac;
					tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * const_fac * norm_fac;
					#endif

					// Update the current bin sum 
					#if defined(__ENST_FLUX)
					proc_data->d_enst_dt_spec[spec_indx] += tmp_deriv + tmp_diss;
					proc_data->enst_diss_spec[spec_indx] += tmp_diss;
					proc_data->enst_flux_spec[spec_indx] += tmp_deriv;
					#endif
					#if defined(__ENRG_FLUX)
					proc_data->d_enrg_dt_spec[spec_indx] += (tmp_deriv + tmp_diss) / k_sqr;
					proc_data->enrg_diss_spec[spec_indx] += tmp_diss / k_sqr;
					proc_data->enrg_flux_spec[spec_indx] += tmp_deriv / k_sqr;
					#endif
					#if defined(__SEC_PHASE_SYNC)
					// Compute the enstrophy dissipation field
					proc_data->enst_diss_field[indx] = pre_fac * run_data->w_hat[indx];

					// Compute the enstrophy flux, dissipation and collective phase for C_theta
					if (k_sqr > sys_vars->kmax_C_sqr && k_sqr <= sys_vars->kmax_sqr) {
					// if (sqrt(k_sqr) >= sqrt(sys_vars->kmax_C_sqr) - 0.5 && sqrt(k_sqr) <= sqrt(sys_vars->kmax_sqr) + 0.5) {
						for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
							if (proc_data->phase_angle[indx] >= proc_data->theta_k3[a] - proc_data->dtheta_k3/2.0 && proc_data->phase_angle[indx] < proc_data->theta_k3[a] + proc_data->dtheta_k3/2.0) {
								
								// Record the flux and dissipation
								proc_data->enst_diss_C_theta[a] += tmp_diss;
								proc_data->enst_flux_C_theta[a] += tmp_deriv;

								// Record the phase sync
								if (run_data->k[0][i] > 0 && cabs(tmp_deriv) != 0.0) {
									proc_data->phase_order_C_theta[a] += (tmp_deriv + tmp_diss) / cabs(tmp_deriv);
								}
							}
						}
					}
					#endif
				}
				else {
					#if defined(__ENRG_FLUX) || defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
					// Get temporary values
					tmp_deriv = creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * const_fac * norm_fac;
					tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * const_fac * norm_fac;
					#endif

					// Update the running sum for the flux
					#if defined(__ENST_FLUX)
					proc_data->d_enst_dt_spec[spec_indx] += 2.0 * (tmp_deriv + tmp_diss);
					proc_data->enst_diss_spec[spec_indx] += 2.0 * tmp_diss;
					proc_data->enst_flux_spec[spec_indx] += 2.0 * tmp_deriv;
					#endif
					#if defined(__ENRG_FLUX)
					proc_data->d_enrg_dt_spec[spec_indx] += 2.0 * (tmp_deriv + tmp_diss) / k_sqr;
					proc_data->enrg_diss_spec[spec_indx] += 2.0 * tmp_diss / k_sqr;
					proc_data->enrg_flux_spec[spec_indx] += 2.0 * tmp_deriv / k_sqr;
					#endif
					#if defined(__SEC_PHASE_SYNC)
					// Compute the enstrophy dissipation field
					proc_data->enst_diss_field[indx] = pre_fac * run_data->w_hat[indx];

					// Compute the enstrophy flux, dissipation and collective phase for C_theta
					if (k_sqr > sys_vars->kmax_C_sqr && k_sqr <= sys_vars->kmax_sqr) {
					// if (sqrt(k_sqr) >= sqrt(sys_vars->kmax_C_sqr) - 0.5 && sqrt(k_sqr) <= sqrt(sys_vars->kmax_sqr) + 0.5) {
						for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
							if (proc_data->phase_angle[indx] >= proc_data->theta_k3[a] - proc_data->dtheta_k3/2.0 && proc_data->phase_angle[indx] < proc_data->theta_k3[a] + proc_data->dtheta_k3/2.0) {

								// Record the flux and dissipation
								proc_data->enst_diss_C_theta[a] += tmp_diss;
								proc_data->enst_flux_C_theta[a] += tmp_deriv;

								// Record the phase sync
								if (cabs(tmp_deriv) != 0.0) {
									proc_data->phase_order_C_theta[a] += (tmp_deriv + tmp_diss) / cabs(tmp_deriv);
								}
							}
						}
					}
					#endif
				}
			}
		}
	}

	// -------------------------------------
	// Accumulate The Flux
	// -------------------------------------
	// Accumulate the flux for each k
	for (int i = 1; i < sys_vars->n_spec; ++i) {
		if (i >= (int)round(sqrt(sys_vars->kmax_C_sqr))) {
			// Record the enstrophy flux out of the set C
			#if defined(__ENST_FLUX)
			proc_data->enst_flux_C[snap] += proc_data->enst_flux_spec[i];
			proc_data->enst_diss_C[snap] += proc_data->enst_diss_spec[i];
			#endif
			#if defined(__ENRG_FLUX)
			proc_data->enrg_flux_C[snap] += proc_data->enrg_flux_spec[i];
			proc_data->enrg_diss_C[snap] += proc_data->enrg_diss_spec[i];
			#endif
		}
	}
}
/**
 * Allocates memory for the spectra, flux and full field computations
 * @param N Array containing the size of each dimension
 */
void AllocateFullFieldMemory(const long int* N) {

	// Iniitialize variables
	int tmp3;
	const long int Ny = N[0];
	const long int Nx = N[1];
	const long int Nx_Fourier = Nx / 2 + 1;

	// Compute maximum wavenumber
	sys_vars->kmax = (int) (Ny / 3.0);

	// Get the various kmax variables
	sys_vars->kmax_sqr   = pow(sys_vars->kmax, 2.0);
	sys_vars->kmax_C   	 = (int) ceil(sys_vars->kmax_frac * sys_vars->kmax);
	sys_vars->kmax_C_sqr = pow(sys_vars->kmax_C, 2.0);

	// --------------------------------	
	//  Allocate Full Field Arrays
	// --------------------------------
	#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	// Allocate memory for the full field phases
	proc_data->phases = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax + 1) * (2 * sys_vars->kmax + 1));
	if (proc_data->phases == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field amplitudes
	proc_data->amps = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax + 1) * (2 * sys_vars->kmax + 1));
	if (proc_data->amps == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field enstrophy spectrum
	proc_data->enst = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax + 1) * (2 * sys_vars->kmax + 1));
	if (proc_data->enst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Enstrophy");
		exit(1);
	}	

	// Allocate memory for the full field enrgy spectrum
	proc_data->enrg = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax + 1) * (2 * sys_vars->kmax + 1));
	if (proc_data->enrg == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Energy");
		exit(1);
	}	

	// Initialize arrays
	for (int i = 0; i < (2 * sys_vars->kmax + 1); ++i) {
		tmp3 = i * (2 * sys_vars->kmax + 1);
		for (int j = 0; j < (2 * sys_vars->kmax + 1); ++j) {
			proc_data->phases[tmp3 + j] = 0.0;
			proc_data->enst[tmp3 + j]   = 0.0;
			proc_data->enrg[tmp3 + j]   = 0.0;
			proc_data->amps[tmp3 + j]   = 0.0;
		}
	}	
	#endif


	// --------------------------------	
	//  Allocate Spectra Arrays
	// --------------------------------	
	// Get the size of the spectra
	sys_vars->n_spec = (int) round(sqrt((double)(Ny / 2.0) * (Ny / 2.0) + (double)(Nx / 2.0) * (Nx / 2.0)) + 1);


	#if defined(__SPECTRA)
	// Allocate memory for the enstrophy spectrum
	proc_data->enst_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Enstrophy Spectrum");
		exit(1);
	}
    
    // Allocate memory for the enstrophy spectrum
	proc_data->enrg_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enrg_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Energy Spectrum");
		exit(1);
	}

	// Initialize arrays
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		proc_data->enst_spec[i] = 0.0;
        proc_data->enrg_spec[i] = 0.0;
	}
	#endif

	// --------------------------------	
	//  Allocate Flux Data
	// --------------------------------
 	///-------------- Nonlinear RHS function arrays
	/// RHS
	proc_data->dw_hat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier * SYS_DIM);
	if (proc_data->dw_hat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "RHS");
		exit(1);
	}
	// The gradient of the real stream function
	proc_data->nabla_psi = (double* )fftw_malloc(sizeof(double) * Ny * Nx * SYS_DIM);
	if (proc_data->nabla_psi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Gradient of Real Space Vorticity");
		exit(1);
	}
	

	#if defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	// --------------------------------	
	//  Allocate C_\theta_k3 Enstrophy Flux
	// --------------------------------
	#if defined(__SEC_PHASE_SYNC)
	// The enstrophy flux out of the set C
	proc_data->enst_flux_C_theta = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enst_flux_C_theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux In/Out of C_theta");
		exit(1);
	}
	// The enstrophy flux out of the set C
	proc_data->enst_diss_C_theta = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enst_diss_C_theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation In/Out C_theta");
		exit(1);
	}
	// Allocate phase order for C_theta
	proc_data->phase_order_C_theta = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_snaps);
	if (proc_data->phase_order_C_theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Collective Phase Order for C_theta");
		exit(1);
	}
	// Allocate enstorphy dissipation field
	proc_data->enst_diss_field = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (proc_data->enst_diss_field == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Field Fourier Space");
		exit(1);
	}	
	#endif

	// --------------------------------	
	//  Allocate Enstrophy Flux
	// --------------------------------
	#if defined(__ENST_FLUX)
	// The time derivative of enstrophy spectrum
	proc_data->d_enst_dt_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->d_enst_dt_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time Derivative of Enstrophy Spectrum");
		exit(1);
	}
	// The enstrophy flux spectrum
	proc_data->enst_flux_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_flux_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}
	// The enstrophy diss spectrum
	proc_data->enst_diss_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_diss_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Spectrum");
		exit(1);
	}
	// The enstrophy flux out of the set C
	proc_data->enst_flux_C = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enst_flux_C == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Out of C");
		exit(1);
	}
	// The enstrophy flux out of the set C
	proc_data->enst_diss_C = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enst_diss_C == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation in C");
		exit(1);
	}
	#endif

	// --------------------------------	
	//  Allocate Energy Flux
	// --------------------------------
	#if defined(__ENRG_FLUX)
	// The time derivative of energy spectrum
	proc_data->d_enrg_dt_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->d_enrg_dt_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time Derivative of Enstrophy Spectrum");
		exit(1);
	}
	// The energy flux spectrum
	proc_data->enrg_flux_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enrg_flux_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}
	// The energy flux spectrum
	proc_data->enrg_diss_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enrg_diss_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Spectrum");
		exit(1);
	}
	// The energy flux out of the set C
	proc_data->enrg_flux_C = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enrg_flux_C == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Out of C");
		exit(1);
	}
	// The energy flux out of the set C
	proc_data->enrg_diss_C = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enrg_diss_C == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation in C");
		exit(1);
	}
	#endif

	// Initialize arrays
	for (int i = 0; i < Ny; ++i) {
		for (int j = 0; j < Nx; ++j) {
			if(j < Nx_Fourier) {
				// proc_data->dw_hat_dt[SYS_DIM * (i * Nx_Fourier + j) + 0] = 0.0 + 0.0 * I;
				// proc_data->dw_hat_dt[SYS_DIM * (i * Nx_Fourier + j) + 1] = 0.0 + 0.0 * I;
			}
		}
	}
	for (int i = 0; i < sys_vars->num_snaps; ++i) {
		#if defined(__SEC_PHASE_SYNC)
		proc_data->enst_flux_C_theta[i] = 0.0;
		proc_data->enst_diss_C_theta[i] = 0.0;
		#endif
		#if defined(__ENST_FLUX)
		proc_data->enst_flux_C[i] = 0.0;
		proc_data->enst_diss_C[i] = 0.0;
		#endif
		#if defined(__ENRG_FLUX)
		proc_data->enrg_flux_C[i] = 0.0;
		proc_data->enrg_diss_C[i] = 0.0;
		#endif
	}
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		#if defined(__ENST_FLUX)
		proc_data->d_enst_dt_spec[i] = 0.0;
		proc_data->enst_flux_spec[i] = 0.0;
		proc_data->enst_diss_spec[i] = 0.0;
		#endif
		#if defined(__ENRG_FLUX)
		proc_data->d_enrg_dt_spec[i] = 0.0;
		proc_data->enrg_flux_spec[i] = 0.0;
		proc_data->enrg_diss_spec[i] = 0.0;
		#endif
	}
	#endif
}
/**
 * Frees memory allocated for the spectra, flux and full field computations
 */
void FreeFullFieldObjects(void) {

	// -------------------------------------
	// Free Memory
	// -------------------------------------
	#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	fftw_free(proc_data->phases);
	fftw_free(proc_data->amps);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	#endif
	#if defined(__SPECTRA)
	fftw_free(proc_data->enst_spec);
    fftw_free(proc_data->enrg_spec);
	#endif
	#if defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	
	#if defined(__SEC_PHASE_SYNC)
	fftw_free(proc_data->enst_flux_C_theta);
	fftw_free(proc_data->enst_diss_C_theta);
	fftw_free(proc_data->phase_order_C_theta);
	fftw_free(proc_data->enst_diss_field);
	#endif
	#if defined(__ENST_FLUX)
	fftw_free(proc_data->enst_flux_C);
	fftw_free(proc_data->enst_diss_C);
	fftw_free(proc_data->d_enst_dt_spec);
	fftw_free(proc_data->enst_flux_spec);
	fftw_free(proc_data->enst_diss_spec);
	#endif
	#if defined(__ENRG_FLUX)
	fftw_free(proc_data->enrg_flux_C);
	fftw_free(proc_data->enrg_diss_C);
	fftw_free(proc_data->d_enrg_dt_spec);
	fftw_free(proc_data->enrg_flux_spec);
	fftw_free(proc_data->enrg_diss_spec);
	#endif
	#endif
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
