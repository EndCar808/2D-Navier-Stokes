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
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// --------------------------------
	// Fill The Full Field Arrays
	// --------------------------------
	for (int i = 0; i < Nx; ++i) {
		if (abs(run_data->k[0][i]) < sys_vars->kmax) {
			tmp  = i * Ny_Fourier;	
			tmp1 = (sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1); // kx > 0 - are the first kmax rows hence the -kx
			tmp2 = (sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1); // kx < 0 - are the next kmax rows hence the +kx
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;
				if (abs(run_data->k[1][j]) < sys_vars->kmax) {

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
	 				if (k_sqr < sys_vars->kmax_sqr) {
	 					// No conjugate for ky = 0
	 					if (run_data->k[1][j] == 0) {
	 						// proc_data->k_full[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 
	 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = phase;
	 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = cabs(run_data->w_hat[indx]);
	 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp * k_sqr_fac;
	 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
	 					}
	 					else {
	 						// Fill data and its conjugate
	 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = phase;
	 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = fmod(-phase + 2.0 * M_PI, 2.0 * M_PI);
	 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] 	 = cabs(run_data->w_hat[indx]);
	 						proc_data->amps[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = cabs(run_data->w_hat[indx]);
	 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp * k_sqr_fac;
	 						proc_data->enrg[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = amp * k_sqr_fac;
	 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
	 						proc_data->enst[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = amp;
	 					}
	 				}
	 				else {
	 					// All dealiased modes set to zero
						if (run_data->k[1][j] == 0) {
	 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = -50.0;
	 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 					}
	 					else {	
	 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = -50.0;
	 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = -50.0;
	 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 						proc_data->amps[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = -50.0;
	 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 						proc_data->enrg[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = -50.0;
	 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 						proc_data->enst[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = -50.0;
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
	const long int Nx         = N[0];
	const long int Ny         = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	#if defined(__DEALIAS_HOU_LI)
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter 
	// --------------------------------------------
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = array_dim * (tmp + j);

			#if defined(__DEALIAS_23)
			if (sqrt((double) run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) > Nx / 3) {
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
			hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Nx / 2), 2.0) + pow(run_data->k[1][j] / (Ny / 2), 2.0))), 36.0));

			for (int l = 0; l < array_dim; ++l) {
				// Apply filter and DFT normaliztion
				array[indx + l] *= hou_li_filter;
			}
			#endif
		}
	}
}
/**
 * Function to compute the nonlinear term of the equation motion for the Fourier space vorticity
 * @param w_hat      The Fourier space vorticity
 * @param dw_hat_dt  The result of the nonlinear term
 * @param nonlinterm The nonlinear term in real space
 * @param u          The real space velocity
 * @param nabla_w    The gradient of the real space velocity
 */
void NonlinearRHS(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* nonlinterm, double* u, double* nabla_w) {

	// Initialize variables
	int tmp, indx;
	double vel1;
	double vel2;
	fftw_complex k_sqr;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// ----------------------------------
	//  Compute Fourier Space Velocity
	// ----------------------------------
	for (int i = 0; i < Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				// Laplacian prefactor
				k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Fill fill fourier velocities array
				dw_hat_dt[SYS_DIM * (indx) + 0] = k_sqr * ((double) run_data->k[1][j]) * w_hat[indx];
				dw_hat_dt[SYS_DIM * (indx) + 1] = -1.0 * k_sqr * ((double) run_data->k[0][i]) * w_hat[indx];
			}
			else {
				dw_hat_dt[SYS_DIM * (indx) + 0] = 0.0 + 0.0 * I;
				dw_hat_dt[SYS_DIM * (indx) + 1] = 0.0 + 0.0 * I;
			}
		}
	}

	// ----------------------------------
	//  Transform to Real Space
	// ----------------------------------
	// Perform transformation on the Fourier velocities
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, dw_hat_dt, u);

	// ----------------------------------
	//  Compute Gradient of Voriticty
	// ----------------------------------
	for (int i = 0; i < Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Fill vorticity derivatives array
			dw_hat_dt[SYS_DIM * indx + 0] = I * ((double) run_data->k[0][i]) * w_hat[indx];
			dw_hat_dt[SYS_DIM * indx + 1] = I * ((double) run_data->k[1][j]) * w_hat[indx]; 
		}
	}

	// ----------------------------------
	//  Transform to Real Space
	// ----------------------------------
	// Perform transformation of the gradient of Fourier space vorticity
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, dw_hat_dt, nabla_w);

	// ----------------------------------
	//  Multiply in Real Space
	// ----------------------------------
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j; 
 			
 			// Perform multiplication of the nonlinear term 
 			vel1 = u[SYS_DIM * indx + 0];
 			vel2 = u[SYS_DIM * indx + 1];
 			nonlinterm[indx] = 1.0 * (vel1 * nabla_w[SYS_DIM * indx + 0] + vel2 * nabla_w[SYS_DIM * indx + 1]);
 		}
 	}

	// ----------------------------------
	//  Transform to Fourier Space
	// ----------------------------------
	// Perform Fourier transform
	fftw_execute_dft_r2c(sys_vars->fftw_2d_dft_r2c, nonlinterm, dw_hat_dt);

	// Normalize result
	for (int i = 0; i < Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			dw_hat_dt[indx] *= 1.0 / pow((Nx * Ny), 2.0);
		}
	}

	// ----------------------------------
	//  Perform Dealiasing
	// ----------------------------------
	ApplyDealiasing(dw_hat_dt, 1, sys_vars->N);
}
/**
* Function to compute 1Denergy spectrum from the Fourier vorticity
*/
void EnergySpectrum(void) {
    
    // Initialize variables
    int tmp, indx;
	int spec_indx;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
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
    for(int i = 0; i < Nx; ++i) {
        tmp = i * Ny_Fourier;
        for(int j = 0; j < Ny_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int) round(sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j])));
            
            if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				proc_data->enrg_spec[spec_indx] += 0.0;
			}
			else {
                // Compute the normalization factor 1.0 / |k|^2
                k_sqr = 1.0 / ((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));
            
                if ((j == 0) || (j == Ny_Fourier - 1)) {
                    proc_data->enrg_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                }
                else {
                    proc_data->enrg_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
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
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
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
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute spectrum index/bin
			spec_indx = (int )round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				proc_data->enst_spec[spec_indx] += 0.0;
			}
			else {
				if ((j == 0) || (j == Ny_Fourier - 1)) {
					proc_data->enst_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					proc_data->enst_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
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
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double pre_fac = 0.0;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
	double tmp_deriv, tmp_diss;

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
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

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Compute the nonlinear term
	NonlinearRHS(run_data->w_hat, proc_data->dw_hat_dt, proc_data->nonlinterm, proc_data->nabla_psi, proc_data->nabla_w);

	// -------------------------------------
	// Compute the Energy Flux Spectrum
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				if ((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
				}
				else if ((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// No hyperviscosity but we have Ekman drag
					pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
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
				spec_indx = (int) ceil(sqrt(k_sqr));

				// Update spectrum bin
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the current bin sum 
					#if defined(__ENST_FLUX)
					tmp_deriv = creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * 4.0 * M_PI * M_PI * norm_fac;
					tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * 4.0 * M_PI * M_PI * norm_fac;
					proc_data->d_enst_dt_spec[spec_indx] += tmp_deriv;
					proc_data->enst_diss_spec[spec_indx] += tmp_diss; 
					proc_data->enst_flux_spec[spec_indx] += tmp_deriv - tmp_diss; 
					#endif
					#if defined(__ENRG_FLUX)
					tmp_deriv = creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * 4.0 * M_PI * M_PI * norm_fac / k_sqr;
					tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * 4.0 * M_PI * M_PI * norm_fac / k_sqr;
					proc_data->d_enrg_dt_spec[spec_indx] += tmp_deriv;
					proc_data->enrg_diss_spec[spec_indx] += tmp_diss; 
					proc_data->enrg_flux_spec[spec_indx] += tmp_deriv - tmp_diss; 
					#endif
					#if defined(__SEC_PHASE_SYNC)
					// Compute the enstrophy dissipation field
					proc_data->enst_diss_field[indx] = pre_fac * run_data->w_hat[indx];

					// Compute the enstrophy flux, dissipation and collective phase for C_theta
					if (k_sqr > sys_vars->kmax_C_sqr && k_sqr < sys_vars->kmax_sqr) {
						for (int a = 0; a < sys_vars->num_sect; ++a) {
							if (proc_data->phase_angle[indx] >= proc_data->theta[a] - proc_data->dtheta/2.0 && proc_data->phase_angle[indx] < proc_data->theta[a] + proc_data->dtheta/2.0) {
								tmp_deriv = creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * 4.0 * M_PI * M_PI * norm_fac;
								tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * 4.0 * M_PI * M_PI * norm_fac;

								// Record the flux and dissipation
								proc_data->enst_diss_C_theta[a] += tmp_diss;
								proc_data->enst_flux_C_theta[a] += tmp_deriv - tmp_diss;

								// Record the phase sync
								if (run_data->k[0][i] > 0 && cabs(tmp_deriv) != 0.0) {
									proc_data->phase_order_C_theta[a] += tmp_deriv / cabs(tmp_deriv);
								}
							}
						}
					}
					#endif
				}
				else {
					// Update the running sum for the flux
					#if defined(__ENST_FLUX)
					tmp_deriv = 2.0 * creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * 4.0 * M_PI * M_PI * norm_fac;
					tmp_diss  = 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * 4.0 * M_PI * M_PI * norm_fac;
					proc_data->d_enst_dt_spec[spec_indx] += tmp_deriv;
					proc_data->enst_diss_spec[spec_indx] += tmp_diss; 
					proc_data->enst_flux_spec[spec_indx] += tmp_deriv - tmp_diss; 
					#endif
					#if defined(__ENRG_FLUX)
					tmp_deriv = 2.0 * creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * 4.0 * M_PI * M_PI * norm_fac / k_sqr;
					tmp_diss  = 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * 4.0 * M_PI * M_PI * norm_fac / k_sqr;
					proc_data->d_enrg_dt_spec[spec_indx] += tmp_deriv;
					proc_data->enrg_diss_spec[spec_indx] += tmp_diss; 
					proc_data->enrg_flux_spec[spec_indx] += tmp_deriv - tmp_diss; 
					#endif
					#if defined(__SEC_PHASE_SYNC)
					// Compute the enstrophy dissipation field
					proc_data->enst_diss_field[indx] = pre_fac * run_data->w_hat[indx];

					// Compute the enstrophy flux, dissipation and collective phase for C_theta
					if (k_sqr > sys_vars->kmax_C_sqr && k_sqr < sys_vars->kmax_sqr) {
						for (int a = 0; a < sys_vars->num_sect; ++a) {
							if (proc_data->phase_angle[indx] >= proc_data->theta[a] - proc_data->dtheta/2.0 && proc_data->phase_angle[indx] < proc_data->theta[a] + proc_data->dtheta/2.0) {
								tmp_deriv = creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]) * 4.0 * M_PI * M_PI * norm_fac;
								tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * 4.0 * M_PI * M_PI * norm_fac;
								
								// Record the flux and dissipation
								proc_data->enst_diss_C_theta[a] += tmp_diss;
								proc_data->enst_flux_C_theta[a] += tmp_deriv - tmp_diss;

								// Record the phase sync
								if (cabs(tmp_deriv) != 0.0) {
									proc_data->phase_order_C_theta[a] += tmp_deriv / cabs(tmp_deriv);
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
		#if defined(__ENST_FLUX)
		proc_data->d_enst_dt_spec[i] += proc_data->d_enst_dt_spec[i - 1];
		proc_data->enst_flux_spec[i] += proc_data->enst_flux_spec[i - 1];
		proc_data->enst_diss_spec[i] += proc_data->enst_diss_spec[i - 1];
		#endif
		#if defined(__ENRG_FLUX)
		proc_data->d_enrg_dt_spec[i] += proc_data->d_enrg_dt_spec[i - 1];
		proc_data->enrg_flux_spec[i] += proc_data->enrg_flux_spec[i - 1];
		proc_data->enrg_diss_spec[i] += proc_data->enrg_diss_spec[i - 1];
		#endif
		if (i >= (int)(sys_vars->kmax_frac * sys_vars->kmax)) {
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
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;

	// --------------------------------	
	//  Allocate Full Field Arrays
	// --------------------------------
	#if defined(__FULL_FIELD)
	// Allocate memory for the full field phases
	proc_data->phases = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->phases == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field amplitudes
	proc_data->amps = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->amps == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field enstrophy spectrum
	proc_data->enst = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->enst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Enstrophy");
		exit(1);
	}	

	// Allocate memory for the full field enrgy spectrum
	proc_data->enrg = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->enrg == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Energy");
		exit(1);
	}	

	// Initialize arrays
	for (int i = 0; i < (2 * sys_vars->kmax - 1); ++i) {
		tmp3 = i * (2 * sys_vars->kmax - 1);
		for (int j = 0; j < (2 * sys_vars->kmax - 1); ++j) {
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
	sys_vars->n_spec = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0)) + 1;

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
	#if defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC)
	///-------------- Nonlinear RHS function arrays
	/// RHS
	proc_data->dw_hat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	if (proc_data->dw_hat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "RHS");
		exit(1);
	}
	// The gradient of the real space vorticity
	proc_data->nabla_w = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (proc_data->nabla_w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Gradient of Real Space Vorticity");
		exit(1);
	}
	// The gradient of the real stream function
	proc_data->nabla_psi = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (proc_data->nabla_psi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Gradient of Real Space Vorticity");
		exit(1);
	}
	// The nonlinear term in real space
	proc_data->nonlinterm = (double* )fftw_malloc(sizeof(double) * Nx * Ny);
	if (proc_data->nonlinterm == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Nonlinear Term in Real Spcace");
		exit(1);
	}

	// --------------------------------	
	//  Allocate C_\theta Enstrophy Flux
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
	proc_data->enst_diss_field = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
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
	// The enstrophy flux spectrum
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
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny; ++j) {
			if(j < Ny_Fourier) {
				proc_data->dw_hat_dt[SYS_DIM * (i * Ny_Fourier + j) + 0] = 0.0 + 0.0 * I;
				proc_data->dw_hat_dt[SYS_DIM * (i * Ny_Fourier + j) + 1] = 0.0 + 0.0 * I;
			}
			proc_data->nabla_w[SYS_DIM * (i * Ny + j) + 0]   = 0.0;
			proc_data->nabla_w[SYS_DIM * (i * Ny + j) + 1]   = 0.0;
			proc_data->nabla_psi[SYS_DIM * (i * Ny + j) + 0] = 0.0;
			proc_data->nabla_psi[SYS_DIM * (i * Ny + j) + 1] = 0.0;
			proc_data->nonlinterm[i * Ny + j]                = 0.0;
		}
	}
	for (int i = 0; i < sys_vars->num_snaps; ++i) {
		#if defined(__ENST_FLUX)
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
	#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC)
	fftw_free(proc_data->phases);
	fftw_free(proc_data->amps);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	#endif
	#if defined(__SPECTRA)
	fftw_free(proc_data->enst_spec);
    fftw_free(proc_data->enrg_spec);
	#endif
	#if defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC)
	fftw_free(proc_data->nabla_w);
	fftw_free(proc_data->nabla_psi);
	fftw_free(proc_data->dw_hat_dt);
	fftw_free(proc_data->nonlinterm);
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
