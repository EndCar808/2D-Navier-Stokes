/**
* @file utils.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the system measurables functions for the pseudospectral solver
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
#include "sys_msr.h"
#include "solver.h"
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to compute the system measurables such as energy, enstrophy, palinstrophy, helicity, energy and enstrophy dissipation rates, and spectra at once on the local processes for the current timestep
 * @param t 		The current time in the simulation
 * @param iter 		The index in the system arrays for the current timestep
 * @param RK_data 	Struct containing the integration varaiables needed for the nonlinear term function
 */
void ComputeSystemMeasurables(double t, int iter, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double k_sqr, pre_fac;
	fftw_complex u_z, v_z, div_u_z;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
    #if defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT)
    double lwr_sbst_lim_sqr = pow(LWR_SBST_LIM, 2.0);
    double upr_sbst_lim_sqr = pow(UPR_SBST_LIM, 2.0);
    #endif
    #if defined(__ENRG_FLUX) || defined(__ENST_FLUX) || defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT) 
    double tmp_deriv, tmp_diss;
    #endif
    #if defined(__PHASE_SYNC)
    double tmp_order;
    #endif


    // Record the initial time
    #if defined(__TIME)
	if (sys_vars->TRANS_ITERS_FLAG != TRANSIENT_ITERS) {
	    if (!(sys_vars->rank)) {
	    	run_data->time[iter] = t;
	    }
	}
	#endif

    // If adaptive stepping check if within memory limits
    if ((iter >= sys_vars->num_print_steps) && (iter % 100 == 0)) {
    	// Print warning to screen if we have exceeded the memory limits for the system measurables arrays
    	printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", iter, t);
    }

	// ------------------------------------
	// Initialize Measurables
	// ------------------------------------
	#if defined(__SYS_MEASURES)
	if (iter < sys_vars->num_print_steps) {
		// Initialize totals
		run_data->tot_enstr[iter]  = 0.0;
		run_data->tot_palin[iter]  = 0.0;
		run_data->tot_energy[iter] = 0.0;
		run_data->tot_forc[iter]   = 0.0;
		run_data->tot_div[iter]    = 0.0;
		run_data->enrg_diss[iter]  = 0.0;
		run_data->enst_diss[iter]  = 0.0;
		#if defined(__ENRG_FLUX)
		run_data->d_enrg_dt_sbst[iter] = 0.0;
		run_data->enrg_flux_sbst[iter] = 0.0;
		run_data->enrg_diss_sbst[iter] = 0.0;
		#endif
		#if defined(__ENST_FLUX)
		run_data->d_enst_dt_sbst[iter] = 0.0;
		run_data->enst_flux_sbst[iter] = 0.0;
		run_data->enst_diss_sbst[iter] = 0.0;
		#endif
	}
	#endif 
	#if defined(__ENRG_SPECT) || defined(__ENST_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	// Initialize spectra
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		#if defined(__ENRG_SPECT)
		run_data->enrg_spect[i] = 0.0;
		#endif
		#if defined(__ENST_SPECT)
		run_data->enst_spect[i] = 0.0;
		#endif
		#if defined(__ENST_FLUX_SPECT)
		run_data->d_enrg_dt_spect[i] = 0.0;
		run_data->enst_flux_spect[i] = 0.0;
		run_data->enst_diss_spect[i] = 0.0;
		#endif
		#if defined(__ENRG_FLUX_SPECT)
		run_data->d_enrg_dt_spect[i] = 0.0;
		run_data->enrg_flux_spect[i] = 0.0;
		run_data->enrg_diss_spect[i] = 0.0;
		#endif
		#if defined(__PHASE_SYNC)
		run_data->phase_order_k[i]        = 0.0 + 0.0 * I;
		run_data->normed_phase_order_k[i] = 0.0 + 0.0 * I;
		#endif
	}
	#endif

	#if defined(__ENRG_FLUX) || defined(__ENST_FLUX) || defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT)
	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, RK_data->RK1, RK_data->nonlin, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			RK_data->RK1[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}
	#endif

	// Compute the total forcing
	for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
		run_data->tot_forc[iter] += cabs(run_data->forcing[i] * run_data->w_hat[run_data->forcing_indx[i]]);
	}

	// -------------------------------------
	// Compute Measurables in Fourier Space
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			#if defined(__ENRG_SPECT) || defined(__ENST_SPECT) || defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__PHASE_SYNC)
			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );
			#endif

			// The |k|^2 prefactor
			k_sqr = (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

			// Get the appropriate prefactor
			if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {  
				// Both Hyperviscosity and Ekman drag
				pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
			}
			else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

			///--------------------------------- System Measures
			#if defined(__SYS_MEASURES)
		    if (iter < sys_vars->num_print_steps) {
				if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {

					// Get the fourier velocities
					u_z = I * ((double )run_data->k[1][j]) * run_data->w_hat[indx] / k_sqr;
					v_z = -I * ((double )run_data->k[0][i]) * run_data->w_hat[indx] / k_sqr;

					// Get the diverence of the Fourier velocity
					div_u_z = I * ((double )run_data->k[0][i] * u_z + (double )run_data->k[1][j] * v_z);

					// Update the sums
					if ((j == 0) || (j == Ny_Fourier - 1)) { // only count the 0 and N/2 modes once as they have no conjugate
						run_data->tot_energy[iter] += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
						run_data->tot_enstr[iter]  += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->tot_div[iter]    += cabs(div_u_z * conj(div_u_z));
						run_data->tot_palin[iter]  += k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->enrg_diss[iter]  += pre_fac * cabs(u_z * conj(u_z) + v_z * conj(v_z));
						run_data->enst_diss[iter]  += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						if ((k_sqr >= lwr_sbst_lim_sqr) && (k_sqr < upr_sbst_lim_sqr)) { // define the subset to consider for the flux and dissipation
							#if defined(__ENRG_FLUX) || defined(__ENST_FLUX)
							// Get the derivative and dissipation terms
							tmp_deriv = creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]);
							tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
							#endif

							#if defined(__ENRG_FLUX)
							run_data->d_enrg_dt_sbst[iter] += tmp_deriv * (1.0 / k_sqr);
							run_data->enrg_diss_sbst[iter] += tmp_diss * (1.0 / k_sqr);
							run_data->enrg_flux_sbst[iter] += (tmp_deriv - tmp_diss) * (1.0 / k_sqr);
							#endif
							#if defined(__ENST_FLUX)
							run_data->d_enst_dt_sbst[iter] += tmp_deriv; 
							run_data->enst_diss_sbst[iter] += tmp_diss;
							run_data->enst_flux_sbst[iter] += tmp_deriv - tmp_diss; 
							#endif
						}
					}
					else {
						run_data->tot_energy[iter] += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
						run_data->tot_enstr[iter]  += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->tot_div[iter]    += 2.0 * cabs(div_u_z * conj(div_u_z));
						run_data->tot_palin[iter]  += 2.0 * k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->enrg_diss[iter]  += 2.0 * pre_fac * cabs(u_z * conj(u_z) + v_z * conj(v_z));
						run_data->enst_diss[iter]  += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						if ((k_sqr >= lwr_sbst_lim_sqr) && (k_sqr < upr_sbst_lim_sqr)) { // define the subset to consider for the flux and dissipation
							#if defined(__ENRG_FLUX) || defined(__ENST_FLUX)
							// Get the derivative and dissipation terms
							tmp_deriv = creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]);
							tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
							#endif

							#if defined(__ENRG_FLUX)
							run_data->d_enrg_dt_sbst[iter] += tmp_deriv * (2.0 / k_sqr);
							run_data->enrg_diss_sbst[iter] += tmp_diss * (2.0 / k_sqr);
							run_data->enrg_flux_sbst[iter] += (tmp_deriv - tmp_diss) * (2.0 / k_sqr);
							#endif
							#if defined(__ENST_FLUX)
							run_data->d_enst_dt_sbst[iter] += 2.0 * tmp_deriv;
							run_data->enst_diss_sbst[iter] += 2.0 * tmp_diss; 
							run_data->enst_flux_sbst[iter] += 2.0 * (tmp_deriv - tmp_diss);
							#endif
						}
					}
				}
				else {
					continue;
				}
			}
			#endif

			///--------------------------------- Spectra
			#if defined(__ENRG_SPECT) || defined(__ENST_SPECT) || defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT)
			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				#if defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT)
				// Get the derivative and dissipation terms
				tmp_deriv = creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]) * const_fac * norm_fac;
				tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * const_fac * norm_fac;
				#endif

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the current bin
					#if defined(__ENRG_SPECT)
					run_data->enrg_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
					#endif
					#if defined(__ENST_SPECT)
					run_data->enst_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
					#endif
					#if defined(__ENST_FLUX_SPECT)
					run_data->d_enst_dt_spect[spec_indx] += tmp_deriv;
					run_data->enst_diss_spect[spec_indx] += tmp_diss;
					run_data->enst_flux_spect[spec_indx] += tmp_deriv - tmp_diss;
					#endif
					#if defined(__ENRG_FLUX_SPECT)
					run_data->d_enrg_dt_spect[spec_indx] += tmp_deriv * (1.0 / k_sqr);
					run_data->enrg_diss_spect[spec_indx] += tmp_diss * (1.0 / k_sqr);
					run_data->enrg_flux_spect[spec_indx] += (tmp_deriv - tmp_diss) * (1.0 / k_sqr);
					#endif
				}
				else {
					// Update the spectra sums for the current mode
					#if defined(__ENRG_SPECT)
					run_data->enrg_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
					#endif
					#if defined(__ENST_SPECT)
					run_data->enst_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
					#endif
					#if defined(__ENST_FLUX_SPECT)
					run_data->d_enst_dt_spect[spec_indx] += 2.0 * tmp_deriv;
					run_data->enst_diss_spect[spec_indx] += 2.0 * tmp_diss;
					run_data->enst_flux_spect[spec_indx] += 2.0 * (tmp_deriv - tmp_diss);
					#endif
					#if defined(__ENRG_FLUX_SPECT)
					run_data->d_enrg_dt_spect[spec_indx] += 2.0 * tmp_deriv / k_sqr;
					run_data->enrg_diss_spect[spec_indx] += 2.0 * tmp_diss / k_sqr;
					run_data->enrg_flux_spect[spec_indx] += (2.0 * tmp_deriv / k_sqr) - (2.0 * tmp_diss / k_sqr);
					#endif
				}
			}
			else {
				continue;
			}
			#endif

			///--------------------------------- Phase Sync
			#if defined(__PHASE_SYNC)
			// Compute the scale dependent collective phase
			tmp_order = RK_data->RK1[indx] * cexp(-I * carg(run_data->w_hat[indx]));

			// Average over the annulus
			if (j == 0 || j == Ny_Fourier - 1) {
				if (run_data->k[0][i] > 0) {
					run_data->phase_order_k[spec_indx] += tmp_order;
					if (cabs(tmp_order) != 0.0) {
						run_data->normed_phase_order_k[spec_indx] += tmp_order / cabs(tmp_order);
					}
					else {
						run_data->normed_phase_order_k[spec_indx] += 0.0 + 0.0 * I;	
					}
				}	
			}
			else {
				run_data->phase_order_k[spec_indx] += tmp_order;
				if (cabs(tmp_order) != 0.0) {
					run_data->normed_phase_order_k[spec_indx] += tmp_order / cabs(tmp_order);
				}
				else {
					run_data->normed_phase_order_k[spec_indx] += 0.0 + 0.0 * I;	
				}
			}
			#endif
		}
	}

	// ------------------------------------
	// Normalize Measureables 
	// ------------------------------------	
	#if defined(__SYS_MEASURES)
	if (iter < sys_vars->num_print_steps) {
		// Normalize results and take into account computation in Fourier space
		run_data->enrg_diss[iter]  *= 2.0 * const_fac * norm_fac;
		run_data->enst_diss[iter]  *= 2.0 * const_fac * norm_fac;
		run_data->tot_enstr[iter]  *= const_fac * norm_fac;
		run_data->tot_palin[iter]  *= const_fac * norm_fac;
		run_data->tot_forc[iter]   *= const_fac * norm_fac;
		run_data->tot_div[iter]    *= const_fac * norm_fac;
		run_data->tot_energy[iter] *= const_fac * norm_fac;
	}
	#endif
	#if defined(__ENRG_FLUX)
	run_data->enrg_flux_sbst[iter] *= const_fac * norm_fac;
	run_data->enrg_diss_sbst[iter] *= 2.0 * const_fac * norm_fac;
	#endif
	#if defined(__ENST_FLUX)
	run_data->enst_flux_sbst[iter] *= const_fac * norm_fac;
	run_data->enst_diss_sbst[iter] *= 2.0 * const_fac * norm_fac;
	#endif
}
/**
 * Function to initialize and compute the system measurables and spectra of the initial conditions
 * @param RK_data The struct containing the Runge-Kutta arrays to compute the nonlinear term for the fluxes
 */
void InitializeSystemMeasurables(RK_data_struct* RK_data) {

	// Set the size of the arrays to twice the number of printing steps to account for extra steps due to adaptive stepping
	if (sys_vars->ADAPT_STEP_FLAG == ADAPTIVE_STEP) {
		sys_vars->num_print_steps = 2 * sys_vars->num_print_steps;
	}
	else {
		sys_vars->num_print_steps = sys_vars->num_print_steps;
	}
	int print_steps = sys_vars->num_print_steps;

	// Get the size of the spectrum arrays
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT) || defined(__PHASE_SYNC)
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];

	sys_vars->n_spect = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0)) + 1;
	int n_spect = sys_vars->n_spect;
	#endif


	// --------------------------------
	// Allocate Phase Sync Memory
	// --------------------------------
	#if defined(__PHASE_SYNC)
	// Scale Dependent Collective Phase
	run_data->phase_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->n_spect);
	if (run_data->phase_order_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Scale Dependent Collective Phase");
		exit(1);
	}

	// Normalized Scale Dependent Collective Phase
	run_data->normed_phase_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->n_spect);
	if (run_data->normed_phase_order_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Normed Scale Dependent Collective Phase");
		exit(1);
	}
	#endif
	
	// --------------------------------
	// Allocate System Totals Memory
	// --------------------------------
	#if defined(__SYS_MEASURES)
	// Total Energy in the system
	run_data->tot_energy = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_energy == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Energy");
		exit(1);
	}	

	// Total Enstrophy
	run_data->tot_enstr = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_enstr == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Enstrophy");
		exit(1);
	}	

	// Total Palinstrophy
	run_data->tot_palin = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_palin == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Palinstrophy");
		exit(1);
	}	

	// Total Forcing Input
	run_data->tot_forc = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_forc == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Forcing Input");
		exit(1);
	}	

	// Total Divergence
	run_data->tot_div = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_div == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Divergence");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate");
		exit(1);
	}	
	#endif

	// --------------------------------
	// Allocate Enstrophy Spec Memory
	// --------------------------------
	#if defined(__ENST_SPECT)
	// Enstrophy Spectrum
	run_data->enst_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Spectrum");
		exit(1);
	}	
	#endif

	// --------------------------------
	// Allocate Energy Spec Memory
	// --------------------------------
	#if defined(__ENRG_SPECT)
	// Energy Spectrum
	run_data->enrg_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Spectrum");
		exit(1);
	}	
	#endif
	
	// --------------------------------
	// Allocate Enstrophy Flux Memory
	// --------------------------------
	#if defined(__ENST_FLUX)
	// Time derivative of the enstrophy
	run_data->d_enst_dt_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->d_enst_dt_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time Derivative of Enstrophy Subset");
		exit(1);
	}

	// Enstrophy flux
	run_data->enst_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Subset");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate Subset");
		exit(1);
	}	
	#endif
	#if defined(__ENST_FLUX_SPECT)
	// Time derivative of Enstrophy Spectrum
	run_data->d_enst_dt_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->d_enst_dt_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time Derivative of Enstrophy Spectrum");
		exit(1);
	}
	// Enstrophy Spectrum
	run_data->enst_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}
	// Enstrophy Dissipation Spectrum
	run_data->enst_diss_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_diss_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Spectrum");
		exit(1);
	}
	#endif

	// --------------------------------
	// Allocate Energy Spec Memory
	// --------------------------------
	#if defined(__ENRG_FLUX)
	// Time derivative of energy 
	run_data->d_enrg_dt_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->d_enrg_dt_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time Derivative of Energy Subset");
		exit(1);
	}

	// Energy Flux
	run_data->enrg_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Subset");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate Subset");
		exit(1);
	}
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	// Energy Flux Spectrum
	run_data->enrg_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}
	// Time derivative of Energy Spectrum
	run_data->d_enrg_dt_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->d_enrg_dt_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time Derivative of Energy Spectrum");
		exit(1);
	}
	// Energy Dissipation Spectrum
	run_data->enrg_diss_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_diss_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Spectrum");
		exit(1);
	}	
	#endif

	// Time
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		run_data->time = (double* )fftw_malloc(sizeof(double) * print_steps);
		if (run_data->time == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time");
			exit(1);
		}	
	}
	#endif

	// ----------------------------
	// Get Measurables of the ICs
	// ----------------------------
	if (sys_vars->TRANS_ITERS_FLAG != TRANSIENT_ITERS) {
		ComputeSystemMeasurables(0.0, 0, RK_data);
		// RecordSystemMeasures(0.0, 0, RK_data);
	}
}
/**
 * Function used to compute the energy spectrum of the current iteration. The energy spectrum is defined as all(sum) of the energy contained in concentric annuli in
 * wavenumber space. 	
 */
void EnergySpectrum(void) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	double k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enrg_spect[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				run_data->enrg_spect[spec_indx] += 0.0;
			}
			else {
				// Compute |k|^2
				k_sqr = 1.0 / ((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the current bin for mode
					run_data->enrg_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
				else {
					// Update the energy sum for the current mode
					run_data->enrg_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
			}
		}
	}
}
/**
 * Function used to compute the enstrophy spectrum for the current iteration. The enstrophy spectrum is defined as the total enstrophy contained in concentric annuli 
 * in wavenumber space
 */
void EnstrophySpectrum(void) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enst_spect[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				run_data->enst_spect[spec_indx] += 0.0;
			}
			else {
				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the current bin for mode
					run_data->enst_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					// Update the enstrophy sum for the current mode
					run_data->enst_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
		}
	}
}
/**
 * Function to compute the total divergence of the velocity field
 * @return  Total divergence
 */
double TotalDivergence(void) {

	// Initialize variables
	int tmp;
	int indx;
	fftw_complex k_sqr_inv;
	fftw_complex u_z, v_z, div_u_z;
	double tot_div = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);


	// ------------------------------------------
	// Compute Fourier Space Velocity & Divergence
	// ------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// The I/k^2 prefactor
				k_sqr_inv = 1.0 / (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the fourier velocities
				u_z = k_sqr_inv * ((double) run_data->k[1][j]) * run_data->w_hat[indx];
				v_z = -k_sqr_inv * ((double) run_data->k[0][i]) * run_data->w_hat[indx];

				// Get the divergence of u_z
				div_u_z = I * ((double )run_data->k[0][i] * u_z + (double )run_data->k[1][j] * v_z);

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the sum for the total energy
					tot_div += cabs(div_u_z * conj(div_u_z));
				}
				else {
					// Update the sum for the total energy
					tot_div += 2.0 * cabs(div_u_z * conj(div_u_z));
				}
			}
			else {
				tot_div += 0.0;
			}
		}
	}
	
	// Return result
	return 4.0 * M_PI * M_PI * tot_div * norm_fac;
}
/**
 * Function to compute the total forcing input into the system
 * @return  The total forcing input
 */
double TotalForcing(void) {

	// Initialize variables
	double tot_forcing = 0.0;
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------------
	// Compute The Total Forcing
	// ------------------------------------------
	for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
		tot_forcing += 4.0 * cabs(sys_vars->forcing[i] * conj(run_data->w_hat[run_data->forcing_indx[i]]));
	}

	return norm_fac * tot_forcing;
}
/**
 * Function to compute the total energy in the system at the current timestep
 * @return  The total energy in the system
 */
double TotalEnergy(void) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double tot_energy = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------------
	// Compute Fourier Space Velocity & Energy
	// ------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// The 1/k^2 prefactor
				k_sqr = 1.0 / (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the sum for the total energy
					tot_energy += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
				else {
					// Update the sum for the total energy
					tot_energy += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
			}
			else {
				tot_energy += 0.0;
			}
		}
	}
	
	// Return result
	return 4.0 * M_PI * M_PI * tot_energy * norm_fac;
}
/**
 * Function to compute the total enstrophy in the system at the current timestep
 * @return  The total enstrophy in the system
 */
double TotalEnstrophy(void) {

	// Initialize variables
	int tmp;
	int indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -------------------------------
	// Compute The Total Energy 
	// -------------------------------
	// Initialize total enstrophy
	double tot_enstr = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 1; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((j == 0) || (j == Ny_Fourier - 1)) {
				// Update the sum for the total enstrophy -> only count the 0 and N/2 modes once as they have no conjugate
				tot_enstr += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}	
			else {
				// Update the sum for the total enstrophy -> factor of two for Fourier conjugates
				tot_enstr += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}	
		}
	}

	// Return result
	return 4.0 * M_PI * M_PI * tot_enstr * norm_fac;
}
/**
 * Function to compute the total palinstrophy at the current timestep on the local process.
 * Results are gathered on the master process at the end of the simulation run
 * @return  The total palinstrophy
 */
double TotalPalinstrophy(void) {
	
	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -------------------------------
	// Compute The Total Palinstrophy 
	// -------------------------------
	// Initialize total enstrophy
	double tot_palin = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Get the |k|^2 prefactor
				k_sqr = 1.0 * (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Update the running sum for the palinstrophy
				if((j == 0) || (j == Ny_Fourier - 1)) {
					tot_palin += k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					tot_palin += 2.0 * k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
			else {
				tot_palin += 0.0;
			}
		}
	}

	// Return result
	return 4.0 * M_PI * M_PI * tot_palin * norm_fac;
}
/**
 * Function to compute the energy dissipation rate \epsilon for the current iteration on the local processes
 * The results are gathered on the master process at the end of the simulation
 * @return  Returns the energy dissipation rate
 */
double EnergyDissipationRate(void) {

	// Initialize variables
	int tmp;
	int indx;
	double pre_fac;
	double k_sqr;
	fftw_complex tmp_u_x, tmp_u_y;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -------------------------------
	// Compute The Energy Diss Rate
	// -------------------------------
	// Initialize total enstrophy
	double enrgy_diss = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute |k|^2
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Get the Fourier velocities
				tmp_u_x = (I / k_sqr) * (run_data->k[1][j] * run_data->w_hat[indx]);
				tmp_u_y = (I / k_sqr) * (-run_data->k[0][i] * run_data->w_hat[indx]);

				// Get the appropriate prefactor
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) { 
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

				// Update the running sum for the palinstrophy -> first and last modes have no conjugate so only count once
				if((j == 0) || (j == Ny_Fourier - 1)) {
					enrgy_diss += pre_fac * cabs(tmp_u_x * conj(tmp_u_x) + tmp_u_y * conj(tmp_u_y));
				}
				else {
					enrgy_diss += 2.0 * pre_fac * cabs(tmp_u_x * conj(tmp_u_x) + tmp_u_y * conj(tmp_u_y));
				}
			}
		}
	}

	// Return result -> 2 (nu * 0.5 *<|w|^2>)
	return 2.0 * (4.0 * M_PI * M_PI * enrgy_diss * norm_fac);
}
/**
 * Function to compute the enstrophy dissipation rate \eta which is equal to 2 * Palinstrophy for the current iteration on the local process
 * Results are gathered on the master process at the end of the simulation run
 * @return  The enstrophy dissipation rate on the local process
 */
double EnstrophyDissipationRate(void) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double pre_fac;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -------------------------------
	// Compute The Enstrophy Diss Rate 
	// -------------------------------
	// Initialize total enstrophy
	double tot_enst_diss = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = 1.0 * (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) { 
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

				// Update the running sum for the enst_dissstrophy -> first and last modes have no conjugate so only count once
				if((j == 0) || (j == Ny_Fourier - 1)) {
					tot_enst_diss += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					tot_enst_diss += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
			else {
				tot_enst_diss += 0.0;
			}
		}
	}


	// Return result -> 2(enst_diss) = 2(0.5 * <|grad \omega|^2>)
	return 2.0 * (4.0 * M_PI * M_PI * tot_enst_diss * norm_fac);
}	
/**
 * Function to compute the enstrophy flux and enstrophy dissipation from a subset of modes on the local process for the current iteration 
 * The results will be gathered on the master process at the end of the simulation	
 * @param enst_flux The enstrophy flux 
 * @param enst_diss The enstrophy dissipation
 * @param RK_data   The Runge-Kutta struct containing the arrays for computing the nonlinear term
 */
void EnstrophyFlux(double* d_e_dt, double* enst_flux, double* enst_diss, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double pre_fac;
	double tmp_d_e_dt;
	double tmp_enst_flux;
	double tmp_enst_diss;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
	double tmp_deriv, tmp_diss;

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Flux");
		exit(1);
	}

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nonlin, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

	// -------------------------------------
	// Compute the Enstrophy Flux & Diss
	// -------------------------------------
	// Initialize sums
	tmp_d_e_dt    = 0.0;
	tmp_enst_flux = 0.0;
	tmp_enst_diss = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Consider only a subset of modes
			if (((run_data->k[1][j] >= LWR_SBST_LIM) && (run_data->k[1][j] <= UPR_SBST_LIM)) && (abs(run_data->k[0][i]) <= UPR_SBST_LIM) && (abs(run_data->k[0][i]) >= LWR_SBST_LIM)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) { 
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

				// Get the derivative and dissipation terms
				tmp_deriv = creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);
				tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));

				// Update sums
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the running sum for the flux of enstrophy
					tmp_d_e_dt += tmp_deriv; 

					// Update the running sum for the enstrophy dissipation 
					tmp_enst_diss += tmp_diss;

					// Update the flux
					tmp_enst_flux += tmp_deriv - tmp_diss;
				}
				else {
					// Update the running sum for the flux of enstrophy
					tmp_d_e_dt += 2.0 * tmp_deriv;

					// Update the running sum for the enstrophy dissipation 
					tmp_enst_diss += 2.0 * tmp_diss;

					// Update the flux
					tmp_enst_flux += 2.0 * (tmp_deriv - tmp_diss);
				}
			}
		}
	}	

	// -----------------------------------
	// Compute the Enstrophy Flux & Diss 
	// -----------------------------------
	// Compue the enstrophy dissipation
	(*d_e_dt) = tmp_d_e_dt * norm_fac;

	// Compue the enstrophy dissipation
	(*enst_diss) = 4.0 * M_PI * M_PI * tmp_enst_diss * norm_fac;

	// Compute the enstrophy flux
	(*enst_flux) = 8.0 * M_PI * M_PI * tmp_enst_flux * norm_fac;

	// -----------------------------------
	// Free memory
	// -----------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the energy flux and dissipation in/out of a subset of modes for the current iteration on the local processes
 * The results will be gathered on the master process and written to file at the end of the simulation
 * @param enrg_flux The energy flux in/out of a subset of modes	
 * @param enrg_diss The energy dissipation of a subset of modes
 * @param RK_data   The struct containing the Runge-Kutta arrays to be used for computing the nonlinear term
 */
void EnergyFlux(double* d_e_dt, double* enrg_flux, double* enrg_diss, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double pre_fac;
	double tmp_d_e_dt;
	double tmp_enrgy_flux;
	double tmp_enrgy_diss;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
	double tmp_deriv, tmp_diss;

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Flux");
		exit(1);
	}

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nonlin, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

	// -------------------------------------
	// Compute the Energy Flux & Diss
	// -------------------------------------
	// Initialize sums
	tmp_d_e_dt     = 0.0;
	tmp_enrgy_flux = 0.0;
	tmp_enrgy_diss = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Consider only a subset of modes
				if (((run_data->k[1][j] >= LWR_SBST_LIM) && (run_data->k[1][j] <= UPR_SBST_LIM)) && (abs(run_data->k[0][i]) <= UPR_SBST_LIM) && (abs(run_data->k[0][i]) >= LWR_SBST_LIM)) {
					// Compute |k|^2
					k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Get the appropriate prefactor
					if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) { 
						// Both Hyperviscosity and Ekman drag
						pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
					}
					else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

					// Get the derivative and dissipation terms
					tmp_deriv = creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;
					tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) / k_sqr;

					// Update sums
					if ((j == 0) || (Ny_Fourier - 1)) {
						// Update the running sum for the time derivative of energy
						tmp_d_e_dt += tmp_deriv;

						// Update the running sum for the energy dissipation 
						tmp_enrgy_diss += tmp_diss;

						// Update the flux
						tmp_enrgy_flux += tmp_deriv - tmp_diss;
					}
					else {
						// Update the running sum for the time derivative of energy
						tmp_d_e_dt += 2.0 * tmp_deriv;

						// Update the running sum for the energy dissipation 
						tmp_enrgy_diss += 2.0 * tmp_diss;

						// Update the flux
						tmp_enrgy_flux += 2.0 * (tmp_deriv - tmp_diss);
					}
				}
			}
			else {
				tmp_d_e_dt     += 0.0;
				tmp_enrgy_diss += 0.0;
				tmp_enrgy_flux += 0.0;
			}
		}
	}

	// -----------------------------------
	// Compute the Energy Flux & Diss 
	// -----------------------------------
	// Compue the time derivative of energy
	(*d_e_dt) = tmp_d_e_dt * norm_fac;

	// Compue the energy dissipation
	(*enrg_diss) = 4.0 * M_PI * M_PI * tmp_enrgy_diss * norm_fac;

	// Compute the energy flux
	(*enrg_flux) = 8.0 * M_PI * M_PI * tmp_enrgy_flux * norm_fac;


	// -----------------------------------
	// Free memory
	// -----------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the energy flux spectrum for the current iteration on the local processes
 * The results of the complete spectrum will be gathered on the master rank before writing to file
 * @param  RK_data The struct containing the Runge-Kutta arrays needed for computing the nonlinear term
 */
void EnergyFluxSpectrum(RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double pre_fac = 0.0;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
	double tmp_deriv, tmp_diss;

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enrg_flux_spect[i] = 0.0;
	}

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear term array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nonlin, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

	// -------------------------------------
	// Compute the Energy Flux Spectrum
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) { 
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

				// Get the derivative and dissipation terms
				tmp_deriv = 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;
				tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) / k_sqr;

				// Update spectrum bin
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the running sum for the flux of energy
					run_data->d_enrg_dt_spect[spec_indx] += tmp_deriv;
					run_data->enrg_diss_spect[spec_indx] += tmp_diss;
					run_data->enrg_flux_spect[spec_indx] += tmp_deriv - tmp_diss;
				}
				else {
					// Update the running sum for the flux of energy
					run_data->d_enrg_dt_spect[spec_indx] += 2.0 * tmp_deriv;
					run_data->enrg_diss_spect[spec_indx] += 2.0 * tmp_diss;
					run_data->enrg_flux_spect[spec_indx] += 2.0 * (tmp_deriv - tmp_diss);
				}
			}
		}
	}

	// -------------------------------------
	// Free Temp Memory
	// -------------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the enstrophy flux spectrum for the current iteration on the local processes
 * The results are gathered on the master rank before being written to file
 * @param  RK_data The struct containing the Runge-Kutta arrays for computing the nonlinear term
 */
void EnstrophyFluxSpectrum(RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double pre_fac = 0.0;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
	double tmp_deriv, tmp_diss;

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enst_flux_spect[i] = 0.0;

	}
	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear term array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nonlin, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

	// -------------------------------------
	// Compute the Energy Flux Spectrum
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) { 
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW);
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
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

				// Get the derivative and flux terms
				tmp_deriv = 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);
				tmp_diss  = pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));

				// Update spectrum bin
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the current bin sum 
					run_data->d_enst_dt_spect[spec_indx] += tmp_deriv;
					run_data->enst_diss_spect[spec_indx] += tmp_diss;
					run_data->enst_flux_spect[spec_indx] += tmp_deriv - tmp_diss;
				}
				else {
					// Update the running sum for the flux of energy
					run_data->d_enst_dt_spect[spec_indx] += 2.0 * tmp_deriv;
					run_data->enst_diss_spect[spec_indx] += 2.0 * tmp_diss;
					run_data->enst_flux_spect[spec_indx] += 2.0 * (tmp_deriv - tmp_diss);
				}
			}
		}
	}

	// -------------------------------------
	// Free Temp Memory
	// -------------------------------------
	fftw_free(dwhat_dt);
}
/**
 * DEPRICATED: Function to record the system measures for the current timestep 
 * @param t          The current time in the simulation
 * @param print_indx The current index of the measurables arrays
 * @param RK_data 	 The Runge-Kutta struct containing the arrays to compute the nonlinear term for the fluxes
 */
void RecordSystemMeasures(double t, int print_indx, RK_data_struct* RK_data) {

	// -------------------------------
	// Record the System Measures 
	// -------------------------------
	// The integration time
	#if defined(__TIME)
	if (!(sys_vars->rank)) {
		run_data->time[print_indx] = t;
	}
	#endif

	// Check if within memory limits
	if (print_indx < sys_vars->num_print_steps) {
		#if defined(__SYS_MEASURES)
		// Total Energy, enstrophy and palinstrophy
		run_data->tot_enstr[print_indx]  = TotalEnstrophy();
		run_data->tot_energy[print_indx] = TotalEnergy();
		run_data->tot_palin[print_indx]  = TotalPalinstrophy();
		// Total Forcing and Divergence
		run_data->tot_forc[print_indx] = TotalForcing();
		run_data->tot_div[print_indx]    = TotalDivergence();
		// Energy and enstrophy dissipation rates
		run_data->enrg_diss[print_indx] = EnergyDissipationRate();
		run_data->enst_diss[print_indx] = EnstrophyDissipationRate();
		#endif
		#if defined(__ENST_FLUX)
		// Enstrophy and energy flux in/out and dissipation of a subset of modes
		EnstrophyFlux(&(run_data->d_enst_dt_sbst[print_indx]), &(run_data->enst_flux_sbst[print_indx]), &(run_data->enst_diss_sbst[print_indx]), RK_data);
		#endif
		#if defined(__ENRG_FLUX)
		EnergyFlux(&(run_data->d_enrg_dt_sbst[print_indx]), &(run_data->enrg_flux_sbst[print_indx]), &(run_data->enrg_diss_sbst[print_indx]), RK_data);
		#endif
	}
	else {
		printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", print_indx, t);
	}
	
	// -------------------------------
	// Record the Spectra 
	// -------------------------------
	// Call spectra functions
	#if defined(__ENST_SPECT )
	EnstrophySpectrum();
	#endif
	#if defined(__ENRG_SPECT)
	EnergySpectrum();
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	EnergyFluxSpectrum(RK_data);
	#endif
	#if defined(__ENST_FLUX_SPECT)
	EnstrophyFluxSpectrum(RK_data);
	#endif
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------