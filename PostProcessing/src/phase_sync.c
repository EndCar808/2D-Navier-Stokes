/**
* @file stats.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the phase sync functions for the pseudospectral solver data
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
#include "utils.h"
#include "data_types.h"
#include "phase_sync.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to compute the phase synchroniztion, enstrophy flux etc in the set \amthcal{S}^{U} for the current snaphsot. 
 * What is computed here is the equivalent of summming over all setcors.
 * @param s The index of the current snapshot
 */
void PhaseSync(int s) {

	// Initialize variables
	int k3_x, k3_y, k2_x, k2_y, k1_x, k1_y;
	double k3_sqr, k1_sqr, k2_sqr;
	double flux_pre_fac, flux_wght;
	double triad_phase, gen_triad_phase;
	int tmp_k1, tmp_k2, tmp_k3;
	double flux_term;

	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		// Normalize the phase order parameters
		proc_data->num_triads_test[i]        = 0;
		proc_data->triad_R_test[i]           = 0.0;
		proc_data->triad_Phi_test[i]         = 0.0;
		proc_data->enst_flux_test[i]         = 0.0;
		proc_data->triad_phase_order_test[i] = 0.0 + 0.0 * I;
	}

	int n = 0;

	// Loop through the k wavevector (k is the k3 wavevector)
	for (int tmp_k3_x = 0; tmp_k3_x <= sys_vars->N[0] - 1; ++tmp_k3_x) {
		
		// Get k3_x
		k3_x = tmp_k3_x - (int) (sys_vars->N[0] / 2) + 1;

		for (int tmp_k3_y = 0; tmp_k3_y <= sys_vars->N[1] - 1; ++tmp_k3_y) {
			
			// Get k3_y
			k3_y = tmp_k3_y - (int) (sys_vars->N[1] / 2) + 1;

			// Get polar coords for the k wavevector
			k3_sqr       = (double) (k3_x * k3_x + k3_y * k3_y);
			
			if ((k3_sqr > sys_vars->kmax_C_sqr && k3_sqr <= sys_vars->kmax_sqr)) {

				// Loop through the k1 wavevector
				for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
					
					// Get k1_x
					k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

					for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
						
						// Get k1_y
						k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

						// Get polar coords for k1
						k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);

						if((k1_sqr > 0.0 && k1_sqr <= sys_vars->kmax_C_sqr)) {
							
							// Find the k2 wavevector
							k2_x = k3_x - k1_x;
							k2_y = k3_y - k1_y;
							
							// Get polar coords for k2
							k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);

							if ((k2_sqr > 0.0 && k2_sqr <= sys_vars->kmax_C_sqr)) {

								// Get correct phase index -> recall that to access kx > 0, use -kx
								tmp_k1 = (sys_vars->kmax - k1_x) * (2 * sys_vars->kmax + 1);	
								tmp_k2 = (sys_vars->kmax - k2_x) * (2 * sys_vars->kmax + 1);
								tmp_k3 = (sys_vars->kmax - k3_x) * (2 * sys_vars->kmax + 1);

								// Compute the flux pre factor
								flux_pre_fac = (double) (k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

								// Get the flux weight term
								flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax + k2_y] * proc_data->amps[tmp_k3 + sys_vars->kmax + k3_y]);

								// Get the triad phase
								triad_phase = proc_data->phases[tmp_k1 + sys_vars->kmax + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax + k2_y] - proc_data->phases[tmp_k3 + sys_vars->kmax + k3_y];

								// Define the generalized triad phase for the first term in the flux
								gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

								// Get flux term
								flux_term = flux_wght * cos(triad_phase);

								proc_data->num_triads_test[0]++;
								proc_data->enst_flux_test[0]         += flux_term;
								proc_data->triad_phase_order_test[0] += cexp(I * gen_triad_phase);

								// Record the wavevector data
								proc_data->phase_sync_wave_vecs_test[K1_X][n] = k1_x;
								proc_data->phase_sync_wave_vecs_test[K1_Y][n] = k1_y;
								proc_data->phase_sync_wave_vecs_test[K2_X][n] = k2_x;
								proc_data->phase_sync_wave_vecs_test[K2_Y][n] = k2_y;
								proc_data->phase_sync_wave_vecs_test[K3_X][n] = k3_x;
								proc_data->phase_sync_wave_vecs_test[K3_Y][n] = k3_y;

								n++;
								// if (n > sys_vars->num_triad_per_sec_est) {
								// 	fprintf(stderr, "\n["RED"ERROR"RESET"] --- The number of triads in ["CYAN"%s"RESET"] exceding the allocated number of ["CYAN"%d"RESET"] -- Need to allocate more memory\n-->> Exiting!!!\n", "PhaseSync", sys_vars->num_triad_per_sec_est);
								// 	exit(1);
								// }

								if (flux_pre_fac < 0) {
									//------------------------------------------ TRIAD TYPE 1
									proc_data->num_triads_test[1]++;		
									proc_data->enst_flux_test[1]         += flux_term;
									proc_data->triad_phase_order_test[1] += cexp(I * gen_triad_phase);
				
								}
								else if (flux_pre_fac > 0) {									
									//------------------------------------------ TRIAD TYPE 2
									proc_data->num_triads_test[2]++;		
									proc_data->enst_flux_test[2]         += flux_term;
									proc_data->triad_phase_order_test[2] += cexp(I * gen_triad_phase);
								}
								else if (flux_pre_fac == 0.0 || flux_wght == 0.0) {

									// Define the generalized triad phase for the zero contribution terms
									gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

									//------------------------------------------ TRIAD TYPE 5
									proc_data->num_triads_test[5]++;		
									proc_data->enst_flux_test[5]         += flux_term;
									proc_data->triad_phase_order_test[5] += cexp(I * gen_triad_phase);
								}
								else {
									// Define the generalized triad phase for the ignored terms
									gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

									//------------------------------------------ TRIAD TYPE 6
									proc_data->num_triads_test[6]++;		
									proc_data->enst_flux_test[6]         += flux_term;
									proc_data->triad_phase_order_test[6] += cexp(I * gen_triad_phase);
								}
							}
						}
					}
				}
			}
			else if ((k3_sqr > 0.0 && k3_sqr <= sys_vars->kmax_C_sqr)) {

				// Loop through the k1 wavevector
				for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
					
					// Get k1_x
					k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

					for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
						
						// Get k1_y
						k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

						// Get polar coords for k1
						k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);

						if((k1_sqr > sys_vars->kmax_C_sqr && k1_sqr <= sys_vars->kmax_sqr)) {									
							
							// Find the k2 wavevector
							k2_x = k3_x - k1_x;
							k2_y = k3_y - k1_y;
							
							// Get polar coords for k2
							k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);

							if ((k2_sqr > sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_sqr)) {

								// Get correct phase index -> recall that to access kx > 0, use -kx
								tmp_k1 = (sys_vars->kmax - k1_x) * (2 * sys_vars->kmax + 1);	
								tmp_k2 = (sys_vars->kmax - k2_x) * (2 * sys_vars->kmax + 1);
								tmp_k3 = (sys_vars->kmax - k3_x) * (2 * sys_vars->kmax + 1);

								// Compute the flux pre factor
								flux_pre_fac = (double) (k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

								// Get the flux weight term
								flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax + k2_y] * proc_data->amps[tmp_k3 + sys_vars->kmax + k3_y]);

								// Get the triad phase
								triad_phase = proc_data->phases[tmp_k1 + sys_vars->kmax + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax + k2_y] - proc_data->phases[tmp_k3 + sys_vars->kmax + k3_y];

								// Define the generalized triad phase for the first term in the flux
								gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

								// Get flux term
								flux_term = flux_wght * cos(triad_phase);

								proc_data->num_triads_test[0]++;
								proc_data->enst_flux_test[0]         += -flux_term;
								proc_data->triad_phase_order_test[0] += cexp(I * gen_triad_phase);
								
								// Record the wavevector data
								proc_data->phase_sync_wave_vecs_test[K1_X][n] = k1_x;
								proc_data->phase_sync_wave_vecs_test[K1_Y][n] = k1_y;
								proc_data->phase_sync_wave_vecs_test[K2_X][n] = k2_x;
								proc_data->phase_sync_wave_vecs_test[K2_Y][n] = k2_y;
								proc_data->phase_sync_wave_vecs_test[K3_X][n] = k3_x;
								proc_data->phase_sync_wave_vecs_test[K3_Y][n] = k3_y;

								n++;
								// if (n > sys_vars->num_triad_per_sec_est) {
								// 	fprintf(stderr, "\n["RED"ERROR"RESET"] --- The number of triads in ["CYAN"%s"RESET"] exceding the allocated number of ["CYAN"%d"RESET"] -- Need to allocate more memory\n-->> Exiting!!!\n", "PhaseSync", sys_vars->num_triad_per_sec_est);
								// 	exit(1);
								// }

								if (flux_pre_fac < 0) {
									//------------------------------------------ TRIAD TYPE 3
									proc_data->num_triads_test[3]++;		
									proc_data->enst_flux_test[3]         += -flux_term;
									proc_data->triad_phase_order_test[3] += cexp(I * gen_triad_phase);
				
								}
								else if (flux_pre_fac > 0) {									
									//------------------------------------------ TRIAD TYPE 4
									proc_data->num_triads_test[4]++;		
									proc_data->enst_flux_test[4]         += -flux_term;
									proc_data->triad_phase_order_test[4] += cexp(I * gen_triad_phase);
								}
								else if (flux_pre_fac == 0.0 || flux_wght == 0.0) {

									// Define the generalized triad phase for the zero contribution terms
									gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

									//------------------------------------------ TRIAD TYPE 5
									proc_data->num_triads_test[5]++;		
									proc_data->enst_flux_test[5]         += -flux_term;
									proc_data->triad_phase_order_test[5] += cexp(I * gen_triad_phase);
								}
								else {
									// Define the generalized triad phase for the ignored terms
									gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

									//------------------------------------------ TRIAD TYPE 6
									proc_data->num_triads_test[6]++;		
									proc_data->enst_flux_test[6]         += -flux_term;
									proc_data->triad_phase_order_test[6] += cexp(I * gen_triad_phase);
								}
							}
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		// Normalize the phase order parameters
		if (proc_data->num_triads_test[i] != 0) {
			proc_data->triad_phase_order_test[i] /= proc_data->num_triads_test[i];
		}
		
		// Record the phase syncs and average phases
		proc_data->triad_R_test[i]   = cabs(proc_data->triad_phase_order_test[i]);
		proc_data->triad_Phi_test[i] = carg(proc_data->triad_phase_order_test[i]);
	}
}
/**
 * Function to compute the phase synchroniztion data sector by sector in wavenumber space for the current snaphsot
 * @param s 			   The index of the current snapshot
 * @param pre_compute_flag Flag to indicate
 */
void PhaseSyncSector(int s, int pre_compute_flag) {

	// Initialize variables
	int k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;
	int tmp_k1, tmp_k2, tmp_k3;
	double k1_sqr, k2_sqr;
	double flux_pre_fac;
	double flux_wght;
	double triad_phase;
	double gen_triad_phase; 
	int gsl_status;
	double flux_term;
	double phase_val[NUM_TRIAD_CLASS];
	fftw_complex collective_phase_term;
	double max_bin;

	// Set the in time histogram bins to 0
	if (pre_compute_flag != PRE_COMPUTE) {
		#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_FLUX_STATS)
		for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
			for (int types = 0; types < NUM_TRIAD_TYPES; ++types) {
				#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
				gsl_histogram_reset(proc_data->triads_all_pdf_t[triad_class][types]);
				gsl_histogram_reset(proc_data->triads_wghtd_all_pdf_t[triad_class][types]);
				#endif

				#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
				if (types < 5) {
					// Reset 2d for current iteration and set ranges
					if (s > 1) {
						gsl_histogram2d_reset(proc_data->triads_wghtd_2d_pdf_t[triad_class][types]);
					}
					max_bin = proc_data->max_enst_flux[triad_class][types][s];
					// max_bin = proc_data->max_bin_enst_flux[triad_class][types];
					gsl_histogram2d_set_ranges_uniform(proc_data->triads_wghtd_2d_pdf_t[triad_class][types], -M_PI, M_PI, 0.0, max_bin);
				}
				#endif
			}
		}
		#endif
	}


	// Loop through the sectors for k3
	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {

		// Initialize counters number of triads and enstrophy flux for each triad type
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			proc_data->num_triads[i][a]                             = 0;
			proc_data->enst_flux[i][a]                              = 0.0;
			proc_data->num_triads_1d[i][a]                          = 0;
			proc_data->enst_flux_1d[i][a]                           = 0.0;
			proc_data->phase_order_C_theta_triads[i][a]             = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_1d[i][a]          = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_unidirec[i][a]    = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_unidirec_1d[i][a] = 0.0 + 0.0 * I;
		}

		// Set the in time histogram bins to 0
		if (pre_compute_flag != PRE_COMPUTE) {
			#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
			for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
				for (int types = 0; types < NUM_TRIAD_TYPES; ++types) {
					gsl_histogram_reset(proc_data->triads_sect_all_pdf_t[triad_class][types][a]);
					gsl_histogram_reset(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][types][a]);
					gsl_histogram_reset(proc_data->triads_sect_1d_pdf_t[triad_class][types][a]);
					gsl_histogram_reset(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][types][a]);
				}
			}
			#endif
		}

		// Loop through the sectors for k1
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {

			// Initialize counters for number of triads and enstrophy flux across sectors for each triad type
			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
				proc_data->num_triads_2d[i][a][l]                          = 0;
				proc_data->enst_flux_2d[i][a][l]                           = 0.0;
				proc_data->phase_order_C_theta_triads_2d[i][a][l]          = 0.0 + 0.0 * I;
				proc_data->phase_order_C_theta_triads_unidirec_2d[i][a][l] = 0.0 + 0.0 * I;
				for (int type = 0; type < 2; ++type) {
					proc_data->phase_order_norm_const[type][i][a][l] = 0.0;
				}
			}

			// Set the in time histogram bins to 0
			if (pre_compute_flag != PRE_COMPUTE) {
				#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
				for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
					for (int types = 0; types < NUM_TRIAD_TYPES; ++types) {
						gsl_histogram_reset(proc_data->triads_sect_2d_pdf_t[triad_class][types][a][l]);
						gsl_histogram_reset(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][types][a][l]);
					}
				}
				#endif
			}
			
			// Loop through wavevectors
			if (proc_data->num_wave_vecs[a][l] != 0) {
				for (int n = 0; n < proc_data->num_wave_vecs[a][l]; ++n) {
					
					// Get k1 and k2 and k3
					k1_x = proc_data->phase_sync_wave_vecs[a][l][K1_X][n];
					k1_y = proc_data->phase_sync_wave_vecs[a][l][K1_Y][n];
					k2_x = proc_data->phase_sync_wave_vecs[a][l][K2_X][n];
					k2_y = proc_data->phase_sync_wave_vecs[a][l][K2_Y][n];
					k3_x = proc_data->phase_sync_wave_vecs[a][l][K3_X][n];
					k3_y = proc_data->phase_sync_wave_vecs[a][l][K3_Y][n];

					// Get the mod square of the wavevectors
					k1_sqr = (double) (k1_x * k1_x + k1_y * k1_y); // proc_data->phase_sync_wave_vecs[a][l][K1_SQR][n];
					k2_sqr = (double) (k2_x * k2_x + k2_y * k2_y); // proc_data->phase_sync_wave_vecs[a][l][K2_SQR][n];

					// Get correct phase index -> recall that to access kx > 0, use -kx
					tmp_k1 = (sys_vars->kmax - k1_x) * (2 * sys_vars->kmax + 1);	
					tmp_k2 = (sys_vars->kmax - k2_x) * (2 * sys_vars->kmax + 1);
					tmp_k3 = (sys_vars->kmax - k3_x) * (2 * sys_vars->kmax + 1);
					
					// Compute the flux pre factor
					flux_pre_fac = (double) (k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

					// Get the flux weight term
					flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax + k2_y] * proc_data->amps[tmp_k3 + sys_vars->kmax + k3_y]);

					// Get the triad phase
					triad_phase  = proc_data->phases[tmp_k1 + sys_vars->kmax + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax + k2_y] - proc_data->phases[tmp_k3 + sys_vars->kmax + k3_y];
					phase_val[0] = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;

					// Get flux term
					flux_term = flux_wght * cos(triad_phase);

					if (pre_compute_flag != PRE_COMPUTE) {						
						///////////////////////////////////////////
						///	 Positive Flux term
						///////////////////////////////////////////
						if (proc_data->phase_sync_wave_vecs[a][l][FLUX_TERM][n] == POS_FLUX_TERM) {

							// Define the generalized triad phase for the first term in the flux
							gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
							phase_val[1]    = gen_triad_phase;
							
							// Get the collective phase term
							collective_phase_term = fabs(flux_wght) * cexp(I * gen_triad_phase);

							//------------------------------------------ TRIAD TYPE 0
							// Update the combined triad phase order parameter with the appropriate contribution
							proc_data->num_triads[0][a]++;
							proc_data->enst_flux[0][a]                  += flux_term;
							proc_data->triad_phase_order[0][a]          += cexp(I * gen_triad_phase);
							
							// Update collective phase order parameter for C_theta
							proc_data->phase_order_C_theta_triads[0][a] += collective_phase_term;						
							// Update the unidirectional collective phase order parameter for C_theta
							if (k3_y >= 0) {
								proc_data->phase_order_C_theta_triads_unidirec[0][a] += collective_phase_term;
							}						
							
							// Update the triad phase order data for the 1d contribution to the flux
							if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
								// 1D contribution only depends on theta_k3
								proc_data->num_triads_1d[0][a]++;
								proc_data->enst_flux_1d[0][a]         += flux_term;
								proc_data->triad_phase_order_1d[0][a] += cexp(I * gen_triad_phase);

								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads_1d[0][a] += collective_phase_term;
								proc_data->phase_order_norm_const[0][0][a][a] += fabs(flux_wght);
								// Update unidirectional collective phase order parameter for C_theta
								if (k3_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec_1d[0][a] += collective_phase_term;
									proc_data->phase_order_norm_const[1][0][a][a] += fabs(flux_wght);
								}							
							}

							// Update the triad phase order data for the 2d contribution to the flux
							if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
								// Update the flux contribution for type 0
								proc_data->num_triads_2d[0][a][l]++;
								proc_data->enst_flux_2d[0][a][l]         += flux_term;
								proc_data->triad_phase_order_2d[0][a][l] += cexp(I * gen_triad_phase);
	 
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads_2d[0][a][l] += collective_phase_term;
								proc_data->phase_order_norm_const[0][0][a][l] += fabs(flux_wght);
								// Update unidirectional collective phase order parameter for C_theta
								if (k3_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec_2d[0][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[1][0][a][l] += fabs(flux_wght);
								}							
							}

							//------ Update the PDFs of the triad phases and weighted flux densisties
							#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
							for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
								#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
								// Updated 2d histogram
								gsl_status = gsl_histogram2d_increment(proc_data->triads_wghtd_2d_pdf_t[triad_class][0], phase_val[triad_class], fabs(flux_wght));
								#endif
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
								// Update the PDFs for all possible triads
								gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][0], phase_val[triad_class]);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][0], phase_val[triad_class], fabs(flux_wght));
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								#endif
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
								// Update the PDFs for all: both 1d and 2d contributions
								gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][0][a], phase_val[triad_class]);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][0][a], phase_val[triad_class], fabs(flux_wght));
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								#endif
								if (a == l) {
					        	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
									// Update the PDFs for 1d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][0][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][0][a], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
								}
								else {
					        	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
									// Update the PDFs for 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][0][a][l], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][0][a][l], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
								}
							}
							#endif

							if (flux_pre_fac < 0) {
								
								//------------------------------------------ TRIAD TYPE 1
								proc_data->num_triads[1][a]++;		
								proc_data->enst_flux[1][a]         += flux_term;
								proc_data->triad_phase_order[1][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[1][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k3_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[1][a] += collective_phase_term;
								}							
								
								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contribution only depends on theta_k3
									proc_data->num_triads_1d[1][a]++;		
									proc_data->enst_flux_1d[1][a]         += flux_term;
									proc_data->triad_phase_order_1d[1][a] += cexp(I * gen_triad_phase);
									
									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[1][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][1][a][a] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[1][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][1][a][a] += fabs(flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 1
									proc_data->num_triads_2d[1][a][l]++;
									proc_data->enst_flux_2d[1][a][l]         += flux_term;
									proc_data->triad_phase_order_2d[1][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[1][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][1][a][l] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[1][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][1][a][l] += fabs(flux_wght);
									}								
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
									// Updated 2d histogram
									gsl_status = gsl_histogram2d_increment(proc_data->triads_wghtd_2d_pdf_t[triad_class][1], phase_val[triad_class], fabs(flux_wght));
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][1], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][1], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][1][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][1][a], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][1][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][1][a], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][1][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][1][a][l], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
							else if (flux_pre_fac > 0) {
								
								//------------------------------------------ TRIAD TYPE 2
								proc_data->num_triads[2][a]++;		
								proc_data->enst_flux[2][a]         += flux_term;
								proc_data->triad_phase_order[2][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[2][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k3_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[2][a] += collective_phase_term;
								}							
								
								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contribution only depends on theta_k3
									proc_data->num_triads_1d[2][a]++;		
									proc_data->enst_flux_1d[2][a]         += flux_term;
									proc_data->triad_phase_order_1d[2][a] += cexp(I * gen_triad_phase);
									
									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[2][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][2][a][a] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[2][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][2][a][a] += fabs(flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 2
									proc_data->num_triads_2d[2][a][l]++;
									proc_data->enst_flux_2d[2][a][l]         += flux_term;
									proc_data->triad_phase_order_2d[2][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[2][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][2][a][l] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[2][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][2][a][l] += fabs(flux_wght);
									}								
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
									// Updated 2d histogram
									gsl_status = gsl_histogram2d_increment(proc_data->triads_wghtd_2d_pdf_t[triad_class][2], phase_val[triad_class], fabs(flux_wght));
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][2], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][2], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][2][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][2][a], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][2][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][2][a], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][2][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][2][a][l], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
							else if (flux_pre_fac == 0.0 || flux_wght == 0.0) {

								//------------------------------------------ TRIAD TYPE 5 - 0 flux contribution
								proc_data->num_triads[5][a]++;		
								proc_data->enst_flux[5][a]         += flux_term;
								proc_data->triad_phase_order[5][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[5][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k3_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[5][a] += collective_phase_term;
								}							

								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contributions
									proc_data->num_triads_1d[5][a]++;		
									proc_data->enst_flux_1d[5][a]         += flux_term;
									proc_data->triad_phase_order_1d[5][a] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[5][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][5][a][a] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[5][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][5][a][a] += fabs(flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 5
									proc_data->num_triads_2d[5][a][l]++;
									proc_data->enst_flux_2d[5][a][l]         += flux_term;
									proc_data->triad_phase_order_2d[5][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[5][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][5][a][l] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[5][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][5][a][l] += fabs(flux_wght);
									}								
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][5], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][5], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][5][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][5][a], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][5][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][5][a], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][5][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][5][a][l], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
							else {

								// Define the generalized triad phase for the ignored terms -> carg of flux_weight is not defined so just triad phases
								gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
								phase_val[1]    = gen_triad_phase;

								// Get the collective phase term
								collective_phase_term = fabs(flux_wght) * cexp(I * gen_triad_phase);

								//------------------------------------------ TRIAD TYPE 6
								proc_data->num_triads[6][a]++;		
								proc_data->enst_flux[6][a]         += flux_term;
								proc_data->triad_phase_order[6][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[6][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k3_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[6][a] += collective_phase_term;
								}							

								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contributions
									proc_data->num_triads_1d[6][a]++;		
									proc_data->enst_flux_1d[6][a]         += flux_term;
									proc_data->triad_phase_order_1d[6][a] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[6][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][6][a][a] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[6][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][6][a][a] += fabs(flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 6
									proc_data->num_triads_2d[6][a][l]++;
									proc_data->enst_flux_2d[6][a][l]         += flux_term;
									proc_data->triad_phase_order_2d[6][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[6][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][6][a][l] += fabs(flux_wght);
									// Update unidirectional collective phase order parameter for C_theta
									if (k3_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[6][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][6][a][l] += fabs(flux_wght);
									}								
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][6], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][6], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][6][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][6][a], phase_val[triad_class], fabs(flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][6][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][6][a], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][6][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][6][a][l], phase_val[triad_class], fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
						}
						///////////////////////////////////////////
						///	 Negative Flux term
						///////////////////////////////////////////
						else if (proc_data->phase_sync_wave_vecs[a][l][FLUX_TERM][n] == NEG_FLUX_TERM) {
							
							// Define the generalized triad phase for the first term in the flux
							gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
							phase_val[1]    = gen_triad_phase;

							// Get the collective phase term
							collective_phase_term = fabs(-flux_wght) * cexp(I * gen_triad_phase);
							
							//------------------------------------------ TRIAD TYPE 0 - All triad types combined
							// Update the combined triad phase order parameter with the appropriate contribution
							proc_data->num_triads[0][a]++;
							proc_data->enst_flux[0][a]         += -flux_term;
							proc_data->triad_phase_order[0][a] += cexp(I * gen_triad_phase);
							
							// Update collective phase order parameter for C_theta
							proc_data->phase_order_C_theta_triads[0][a] += collective_phase_term;
							// Update the unidirectional collective phase order parameter for C_theta
							if (k1_y >= 0 && k2_y >= 0) {
								proc_data->phase_order_C_theta_triads_unidirec[0][a] += collective_phase_term;
							}						

							// Update the triad phase order data for the 1d contribution to the flux
							if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
								// 1D contributions
								proc_data->num_triads_1d[0][a]++;
								proc_data->enst_flux_1d[0][a]         += -flux_term;
								proc_data->triad_phase_order_1d[0][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads_1d[0][a] += collective_phase_term;
								proc_data->phase_order_norm_const[0][0][a][a] += fabs(-flux_wght);
								// Update the unidirectional collective phase order parameter for C_theta
								if (k1_y >= 0 && k2_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec_1d[0][a] += collective_phase_term;
									proc_data->phase_order_norm_const[1][0][a][a] += fabs(-flux_wght);
								}							
							}

							// Update the triad phase order data for the 2d contribution to the flux
							if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
								// Update the flux contribution for type 0
								proc_data->num_triads_2d[0][a][l]++;
								proc_data->enst_flux_2d[0][a][l]         += -flux_term;
								proc_data->triad_phase_order_2d[0][a][l] += cexp(I * gen_triad_phase);

								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads_2d[0][a][l] += collective_phase_term;
								proc_data->phase_order_norm_const[0][0][a][l] += fabs(-flux_wght);
								// Update the unidirectional collective phase order parameter for C_theta
								if (k1_y >= 0 && k2_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec_2d[0][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[1][0][a][l] += fabs(-flux_wght);
								}							
							}

							//------ Update the PDFs of the triad phases and weighted flux densisties
							#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
							for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
								#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
								// Updated 2d histogram
								gsl_status = gsl_histogram2d_increment(proc_data->triads_wghtd_2d_pdf_t[triad_class][0], phase_val[triad_class], fabs(-flux_wght));
								#endif
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
								// Update the PDFs for all possible triads
								gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][0], phase_val[triad_class]);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][0], phase_val[triad_class], fabs(-flux_wght));
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								#endif
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
								// Update the PDFs for all: both 1d and 2d contributions
								gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][0][a], phase_val[triad_class]);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][0][a], phase_val[triad_class], fabs(-flux_wght));
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
									exit(1);
								}
								#endif
								if (a == l) {
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
									// Update the PDFs for 1d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][0][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][0][a], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
								}
								else {
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
									// Update the PDFs for 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][0][a][l], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][0][a][l], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
								}
							}
							#endif

							if (flux_pre_fac < 0) {
								//------------------------------------------ TRIAD TYPE 3
								proc_data->num_triads[3][a]++;		
								proc_data->enst_flux[3][a]         += -flux_term;
								proc_data->triad_phase_order[3][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[3][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k1_y >= 0 && k2_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[3][a] += collective_phase_term;
								}							


								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D Contirbutions
									proc_data->num_triads_1d[3][a]++;		
									proc_data->enst_flux_1d[3][a]         += -flux_term;
									proc_data->triad_phase_order_1d[3][a] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[3][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][3][a][a] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[3][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][3][a][a] += fabs(-flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 3
									proc_data->num_triads_2d[3][a][l]++;
									proc_data->enst_flux_2d[3][a][l]         += -flux_term;
									proc_data->triad_phase_order_2d[3][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[3][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][3][a][l] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[3][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][3][a][l] += fabs(-flux_wght);
									}																
								}


								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
									// Updated 2d histogram
									gsl_status = gsl_histogram2d_increment(proc_data->triads_wghtd_2d_pdf_t[triad_class][3], phase_val[triad_class], fabs(-flux_wght));
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][3], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][3], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][3][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][3][a], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][3][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][3][a], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][3][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][3][a][l], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
							else if (flux_pre_fac > 0) {
								
								//------------------------------------------ TRIAD TYPE 4
								proc_data->num_triads[4][a]++;		
								proc_data->enst_flux[4][a]         += -flux_term;
								proc_data->triad_phase_order[4][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[4][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k1_y >= 0 && k2_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[4][a] += collective_phase_term;
								}							

								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contributions
									proc_data->num_triads_1d[4][a]++;		
									proc_data->enst_flux_1d[4][a]         += -flux_term;
									proc_data->triad_phase_order_1d[4][a] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[4][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][4][a][a] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[4][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][4][a][a] += fabs(-flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 4
									proc_data->num_triads_2d[4][a][l]++;
									proc_data->enst_flux_2d[4][a][l]         += -flux_term;
									proc_data->triad_phase_order_2d[4][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[4][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][4][a][l] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[4][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][4][a][l] += fabs(-flux_wght);
									}										
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
									// Updated 2d histogram
									gsl_status = gsl_histogram2d_increment(proc_data->triads_wghtd_2d_pdf_t[triad_class][4], phase_val[triad_class], fabs(-flux_wght));
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][4], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][4], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][4][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][4][a], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][4][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][4][a], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][4][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][4][a][l], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
							else if (flux_pre_fac == 0.0 || flux_wght == 0.0) {

								//------------------------------------------ TRIAD TYPE 5
								proc_data->num_triads[5][a]++;		
								proc_data->enst_flux[5][a]         += -flux_term;
								proc_data->triad_phase_order[5][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[5][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k1_y >= 0 && k2_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[5][a] += collective_phase_term;
								}							

								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contributions
									proc_data->num_triads_1d[5][a]++;		
									proc_data->enst_flux_1d[5][a]         += -flux_term;
									proc_data->triad_phase_order_1d[5][a] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[5][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][5][a][a] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[5][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][5][a][a] += fabs(-flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 5
									proc_data->num_triads_2d[5][a][l]++;
									proc_data->enst_flux_2d[5][a][l]         += -flux_term;
									proc_data->triad_phase_order_2d[5][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[5][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][5][a][l] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[5][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][5][a][l] += fabs(-flux_wght);
									}								
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][5], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][5], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][5][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][5][a], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][5][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][5][a], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][5][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][5][a][l], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}											
										#endif						
									}
								}
								#endif
							}
							else {
								// Define the generalized triad phase for the ignored terms -> carg of flux weight is not defined here so just triad phase
								gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
								phase_val[1] = gen_triad_phase;

								// Get the collective phase term
								collective_phase_term = fabs(-flux_wght) * cexp(I * gen_triad_phase);

								//------------------------------------------ TRIAD TYPE 6
								proc_data->num_triads[6][a]++;		
								proc_data->enst_flux[6][a]         += -flux_term;
								proc_data->triad_phase_order[6][a] += cexp(I * gen_triad_phase);
								
								// Update collective phase order parameter for C_theta
								proc_data->phase_order_C_theta_triads[6][a] += collective_phase_term;
								// Update the unidirectional collective phase order parameter for C_theta
								if (k1_y >= 0 && k2_y >= 0) {
									proc_data->phase_order_C_theta_triads_unidirec[6][a] += collective_phase_term;
								}							

								// Update the triad phase order data for the 1d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_1D) {
									// 1D contributions
									proc_data->num_triads_1d[6][a]++;		
									proc_data->enst_flux_1d[6][a]         += -flux_term;
									proc_data->triad_phase_order_1d[6][a] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_1d[6][a] += collective_phase_term;
									proc_data->phase_order_norm_const[0][6][a][a] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_1d[6][a] += collective_phase_term;
										proc_data->phase_order_norm_const[1][6][a][a] += fabs(-flux_wght);
									}								
								}

								// Update the triad phase order data for the 2d contribution to the flux
								if (proc_data->phase_sync_wave_vecs[a][l][CONTRIB_TYPE][n] == CONTRIB_2D) {
									// Update the flux contribution for type 6
									proc_data->num_triads_2d[6][a][l]++;
									proc_data->enst_flux_2d[6][a][l]         += -flux_term;
									proc_data->triad_phase_order_2d[6][a][l] += cexp(I * gen_triad_phase);

									// Update collective phase order parameter for C_theta
									proc_data->phase_order_C_theta_triads_2d[6][a][l] += collective_phase_term;
									proc_data->phase_order_norm_const[0][6][a][l] += fabs(-flux_wght);
									// Update the unidirectional collective phase order parameter for C_theta
									if (k1_y >= 0 && k2_y >= 0) {
										proc_data->phase_order_C_theta_triads_unidirec_2d[6][a][l] += collective_phase_term;
										proc_data->phase_order_norm_const[1][6][a][l] += fabs(-flux_wght);
									}								
								}

								//------ Update the PDFs of the triad phases and weighted flux densisties
								#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
								for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
									// Update the PDFs for all possible triads
									gsl_status = gsl_histogram_increment(proc_data->triads_all_pdf_t[triad_class][6], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_wghtd_all_pdf_t[triad_class][6], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "All Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL)
									// Update the PDFs for all: both 1d and 2d contributions
									gsl_status = gsl_histogram_increment(proc_data->triads_sect_all_pdf_t[triad_class][6][a], phase_val[triad_class]);
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][6][a], phase_val[triad_class], fabs(-flux_wght));
									if (gsl_status != 0) {
										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
										exit(1);
									}
									#endif
									if (a == l) {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
										// Update the PDFs for 1d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_1d_pdf_t[triad_class][6][a], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][6][a], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
									else {
										#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
										// Update the PDFs for 2d contributions
										gsl_status = gsl_histogram_increment(proc_data->triads_sect_2d_pdf_t[triad_class][6][a][l], phase_val[triad_class]);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][6][a][l], phase_val[triad_class], fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, phase_val[triad_class]);
											exit(1);
										}
										#endif
									}
								}
								#endif
							}
						}
					}
					else if (pre_compute_flag == PRE_COMPUTE) {

						//---------------- In Pre Compute Mode
						if (proc_data->phase_sync_wave_vecs[a][l][FLUX_TERM][n] == POS_FLUX_TERM) { 
							for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
								proc_data->max_enst_flux[triad_class][0][s] = fmax(proc_data->max_enst_flux[triad_class][0][s], fabs(flux_wght));
								proc_data->max_bin_enst_flux[triad_class][0] = fmax(proc_data->max_bin_enst_flux[triad_class][0], fabs(flux_wght));
								if (flux_pre_fac < 0) {
									proc_data->max_enst_flux[triad_class][1][s] = fmax(proc_data->max_enst_flux[triad_class][1][s], fabs(flux_wght));
									proc_data->max_bin_enst_flux[triad_class][1] = fmax(proc_data->max_bin_enst_flux[triad_class][1], fabs(flux_wght));
								}
								else if (flux_pre_fac > 0) { 
									proc_data->max_enst_flux[triad_class][2][s] = fmax(proc_data->max_enst_flux[triad_class][2][s], fabs(flux_wght));
									proc_data->max_bin_enst_flux[triad_class][2] = fmax(proc_data->max_bin_enst_flux[triad_class][2], fabs(flux_wght));
								}
							}
						}
						else if (proc_data->phase_sync_wave_vecs[a][l][FLUX_TERM][n] == NEG_FLUX_TERM) {
							for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
								proc_data->max_enst_flux[triad_class][0][s] = fmax(proc_data->max_enst_flux[triad_class][0][s], fabs(-flux_wght));
								proc_data->max_bin_enst_flux[triad_class][0] = fmax(proc_data->max_bin_enst_flux[triad_class][0], fabs(-flux_wght));
								if (flux_pre_fac < 0) {
									proc_data->max_enst_flux[triad_class][3][s] = fmax(proc_data->max_enst_flux[triad_class][3][s], fabs(-flux_wght));
									proc_data->max_bin_enst_flux[triad_class][3] = fmax(proc_data->max_bin_enst_flux[triad_class][3], fabs(-flux_wght));
								}
								else if (flux_pre_fac > 0) { 
									proc_data->max_enst_flux[triad_class][4][s] = fmax(proc_data->max_enst_flux[triad_class][4][s], fabs(-flux_wght));
									proc_data->max_bin_enst_flux[triad_class][4] = fmax(proc_data->max_bin_enst_flux[triad_class][4], fabs(-flux_wght));
								}
							}
						}
					}
				}
			}
		}
	}

	if (pre_compute_flag != PRE_COMPUTE) {
		//------------------- Record the data for the triads
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				// Normalize the phase order parameters
				if (proc_data->num_triads[i][a] != 0) {
					proc_data->triad_phase_order[i][a] /= proc_data->num_triads[i][a];
				}
				if (proc_data->num_triads_1d[i][a] != 0) {
					proc_data->triad_phase_order_1d[i][a] /= proc_data->num_triads_1d[i][a];
				}
				// // Normalize the 1d collective phase order parameters 
				// if (proc_data->phase_order_norm_const[0][i][a][a] > 0.0) {
				// 	proc_data->phase_order_C_theta_triads_1d[i][a] /= proc_data->phase_order_norm_const[0][i][a][a];
				// }
				// if (proc_data->phase_order_norm_const[1][i][a][a] > 0.0) {
				// 	proc_data->phase_order_C_theta_triads_unidirec_1d[i][a] /= proc_data->phase_order_norm_const[1][i][a][a];
				// }
				// // Initialize total norm constant over k1 sectors
				// double tot_norm_const_1 = 0.0;
				// double tot_norm_const_2 = 0.0;
				
				// Record the phase syncs and average phases for combined and 1d contributions
				proc_data->triad_R[i][a]      = cabs(proc_data->triad_phase_order[i][a]);
				proc_data->triad_Phi[i][a]    = carg(proc_data->triad_phase_order[i][a]);
				proc_data->triad_R_1d[i][a]   = cabs(proc_data->triad_phase_order_1d[i][a]);
				proc_data->triad_Phi_1d[i][a] = carg(proc_data->triad_phase_order_1d[i][a]);
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					// Normalize the 2d phase order parameter
					if (proc_data->num_triads_2d[i][a][l] != 0) {
						proc_data->triad_phase_order_2d[i][a][l] /= proc_data->num_triads_2d[i][a][l];
					}
					// // Normalize the collective phase order parameters
					// if (proc_data->phase_order_norm_const[0][i][a][l] > 0.0) {
					// 	proc_data->phase_order_C_theta_triads_2d[i][a][l] /= proc_data->phase_order_norm_const[0][i][a][l];
					// 	tot_norm_const_1 += proc_data->phase_order_norm_const[0][i][a][l];
					// }
					// if (proc_data->phase_order_norm_const[1][i][a][l] > 0.0) {
					// 	proc_data->phase_order_C_theta_triads_unidirec_2d[i][a][l] /= proc_data->phase_order_norm_const[1][i][a][l];
					// 	tot_norm_const_2 += proc_data->phase_order_norm_const[1][i][a][l];
					// }
					
					// Record the phase syncs and average phases for 2D contributions
					proc_data->triad_R_2d[i][a][l]   = cabs(proc_data->triad_phase_order_2d[i][a][l]);
					proc_data->triad_Phi_2d[i][a][l] = carg(proc_data->triad_phase_order_2d[i][a][l]);

					#if defined(__SEC_PHASE_SYNC_STATS)
					if (i == 0) {
						// Update the histograms
						gsl_status = gsl_histogram_increment(proc_data->triad_R_2d_pdf[a][l], proc_data->triad_R_2d[0][a][l]);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Phase Sync 2D PDF", a, s, gsl_status, proc_data->triad_R_2d[0][a][l]);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_Phi_2d_pdf[a][l], proc_data->triad_Phi_2d[0][a][l]);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Average Phase 2D PDF", a, s, gsl_status, proc_data->triad_Phi_2d[0][a][l]);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->enst_flux_2d_pdf[a][l], proc_data->enst_flux_2d[0][a][l]);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux PDF", a, s, gsl_status, proc_data->enst_flux_2d[0][a][l]);
							exit(1);
						}

						// Update the running stats
						gsl_rstat_add(proc_data->triad_R_2d[0][a][l], proc_data->triad_R_2d_stats[a][l]);
						gsl_rstat_add(proc_data->triad_Phi_2d[0][a][l], proc_data->triad_Phi_2d_stats[a][l]);
						gsl_rstat_add(proc_data->enst_flux_2d[0][a][l], proc_data->enst_flux_2d_stats[a][l]);
					}
					#endif
				}
				// // Normalize the combined collective phase order parameters
				// if (tot_norm_const_1 > 0.0) {
				// 	proc_data->phase_order_C_theta_triads[i][a] /= tot_norm_const_1;
				// }
				// if (tot_norm_const_2 > 0.0) {
				// 	proc_data->phase_order_C_theta_triads_unidirec[i][a] /= tot_norm_const_2;
				// }
							
				#if defined(__SEC_PHASE_SYNC_COND_STATS)
				// Update the maximum Sync value
				proc_data->max_sync[i][a] = fmax(cabs(proc_data->phase_order_C_theta_triads[i][a]), proc_data->max_sync[i][a]);
				#endif
			}
		}

		//------------- Reset order parameters for next iteration
		for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
			for (int type = 0; type < NUM_TRIAD_TYPES; ++type) {
				proc_data->triad_phase_order[type][a]    = 0.0 + 0.0 * I;
				proc_data->triad_phase_order_1d[type][a] = 0.0 + 0.0 * I;
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					proc_data->triad_phase_order_2d[type][a][l] = 0.0 + 0.0 * I;
				}
			}
		}
	}
}

void ComputePhaseSyncConditionalStats(void) {

	// Initialize Variables
	int indx, tmp;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double max_norm;
	int r;
	double R_threshold = 0.5;
	int x_incr, y_incr;
	int cond_type;
	int gsl_status;
	double vort_long_increment, vort_trans_increment;
	double std_w;
	herr_t status;
	hid_t dset;
	char coll_phase_group_string[64];
	char vort_group_string[64];
	int increments[NUM_INCR] = {1, 2, 4, 8, Nx/2};
	double R_thresh[NUM_THRESH_TYPES] = {0.55, 0.65, 0.75, 0.85, 0.95};
	double flux_min = 1e10; 
	double flux_max = -1e10;
	double sync_min = 0.0;
	double sync_max = 0.0;
	double tmp_sync, tmp_flux;

	// --------------------------------	
	//	Initialize Stats Objects
	// --------------------------------
	// Initialize increment stats
	for (int i = 0; i < NUM_COND_TYPES; ++i) {
		for (int j = 0; j < INCR_TYPES; ++j) {
			for (int k = 0; k < NUM_INCR; ++k) {
				for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
					proc_data->cond_t_w_incr_hist[i][j][k][l]  = gsl_histogram_alloc(N_BINS);
					proc_data->cond_t_w_incr_stats[i][j][k][l] = gsl_rstat_alloc();
				}
			}
		}
	}

	// Initialize joint histogram objects
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		proc_data->joint_sync_enst_flux_hist[i] = gsl_histogram2d_alloc(N_BINS_X_JOINT, N_BINS_Y_JOINT);
		if (proc_data->joint_sync_enst_flux_hist[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Joint PDF Sync and Enstrophy Flux");
			exit(1);
		}
	}

	// Initialize temp memory
	fftw_complex* tmp_collec_phase = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors * (NUM_TRIAD_TYPES + 1));
	double* tmp_max_norm = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);

	/////////////////////////////////////////////////////	
	//	Loop Through Data to PreCompute Stats
	/////////////////////////////////////////////////////
	printf("\n\nConditional Sync Stats Processing:\n");
	for (int s = 0; s < sys_vars->num_snaps; ++s) { 
		
		// --------------------------------	
		//	Open Files
		// --------------------------------
		// Open output file with default I/O access properties -- Output file stores the Phase Sync Data
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, s);
			exit(1);
		}

		// Open input file with default I/O access properties -- input file stores the vorticity field Data
		file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->input_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, s);
			exit(1);
		}

		// --------------------------------	
		//	Read in Data
		// --------------------------------
		///------------------------------------ Read in the Collective Phase Data
		// Initialize Group Name
		sprintf(coll_phase_group_string, "/Snap_%05d/CollectivePhaseOrder_C_theta_Triads", s);
		if (H5Lexists(file_info->output_file_handle, coll_phase_group_string, H5P_DEFAULT) > 0 ) {
			dset = H5Dopen (file_info->output_file_handle, coll_phase_group_string, H5P_DEFAULT);
			if (dset < 0 ) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Collective Phase Order", s);
				exit(1);		
			} 
			// Read in Fourier space vorticity
			if(H5LTread_dataset(file_info->output_file_handle, coll_phase_group_string, file_info->COMPLEX_DTYPE, tmp_collec_phase) < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Collective Phase Order", s);
				exit(1);	
			}
		}
		else {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to find ["CYAN"%s"RESET"] in file ["CYAN"%s"RESET"]. Please check input file\n-->> Exiting...\n", "Collective Phase Order", file_info->output_file_name);
			exit(1);
		}

		///------------------------------------ Read in the Vorticity Data
		/// Initialize Group Name
		sprintf(vort_group_string, "/Iter_%05d/w_hat", s);	
		if (H5Lexists(file_info->input_file_handle, vort_group_string, H5P_DEFAULT) > 0 ) {
			dset = H5Dopen (file_info->input_file_handle, vort_group_string, H5P_DEFAULT);
			if (dset < 0 ) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", s);
				exit(1);		
			} 
			// Read in Fourier space vorticity
			if(H5LTread_dataset(file_info->input_file_handle, vort_group_string, file_info->COMPLEX_DTYPE, run_data->w_hat) < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", s);
				exit(1);	
			}
		}
		else {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to find ["CYAN"%s"RESET"] in file ["CYAN"%s"RESET"]. Please check input file\n-->> Exiting...\n", "w_hat", file_info->input_file_name);
			exit(1);
		}

		// Get the real space vorticity from the Fourier space
		for (int i = 0; i < Nx; ++i) {	
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Save the fourier vorticity before transform as it will be overwrittend
				run_data->tmp_w_hat[indx] = run_data->w_hat[indx];
			}
		}
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->tmp_w_hat, run_data->w);
		for (int i = 0; i < Nx; ++i) {	
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize the vorticity
				run_data->w[indx] *= 1.0; // / (Nx * Ny);
			}
		}

		// --------------------------------	
		//	Compute Sync Indicator
		// --------------------------------
		max_norm = 0.0;
		for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
			// Read in collective phase to appropraite array
			proc_data->phase_order_C_theta_triads[0][a] = tmp_collec_phase[0 * sys_vars->num_k3_sectors + a];

			// Compute normalized collective phase
			proc_data->phase_order_C_theta_triads[0][a] /= proc_data->max_sync[0][a];

			// Update max and mins
			max_norm = fmax(cabs(proc_data->phase_order_C_theta_triads[0][a]), max_norm);
			sync_max = fmax(cabs(proc_data->phase_order_C_theta_triads[0][a]), sync_max);
			sync_min = 0.0;
			flux_max = fmax(creal(proc_data->phase_order_C_theta_triads[0][a]), flux_max);
			flux_min = fmin(creal(proc_data->phase_order_C_theta_triads[0][a]), flux_min);
		}
		// printf("Snap: %d - Cp: %1.16lf %1.21lf\tNormed: %1.16lf\n", s, creal(proc_data->phase_order_C_theta_triads[0][0]), cimag(proc_data->phase_order_C_theta_triads[0][0]), max_norm);

		// // --------------------------------	
		// //	Compute Conditional Stats
		// // --------------------------------
		// // Get the conditional type: high sync event or low sync event using the max norm
		// if (max_norm <= R_threshold) {
		// 	cond_type = 0;
		// }
		// else {
		// 	cond_type = 1;
		// }

		// Compute the increment histogram
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Compute velocity increments and update histograms
				for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
					// Get the current increment
					r = increments[r_indx];

					// Increment in the x direction 
					x_incr = i + r;
					if (x_incr < Nx) {
						vort_long_increment  = run_data->w[x_incr * Ny + j] - run_data->w[i * Ny + j];
						for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
							// Cheeck the threshold criteria
							if (max_norm <= R_thresh[l]) {
								gsl_rstat_add(vort_long_increment, proc_data->cond_t_w_incr_stats[0][0][r_indx][l]);
								cond_type = 0;
							}
							else {
								gsl_rstat_add(vort_long_increment, proc_data->cond_t_w_incr_stats[1][0][r_indx][l]);
								cond_type = 1;
							}

							// Update the appropriate 	
							gsl_rstat_add(vort_long_increment, proc_data->cond_t_w_incr_stats[2][0][r_indx][l]);
						}
					}

					// Increment in the y direction
					y_incr = j + r; 
					if (y_incr < Ny) {
						vort_trans_increment = run_data->w[i * Ny + y_incr] - run_data->w[i * Ny + j];
						for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
							// Cheeck the threshold criteria
							if (max_norm <= R_thresh[l]) {
								gsl_rstat_add(vort_trans_increment, proc_data->cond_t_w_incr_stats[0][1][r_indx][l]);	
								cond_type = 0;
							}
							else {
								gsl_rstat_add(vort_trans_increment, proc_data->cond_t_w_incr_stats[1][1][r_indx][l]);	
								cond_type = 1;
							}

							// Update the appropriate
							gsl_rstat_add(vort_trans_increment, proc_data->cond_t_w_incr_stats[2][1][r_indx][l]);
						}
					}
				}
			}
		}


		// --------------------------------	
		//	Close File
		// --------------------------------
		status = H5Dclose(dset);
		status = H5Fclose(file_info->output_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, s);
			exit(1);	
		}
	}

	// --------------------------------	
	//	Initialize Histogram Objects
	// --------------------------------
	// Set the bin limits for the vorticity increments
	for (int cond_t = 0; cond_t < NUM_COND_TYPES; ++cond_t) {
		for (int i = 0; i < INCR_TYPES; ++i) {
			for (int j = 0; j < NUM_INCR; ++j) {
				for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
					// Get the std of the incrments
					std_w = gsl_rstat_sd(proc_data->cond_t_w_incr_stats[cond_t][i][j][l]);
					if (std_w == 0.0) {
						std_w = 1.0;
					}
					
					// Vorticity increments
					gsl_status = gsl_histogram_set_ranges_uniform(proc_data->cond_t_w_incr_hist[cond_t][i][j][l], -BIN_LIM * std_w, BIN_LIM * std_w);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Increments");
						exit(1);
					}
				}
			}
		}
	}

	// printf("sync: %lf %lf\t Enst: %lf %lf\n", sync_min, sync_max, flux_min, flux_max);

	// Set the bins for the joint histograms
	gsl_histogram2d_set_ranges_uniform(proc_data->joint_sync_enst_flux_hist[0], sync_min, sync_max, flux_min, flux_max);

	/////////////////////////////////////////////////////	
	//	Compute Conditional Stats
	/////////////////////////////////////////////////////
	for (int s = 0; s < sys_vars->num_snaps; ++s) { 
		
		// Start timer
		double loop_begin = omp_get_wtime();
		
		// --------------------------------	
		//	Open Files
		// --------------------------------
		// Open output file with default I/O access properties -- Output file stores the Phase Sync Data
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, s);
			exit(1);
		}

		// Open input file with default I/O access properties -- input file stores the vorticity field Data
		file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->input_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, s);
			exit(1);
		}

		// --------------------------------	
		//	Read in Data
		// --------------------------------
		///------------------------------------ Read in the Collective Phase Data
		// Initialize Group Name
		sprintf(coll_phase_group_string, "/Snap_%05d/CollectivePhaseOrder_C_theta_Triads", s);
		if (H5Lexists(file_info->output_file_handle, coll_phase_group_string, H5P_DEFAULT) > 0 ) {
			dset = H5Dopen (file_info->output_file_handle, coll_phase_group_string, H5P_DEFAULT);
			if (dset < 0 ) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Collective Phase Order", s);
				exit(1);		
			} 
			// Read in Fourier space vorticity
			if(H5LTread_dataset(file_info->output_file_handle, coll_phase_group_string, file_info->COMPLEX_DTYPE, tmp_collec_phase) < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Collective Phase Order", s);
				exit(1);	
			}
		}
		else {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to find ["CYAN"%s"RESET"] in file ["CYAN"%s"RESET"]. Please check input file\n-->> Exiting...\n", "Collective Phase Order", file_info->output_file_name);
			exit(1);
		}

		///------------------------------------ Read in the Vorticity Data
		/// Initialize Group Name
		sprintf(vort_group_string, "/Iter_%05d/w_hat", s);	
		if (H5Lexists(file_info->input_file_handle, vort_group_string, H5P_DEFAULT) > 0 ) {
			dset = H5Dopen (file_info->input_file_handle, vort_group_string, H5P_DEFAULT);
			if (dset < 0 ) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", s);
				exit(1);		
			} 
			// Read in Fourier space vorticity
			if(H5LTread_dataset(file_info->input_file_handle, vort_group_string, file_info->COMPLEX_DTYPE, run_data->w_hat) < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", s);
				exit(1);	
			}
		}
		else {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to find ["CYAN"%s"RESET"] in file ["CYAN"%s"RESET"]. Please check input file\n-->> Exiting...\n", "w_hat", file_info->input_file_name);
			exit(1);
		}

		// Get the real space vorticity from the Fourier space
		for (int i = 0; i < Nx; ++i) {	
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Save the fourier vorticity before transform as it will be overwrittend
				run_data->tmp_w_hat[indx] = run_data->w_hat[indx];
			}
		}
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->tmp_w_hat, run_data->w);
		for (int i = 0; i < Nx; ++i) {	
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize the vorticity
				run_data->w[indx] *= 1.0; // / (Nx * Ny);
			}
		}


		// --------------------------------	
		//	Compute Sync Indicator
		// --------------------------------
		max_norm = 0.0;
		for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
			// Read in collective phase to appropraite array
			proc_data->phase_order_C_theta_triads[0][a] = tmp_collec_phase[0 * sys_vars->num_k3_sectors + a];

			// Compute normalized collective phase
			proc_data->phase_order_C_theta_triads[0][a] /= proc_data->max_sync[0][a];

			// Update max norm
			max_norm = fmax(cabs(proc_data->phase_order_C_theta_triads[0][a]), max_norm);
		}
		// Record the max norm for this snapshot
		tmp_max_norm[s] = max_norm;


		// // --------------------------------	
		// //	Compute Conditional Stats
		// // --------------------------------
		// // Get the conditional type: high sync event or low sync event using the max norm
		// if (max_norm <= R_threshold) {
		// 	cond_type = 0;
		// }
		// else {
		// 	cond_type = 1;
		// }
		// printf("%lf-%lf-%d\n", max_norm, R_threshold, cond_type);

		// Compute the increment histogram
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Compute velocity increments and update histograms
				for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
					// Get the current increment
					r = increments[r_indx];


					// Increment in the x direction 
					x_incr = i + r;
					if (x_incr < Nx) {
						vort_long_increment  = run_data->w[x_incr * Ny + j] - run_data->w[i * Ny + j];
						for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
							// Cheeck the threshold criteria
							if (max_norm <= R_thresh[l]) {
								gsl_status = gsl_histogram_increment(proc_data->cond_t_w_incr_hist[0][0][r_indx][l], vort_long_increment);
								cond_type = 0;
							}
							else {
								gsl_status = gsl_histogram_increment(proc_data->cond_t_w_incr_hist[1][0][r_indx][l], vort_long_increment);
								cond_type = 1;
							}

							gsl_status = gsl_histogram_increment(proc_data->cond_t_w_incr_hist[2][0][r_indx][l], vort_long_increment);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Vorticity Increment", s, gsl_status, vort_long_increment);
								exit(1);
							}
						}
					}

					// Increment in the y direction
					y_incr = j + r; 
					if (y_incr < Ny) {
						vort_trans_increment = run_data->w[i * Ny + y_incr] - run_data->w[i * Ny + j];
						for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
							// Cheeck the threshold criteria
							if (max_norm <= R_thresh[l]) {
								gsl_status = gsl_histogram_increment(proc_data->cond_t_w_incr_hist[0][1][r_indx][l], vort_trans_increment);
								cond_type = 0;
							}
							else {
								gsl_status = gsl_histogram_increment(proc_data->cond_t_w_incr_hist[1][1][r_indx][l], vort_trans_increment);
								cond_type = 1;
							}

							gsl_status = gsl_histogram_increment(proc_data->cond_t_w_incr_hist[2][1][r_indx][l], vort_trans_increment);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Vorticity Increment", s, gsl_status, vort_trans_increment);
								exit(1);
							}
						}			
					}
				}
			}
		}

		// --------------------------------	
		//	Compute Joint Stats
		// --------------------------------
		for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
			tmp_sync = cabs(proc_data->phase_order_C_theta_triads[0][a]);
			tmp_flux = creal(proc_data->phase_order_C_theta_triads[0][a]);
			gsl_status = gsl_histogram2d_increment(proc_data->joint_sync_enst_flux_hist[0], tmp_sync, tmp_flux);
		}

		// --------------------------------	
		//	Close File
		// --------------------------------
		status = H5Dclose(dset);
		status = H5Fclose(file_info->output_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, s);
			exit(1);	
		}

		// End timer for current loop
		double loop_end = omp_get_wtime();

		// Print update to screen
		printf("Snapshot: %d/%ld\tTime: %g(s)\n", s + 1, sys_vars->num_snaps, (loop_end - loop_begin));
	}

	// --------------------------------	
	//	Write Data To File
	// --------------------------------
	// Open file with default I/O access properties
	file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
	if (file_info->output_file_handle < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at end\n-->> Exiting...\n", file_info->output_file_name);
		exit(1);
	}

	//-------- Write Increment Stats
	// Allocate temp momery
	int num_bins = proc_data->cond_t_w_incr_hist[0][0][0][0]->n;
	double* tmp_inc_hist_ranges = (double* )fftw_malloc(sizeof(double) * NUM_COND_TYPES * INCR_TYPES * NUM_INCR * NUM_THRESH_TYPES * (num_bins + 1));
	double* tmp_inc_hist_counts = (double* )fftw_malloc(sizeof(double) * NUM_COND_TYPES * INCR_TYPES * NUM_INCR * NUM_THRESH_TYPES * num_bins);
	static const hsize_t Dims5D = 5;
	hsize_t dset_dims_5d[Dims5D];

	for (int i = 0; i < NUM_COND_TYPES; ++i) {
		for (int j = 0; j < INCR_TYPES; ++j) {
			for (int k = 0; k < NUM_INCR; ++k) {
				for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
					for (int b = 0; b < num_bins + 1; ++b) {
						if (b < num_bins) {
							tmp_inc_hist_counts[num_bins * (NUM_THRESH_TYPES * (NUM_INCR * (i * INCR_TYPES + j) + k) + l) + b] = proc_data->cond_t_w_incr_hist[i][j][k][l]->bin[b];
						}
						tmp_inc_hist_ranges[(num_bins + 1) * (NUM_THRESH_TYPES * (NUM_INCR * (i * INCR_TYPES + j) + k) + l) + b] = proc_data->cond_t_w_incr_hist[i][j][k][l]->range[b];
					}
				}
			}
		}
	}

	// Write Counts
	dset_dims_5d[0] = NUM_COND_TYPES;
	dset_dims_5d[1] = INCR_TYPES;
	dset_dims_5d[2] = NUM_INCR;
	dset_dims_5d[3] = NUM_THRESH_TYPES;
	dset_dims_5d[4] = num_bins;
   	status = H5LTmake_dataset(file_info->output_file_handle, "SyncConditional_VortcityIncrement_Counts", Dims5D, dset_dims_5d, H5T_NATIVE_DOUBLE, tmp_inc_hist_counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "SyncConditional_VortcityIncrement_Counts");
        exit(1);
    }
    // Write Ranges
	dset_dims_5d[4] = num_bins + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "SyncConditional_VortcityIncrement_Ranges", Dims5D, dset_dims_5d, H5T_NATIVE_DOUBLE, tmp_inc_hist_ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "SyncConditional_VortcityIncrement_Ranges");
        exit(1);
    }	

	fftw_free(tmp_inc_hist_ranges);
	fftw_free(tmp_inc_hist_counts);

	//-------- Write Joint Histgram
	int num_bins_x = proc_data->joint_sync_enst_flux_hist[0]->nx;
	int num_bins_y = proc_data->joint_sync_enst_flux_hist[0]->ny;
	double* tmp_joint_hist_ranges_x = (double* )fftw_malloc(sizeof(double) * (num_bins_x + 1));
	double* tmp_joint_hist_ranges_y = (double* )fftw_malloc(sizeof(double) * (num_bins_y + 1));
	double* tmp_joint_hist_counts   = (double* )fftw_malloc(sizeof(double) * num_bins_x * num_bins_y);
	static const hsize_t Dims1D = 1;
	hsize_t dset_dims_1d[Dims1D];
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims_2d[Dims2D];

	for (int i = 0; i < num_bins_x + 1; ++i) {
		tmp_joint_hist_ranges_x[i] = proc_data->joint_sync_enst_flux_hist[0]->xrange[i];
	}
	for (int i = 0; i < num_bins_y + 1; ++i) {
		tmp_joint_hist_ranges_y[i] = proc_data->joint_sync_enst_flux_hist[0]->yrange[i];
	}
	for (int i = 0; i < num_bins_x; ++i) {
		for (int j = 0; j < num_bins_y; ++j) {
			tmp_joint_hist_counts[i * num_bins_y + j] = proc_data->joint_sync_enst_flux_hist[0]->bin[i * num_bins_y + j];
		}
	}

	// Write ranges
	dset_dims_1d[0] = num_bins_x + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "SyncConditional_JointHist_Ranges_x", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, tmp_joint_hist_ranges_x);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "SyncConditional_JointHist_Ranges_x");
        exit(1);
    }
	dset_dims_1d[0] = num_bins_y + 1;
    status = H5LTmake_dataset(file_info->output_file_handle, "SyncConditional_JointHist_Ranges_y", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, tmp_joint_hist_ranges_y);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "SyncConditional_JointHist_Ranges_y");
        exit(1);
    }
    // Write counts
    dset_dims_2d[0] = num_bins_x;
	dset_dims_2d[1] = num_bins_y;
   	status = H5LTmake_dataset(file_info->output_file_handle, "SyncConditional_JointHist_Counts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp_joint_hist_counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "SyncConditional_JointHist_Counts");
        exit(1);
    }

    fftw_free(tmp_joint_hist_ranges_x);
    fftw_free(tmp_joint_hist_ranges_y);
	fftw_free(tmp_joint_hist_counts);

	//----------- Max Norm Time series
	dset_dims_1d[0] = sys_vars->num_snaps;
    status = H5LTmake_dataset(file_info->output_file_handle, "SyncConditional_MaxNorm_Tseries", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, tmp_max_norm);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "SyncConditional_MaxNorm_Tseries");
        exit(1);
    }

	// Close file
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at end\n-->> Exiting...\n", file_info->output_file_name);
		exit(1);	
	}

	// --------------------------------	
	//	Free temp Memory
	// --------------------------------
	fftw_free(tmp_collec_phase);
	fftw_free(tmp_max_norm);
}
/**
 * Allocate memory and objects needed for the Phase sync computation
 * @param N Array containing the size of the each dimension
 */
void AllocatePhaseSyncMemory(const long int* N) {

	// Initialize variables
	int gsl_status;
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;


	// --------------------------------	
	//  Allocate Sector Angles
	// --------------------------------
	// Allocate the array of sector angles
	proc_data->theta_k3 = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_k3_sectors));
	if (proc_data->theta_k3 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Sector Angles for k3");
		exit(1);
	}
	proc_data->theta_k1 = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_k1_sectors));
	if (proc_data->theta_k1 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Sector Angles for k1");
		exit(1);
	}


	// --------------------------------------------
	//  Allocate Number of Triads & Enstrophy Flux
	// ---------------------------------------------
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		///--------------- Number of Triads
		// Allocate memory for the phase order parameter for the triad phases
		proc_data->num_triads[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k3_sectors);
		if (proc_data->num_triads[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}
		proc_data->num_triads_1d[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k3_sectors);
		if (proc_data->num_triads_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector 1D");
			exit(1);
		}

		///--------------- Enstrophy Flux
		proc_data->enst_flux[i] = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_k3_sectors));
		if (proc_data->enst_flux[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
			exit(1);
		}
		proc_data->enst_flux_1d[i] = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_k3_sectors));
		if (proc_data->enst_flux_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector 1D");
			exit(1);
		}

		///-------------- Number of Triads and Enstrophy Flux across sectors
		// Allocate memory for the number of triads per sector
		proc_data->num_triads_2d[i] = (int** )fftw_malloc(sizeof(int* ) * sys_vars->num_k3_sectors);
		if (proc_data->num_triads_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}
		// Allocate memory for the flux of enstrophy across sectors
		proc_data->enst_flux_2d[i] = (double** )fftw_malloc(sizeof(double* ) * sys_vars->num_k3_sectors);
		if (proc_data->enst_flux_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
			exit(1);
		}	
		for (int l = 0; l < sys_vars->num_k3_sectors; ++l) {
			proc_data->num_triads_2d[i][l] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k1_sectors);
			if (proc_data->num_triads_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
				exit(1);
			}
			proc_data->enst_flux_2d[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
			if (proc_data->enst_flux_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
				exit(1);
			}
		}
	}


	// --------------------------------	
	//  Allocate Mid Angle Sum
	// --------------------------------
	// Allocate memory for the precomputed sector midpoint angle sums -> this is used to determine which sector k2 is
	proc_data->mid_angle_sum = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors);
	if (proc_data->mid_angle_sum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Midpoint Sector Angles");
		exit(1);
	}


	// -------------------------------------
	//  Allocate Precomputed Vector Angles
	// -------------------------------------
	// Allocate memory for the arctangent arrays
	proc_data->phase_angle = (double* )fftw_malloc(sizeof(double) * Nx * Ny_Fourier);
	if (proc_data->phase_angle == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of Negative k2");
		exit(1);
	}

	// Fill the array for the individual phases with the precomputed arctangents
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			proc_data->phase_angle[i * Ny_Fourier + j] = atan2((double) run_data->k[0][i], (double)run_data->k[1][j]);
		}
	}


	// -------------------------------------
	//  Allocate Individual Phases Sync
	// -------------------------------------
	// Allocate memory for the phase order parameter for the individual phases
	proc_data->phase_order = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
	if (proc_data->phase_order == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Phase Order Parameter");
		exit(1);
	}
	// Allocate the array of phase sync per sector for the individual phases
	proc_data->phase_R = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
	if (proc_data->phase_R == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Parameter");
		exit(1);
	}
	// Allocate the array of average phase per sector for the individual phases
	proc_data->phase_Phi = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
	if (proc_data->phase_Phi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Phase");
		exit(1);
	}


	// -------------------------------------
	//  Allocate Triad Phases Sync
	// -------------------------------------
	// Allocate memory for each  of the triad types
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		//--------------- Allocate memory for the collective phase order parameter for theta_k3
		proc_data->phase_order_C_theta_triads[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
		if (proc_data->phase_order_C_theta_triads[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		proc_data->phase_order_C_theta_triads_1d[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
		if (proc_data->phase_order_C_theta_triads_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter 1D");
			exit(1);
		}
		proc_data->phase_order_C_theta_triads_2d[i] = (fftw_complex** )fftw_malloc(sizeof(fftw_complex*) * sys_vars->num_k3_sectors);
		if (proc_data->phase_order_C_theta_triads_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter 2D");
			exit(1);
		}
		proc_data->phase_order_C_theta_triads_unidirec[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
		if (proc_data->phase_order_C_theta_triads_unidirec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Unidirectional Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		proc_data->phase_order_C_theta_triads_unidirec_1d[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
		if (proc_data->phase_order_C_theta_triads_unidirec_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Unidirectional Triad Phase Sync Phase Order Parameter 1D");
			exit(1);
		}
		proc_data->phase_order_C_theta_triads_unidirec_2d[i] = (fftw_complex** )fftw_malloc(sizeof(fftw_complex*) * sys_vars->num_k3_sectors);
		if (proc_data->phase_order_C_theta_triads_unidirec_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Unidirectional Triad Phase Sync Phase Order Parameter 2D");
			exit(1);
		}

		//--------------- Allocate memory for the phase order parameter for the triad phases
		proc_data->triad_phase_order[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
		if (proc_data->triad_phase_order[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
		if (proc_data->triad_R[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
		if (proc_data->triad_Phi[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
			exit(1);
		}

		//--------------- Allocate memory for the phase order parameter for the triad phases for 1d contributions
		proc_data->triad_phase_order_1d[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k3_sectors);
		if (proc_data->triad_phase_order_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter 1D");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R_1d[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
		if (proc_data->triad_R_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter 1D");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi_1d[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
		if (proc_data->triad_Phi_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase 1D");
			exit(1);
		}

		///------------- Allocate memory for arrays across sectors
		/// Allocate memory for the phase order parameter for the triad phases
		proc_data->triad_phase_order_2d[i] = (fftw_complex** )fftw_malloc(sizeof(fftw_complex*) * sys_vars->num_k3_sectors);
		if (proc_data->triad_phase_order_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R_2d[i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_k3_sectors);
		if (proc_data->triad_R_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi_2d[i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_k3_sectors);
		if (proc_data->triad_Phi_2d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
			exit(1);
		}

		///---------------- Allocate the memory for the normalization constants for the phase order parameters
		for (int type = 0; type < 2; ++type){
			proc_data->phase_order_norm_const[type][i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_k3_sectors);
			if (proc_data->phase_order_norm_const[type][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Normalization Constant");
				exit(1);
			}
		}		
		for (int l = 0; l < sys_vars->num_k3_sectors; ++l) {
			// Allocate memory for the collective phase order parameter for 2d contributions to the enstrophy flux
			proc_data->phase_order_C_theta_triads_2d[i][l] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k1_sectors);
			if (proc_data->phase_order_C_theta_triads_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter 2D");
				exit(1);
			}
			// Allocate memory for the collective phase order parameter for 2d contributions to the enstrophy flux
			proc_data->phase_order_C_theta_triads_unidirec_2d[i][l] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k1_sectors);
			if (proc_data->phase_order_C_theta_triads_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Unidirectional Triad Phase Sync Phase Order Parameter 2D");
				exit(1);
			}
			// Allocate memory for the phase order parameter for the triad phases
			proc_data->triad_phase_order_2d[i][l] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k1_sectors);
			if (proc_data->triad_phase_order_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
				exit(1);
			}
			// Allocate the array of phase sync per sector for the triad phases
			proc_data->triad_R_2d[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
			if (proc_data->triad_R_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
				exit(1);
			}
			// Allocate the array of average phase per sector for the triad phases
			proc_data->triad_Phi_2d[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
			if (proc_data->triad_Phi_2d[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
				exit(1);
			}
			// Allocate the memory for the normalization constants for the phase order parameters
			for (int type = 0; type < 2; ++type){
				proc_data->phase_order_norm_const[type][i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
				if (proc_data->phase_order_norm_const[type][i][l] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Normalization Constant");
					exit(1);
				}
			}
		}
	}

	//--------------- Initialize arrays
	proc_data->dtheta_k3 = M_PI / (double )sys_vars->num_k3_sectors;
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
		proc_data->theta_k3[i] = -M_PI / 2.0 + i * proc_data->dtheta_k3 + proc_data->dtheta_k3 / 2.0 + 1e-10;
		proc_data->phase_R[i]     = 0.0;
		proc_data->phase_Phi[i]   = 0.0;
		proc_data->phase_order[i] = 0.0 + 0.0 * I;
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			proc_data->num_triads[j][i]                             = 0;
			proc_data->enst_flux[j][i]                              = 0.0;
			proc_data->triad_R[j][i]                                = 0.0;
			proc_data->triad_Phi[j][i]                              = 0.0;
			proc_data->num_triads_1d[j][i]                          = 0;                        
			proc_data->enst_flux_1d[j][i]                           = 0.0;                  
			proc_data->triad_R_1d[j][i]                             = 0.0;                  
			proc_data->triad_Phi_1d[j][i]                           = 0.0;                    
			proc_data->triad_phase_order[j][i]                      = 0.0 + 0.0 * I;
			proc_data->triad_phase_order_1d[j][i]                   = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads[j][i]             = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_1d[j][i]          = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_unidirec[j][i]    = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_unidirec_1d[j][i] = 0.0 + 0.0 * I;
			for (int k = 0; k < sys_vars->num_k1_sectors; ++k) {
				proc_data->num_triads_2d[j][i][k]                          = 0;
				proc_data->enst_flux_2d[j][i][k]                           = 0.0;
				proc_data->triad_R_2d[j][i][k]                             = 0.0;
				proc_data->triad_Phi_2d[j][i][k]                           = 0.0;
				proc_data->triad_phase_order_2d[j][i][k]                   = 0.0 + 0.0 * I;
				proc_data->phase_order_C_theta_triads_2d[j][i][k]          = 0.0 + 0.0 * I;
				proc_data->phase_order_C_theta_triads_unidirec_2d[j][i][k] = 0.0 + 0.0 * I;
				proc_data->phase_order_norm_const[0][j][i][k] = 0.0;
				proc_data->phase_order_norm_const[1][j][i][k] = 0.0;
			}
		}
	}
	// Initialize k1 sector array
	proc_data->dtheta_k1 = M_PI / (double )sys_vars->num_k1_sectors;
	for (int i = 0; i < sys_vars->num_k1_sectors; ++i) {
		proc_data->theta_k1[i] = -M_PI / 2.0 + i * proc_data->dtheta_k1 + proc_data->dtheta_k1 / 2.0 + 1e-10;
	}


	// -------------------------------------
	//  Allocate Phases Sync Stats
	// -------------------------------------
	///------------------------------------- Allocate memory for the In time Stats -> triad pdfs and weighted flux pdfs
	#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D) || defined(__SEC_PHASE_SYNC_COND_STATS)
	for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
		for (int triad_type = 0; triad_type < NUM_TRIAD_TYPES + 1; ++triad_type) {
			proc_data->triads_sect_all_pdf_t[triad_class][triad_type]       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
			proc_data->triads_sect_wghtd_all_pdf_t[triad_class][triad_type] = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
       	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
			proc_data->triads_sect_1d_pdf_t[triad_class][triad_type]        = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
			proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][triad_type]  = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
			#endif
			#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
			proc_data->triads_sect_2d_pdf_t[triad_class][triad_type]        = (gsl_histogram*** )fftw_malloc(sizeof(gsl_histogram**) * sys_vars->num_k3_sectors);
			proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type]  = (gsl_histogram*** )fftw_malloc(sizeof(gsl_histogram**) * sys_vars->num_k3_sectors);
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a]       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k1_sectors);
				proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type][a] = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k1_sectors);
			}
			#endif
			#if defined(__SEC_PHASE_SYNC_COND_STATS)
			if (triad_class == 0) {
				// Allocate memory for the maximum sync over the sectors
				proc_data->max_sync[triad_type] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors);
				for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
					proc_data->max_sync[triad_type][a] = 0.0;
				}
			}
			#endif
		}
	}
	#endif

	///------------------------------------- Initialize In Time PDFs -> triad pdfs and weighted flux pdfs
	#if defined(__SEC_PHASE_SYNC_FLUX_STATS) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
	for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
		for (int triad_type = 0; triad_type < NUM_TRIAD_TYPES + 1; ++triad_type) {
			#if defined(__SEC_PHASE_SYNC_FLUX_STATS)
			///-------------------------------------- Allocate Running stats for the enstrophy flux
			if (triad_type < 5) {
				proc_data->max_enst_flux[triad_class][triad_type] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
				for (int i = 0; i < sys_vars->num_snaps; ++i) {
					proc_data->max_enst_flux[triad_class][triad_type][i] = 0.0;
				}

				// Allocate the 2d histogram structs
				proc_data->triads_wghtd_2d_pdf_t[triad_class][triad_type] = gsl_histogram2d_alloc(N_BINS_X_JOINT_ALL_T, N_BINS_Y_JOINT_ALL_T);
				if (proc_data->triads_wghtd_2d_pdf_t[triad_class][triad_type] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDFs All Contributions");
					exit(1);
				}
			}
			#endif
			#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME)
			///-------------------------------------- Allocate histogram structs for all contributions
			proc_data->triads_all_pdf_t[triad_class][triad_type] = gsl_histogram_alloc(N_BINS_TRIADS_ALL_T);
			if (proc_data->triads_all_pdf_t[triad_class][triad_type] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDFs All Contributions");
				exit(1);
			}	
			proc_data->triads_wghtd_all_pdf_t[triad_class][triad_type] = gsl_histogram_alloc(N_BINS_TRIADS_ALL_T);
			if (proc_data->triads_all_pdf_t[triad_class][triad_type] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "All Weight Triad Phase PDFs All Contributions");
				exit(1);
			}

			// Initialize bins for all contirbutions
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_all_pdf_t[triad_class][triad_type], -M_PI,  M_PI);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDFs All Contribution");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_wghtd_all_pdf_t[triad_class][triad_type], -M_PI,  M_PI);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "All Triad Phase PDFs All Contribution");
				exit(1);
			}
			#endif
			#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				///-------------------------------------- Allocate histogram structs for all contributions
				proc_data->triads_sect_all_pdf_t[triad_class][triad_type][a] = gsl_histogram_alloc(N_BINS_SEC_ALL_T);
				if (proc_data->triads_sect_all_pdf_t[triad_class][triad_type][a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs All Contributions");
					exit(1);
				}	
				proc_data->triads_sect_wghtd_all_pdf_t[triad_class][triad_type][a] = gsl_histogram_alloc(N_BINS_SEC_ALL_T);
				if (proc_data->triads_sect_all_pdf_t[triad_class][triad_type][a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Weight Triad Phase PDFs All Contributions");
					exit(1);
				}

				// Initialize bins for all contirbutions
				gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_sect_all_pdf_t[triad_class][triad_type][a], -M_PI,  M_PI);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs All Contribution");
					exit(1);
				}
				gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][triad_type][a], -M_PI,  M_PI);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs All Contribution");
					exit(1);
				}

	      	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
				///-------------------------------------- Allocate histogram structs for 1d contributions
				// Initialize histogram for 1d and 2d contributions
				proc_data->triads_sect_1d_pdf_t[triad_class][triad_type][a] = gsl_histogram_alloc(N_BINS_SEC_1D_T);
				if (proc_data->triads_sect_1d_pdf_t[triad_class][triad_type][a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs 1D Contributions");
					exit(1);
				}	
				proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][triad_type][a] = gsl_histogram_alloc(N_BINS_SEC_1D_T);
				if (proc_data->triads_sect_1d_pdf_t[triad_class][triad_type][a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Weight Triad Phase PDFs 1D Contributions");
					exit(1);
				}
				// Initialize bins for all contirbutions
				gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_sect_1d_pdf_t[triad_class][triad_type][a], -M_PI,  M_PI);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs 1D Contribution");
					exit(1);
				}
				gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][triad_type][a], -M_PI,  M_PI);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs 1D Contribution");
					exit(1);
				}
				#endif

        	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
				///-------------------------------------- Allocate histogram structs for all contributions
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					// Initialize histogram for 1d and 2d contributions
					proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a][l] = gsl_histogram_alloc(N_BINS_SEC_2D_T);
					if (proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a][l] == NULL) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs 2D Contributions");
						exit(1);
					}	
					proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type][a][l] = gsl_histogram_alloc(N_BINS_SEC_2D_T);
					if (proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a][l] == NULL) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Weight Triad Phase PDFs 2D Contributions");
						exit(1);
					}				

					///-------------------------------------- Initialize histogram bin ranges
					// Initialize bins for all contirbutions
					gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a][l], -M_PI,  M_PI);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs 1D + 2D Contribution");
						exit(1);
					}
					gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type][a][l], -M_PI,  M_PI);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase PDFs 1D + 2D Contribution");
						exit(1);
					}
				}
				#endif
			}
			#endif
		}
	}	
	#endif
	#if defined(__SEC_PHASE_SYNC_STATS) 
	// Allocate memory for the arrays stats objects
	proc_data->phase_sect_pdf         = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
	proc_data->phase_sect_pdf_t       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
	proc_data->phase_sect_wghtd_pdf_t = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
	proc_data->triad_R_2d_pdf         = (gsl_histogram*** )fftw_malloc(sizeof(gsl_histogram**) * sys_vars->num_k3_sectors);
	proc_data->triad_Phi_2d_pdf       = (gsl_histogram*** )fftw_malloc(sizeof(gsl_histogram**) * sys_vars->num_k3_sectors);
	proc_data->enst_flux_2d_pdf       = (gsl_histogram*** )fftw_malloc(sizeof(gsl_histogram**) * sys_vars->num_k3_sectors);
	for (int m = 0; m < NUM_MOMMENTS; ++m) {
		proc_data->triad_R_2d_stats[m]       = (gsl_rstat_workspace*** )fftw_malloc(sizeof(gsl_rstat_workspace**) * sys_vars->num_k3_sectors);
		proc_data->triad_Phi_2d_stats[m]     = (gsl_rstat_workspace*** )fftw_malloc(sizeof(gsl_rstat_workspace**) * sys_vars->num_k3_sectors);
		proc_data->enst_flux_2d_stats[m]     = (gsl_rstat_workspace*** )fftw_malloc(sizeof(gsl_rstat_workspace**) * sys_vars->num_k3_sectors);
		for (int l = 0; l < sys_vars->num_k3_sectors; ++l) {
			proc_data->triad_R_2d_stats[m][l]   = (gsl_rstat_workspace** )fftw_malloc(sizeof(gsl_rstat_workspace*) * sys_vars->num_k1_sectors);
			proc_data->triad_Phi_2d_stats[m][l] = (gsl_rstat_workspace** )fftw_malloc(sizeof(gsl_rstat_workspace*) * sys_vars->num_k1_sectors);
			proc_data->enst_flux_2d_stats[m][l] = (gsl_rstat_workspace** )fftw_malloc(sizeof(gsl_rstat_workspace*) * sys_vars->num_k1_sectors);
		}
	}
	for (int l = 0; l < sys_vars->num_k3_sectors; ++l) {
		proc_data->triad_R_2d_pdf[l]     = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k1_sectors);
		proc_data->triad_Phi_2d_pdf[l]   = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k1_sectors);
		proc_data->enst_flux_2d_pdf[l]   = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k1_sectors);
	}
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		proc_data->triad_sect_pdf[i]         = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
		proc_data->triad_sect_pdf_t[i]       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
		proc_data->triad_sect_wghtd_pdf_t[i] = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram*) * sys_vars->num_k3_sectors);
	}
	
	// Allocate stats objects for each sector and set ranges
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
		// Allocate pdfs for the individual phases
		proc_data->phase_sect_pdf[i] = gsl_histogram_alloc(N_BINS_SEC);
		if (proc_data->phase_sect_pdf[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF");
			exit(1);
		}	
		proc_data->phase_sect_pdf_t[i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
		if (proc_data->phase_sect_pdf_t[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time");
			exit(1);
		}	
		proc_data->phase_sect_wghtd_pdf_t[i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
		if (proc_data->phase_sect_wghtd_pdf_t[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phase Weighted PDF In Time");
			exit(1);
		}	
		// Set bin ranges for the individual phases
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_pdf[i], -M_PI,  M_PI);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases PDF");
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_pdf_t[i], -M_PI,  M_PI);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases PDF In Time");
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_wghtd_pdf_t[i], -M_PI,  M_PI);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases Weighted PDF In Time");
			exit(1);
		}

		// Initialize the 2D contribution histograms and stats
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
			proc_data->triad_R_2d_pdf[j][i][l] = gsl_histogram_alloc(N_BINS_SEC);
			if (proc_data->triad_R_2d_pdf[j][i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync 2D PDF");
				exit(1);
			}
			proc_data->triad_Phi_2d_pdf[j][i][l] = gsl_histogram_alloc(N_BINS_SEC);
			if (proc_data->triad_Phi_2d_pdf[j][i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Phase 2D PDF");
				exit(1);
			}
			proc_data->enst_flux_2d_pdf[j][i][l] = gsl_histogram_alloc(N_BINS_SEC);
			if (proc_data->enst_flux_2d_pdf[j][i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enst Flux 2D PDF");
				exit(1);
			}

			// Set the bin ranges
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_R_2d_pdf[j][i][l], 0.0 - 0.05,  1.0 + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync 2D PDF");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_Phi_2d_pdf[j][i][l], -M_PI,  M_PI);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Phase 2D PDF");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->enst_flux_2d_pdf[j][i][l], -1e10 - 0.05,  1e10 + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux PDF");
				exit(1);
			}

			// Initialize stats objects
			proc_data->triad_R_2d_stats[i][l]   = gsl_rstat_alloc();
			proc_data->triad_Phi_2d_stats[i][l] = gsl_rstat_alloc();
			proc_data->enst_flux_2d_stats[i][l] = gsl_rstat_alloc();
		}

		// Allocate and set ranges for each of the triad types
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {

			// Allocate pdfs for the triad phases
			proc_data->triad_sect_pdf[j][i] = gsl_histogram_alloc(N_BINS_SEC);
			if (proc_data->triad_sect_pdf[j][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF");
				exit(1);
			}	
			proc_data->triad_sect_pdf_t[j][i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
			if (proc_data->triad_sect_pdf_t[j][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time");
				exit(1);
			}
			proc_data->triad_sect_wghtd_pdf_t[j][i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
			if (proc_data->triad_sect_wghtd_pdf_t[j][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time");
				exit(1);
			}	
			// Set bin ranges for the triad phases
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_pdf[j][i], -M_PI,  M_PI);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_pdf_t[j][i], -M_PI,  M_PI);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF In Time");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_wghtd_pdf_t[j][i], -M_PI,  M_PI);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF In Time");
				exit(1);
			}
		}
	}
	#endif


	// -------------------------------------
	//  Allocate Triad Wavevector Array
	// -------------------------------------
	#if defined (__PHASE_SYNC)
	// Allocate the wavevector objects for the phase sync test computation
	AllocateWavecData(SYNC_TEST_DATA_FLAG);
	#endif

	// Allocate the wavevector objects for the phase sync test computation
	AllocateWavecData(SYNC_DATA_FLAG);


}
void AllocateWavecData(int data_flag) {

	// Initialize vairables
	int k3_x, k3_y, k2_x, k2_y, k1_x, k1_y;
	double k1_sqr, k2_sqr, k3_sqr;

	// --------------------------------
	//  Phase Sync: Test Data Size Search
	// --------------------------------
	if (data_flag == SYNC_TEST_DATA_FLAG) {
		int test_data_size = 0;

		// --------------------------------
		//  Pre Search to Find Data Size
		// --------------------------------
		// Loop through the wavevectors to find the triad wavevector data
		// That falls into each sector
		printf("\n["YELLOW"NOTE"RESET"] --- Performing search over wavevectors for Phase Sync -TEST- computation...\n");
		
		// Loop through the k wavevector (k is the k3 wavevector)
		for (int tmp_k3_x = 0; tmp_k3_x <= sys_vars->N[0] - 1; ++tmp_k3_x) {
			
			// Get k3_x
			k3_x = tmp_k3_x - (int) (sys_vars->N[0] / 2) + 1;

			for (int tmp_k3_y = 0; tmp_k3_y <= sys_vars->N[1] - 1; ++tmp_k3_y) {
				
				// Get k3_y
				k3_y = tmp_k3_y - (int) (sys_vars->N[1] / 2) + 1;

				// Get polar coords for the k wavevector
				k3_sqr       = (double) (k3_x * k3_x + k3_y * k3_y);
				
				if ((k3_sqr > sys_vars->kmax_C_sqr && k3_sqr <= sys_vars->kmax_sqr)) {

					// Loop through the k1 wavevector
					for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
						
						// Get k1_x
						k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

						for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
							
							// Get k1_y
							k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

							// Get polar coords for k1
							k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);

							if((k1_sqr > 0.0 && k1_sqr <= sys_vars->kmax_C_sqr)) {
								
								// Find the k2 wavevector
								k2_x = k3_x - k1_x;
								k2_y = k3_y - k1_y;
								
								// Get polar coords for k2
								k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);

								if ((k2_sqr > 0.0 && k2_sqr <= sys_vars->kmax_C_sqr)) {

									// Increment test data size counter
									test_data_size++;
								}
							}
						}
					}
				}
				else if ((k3_sqr > 0.0 && k3_sqr <= sys_vars->kmax_C_sqr)) {

					// Loop through the k1 wavevector
					for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
						
						// Get k1_x
						k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

						for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
							
							// Get k1_y
							k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

							// Get polar coords for k1
							k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);

							if((k1_sqr > sys_vars->kmax_C_sqr && k1_sqr <= sys_vars->kmax_sqr)) {									
								
								// Find the k2 wavevector
								k2_x = k3_x - k1_x;
								k2_y = k3_y - k1_y;
								
								// Get polar coords for k2
								k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);

								if ((k2_sqr > sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_sqr)) {

									// Increment test data size counter
									test_data_size++;
								}
							}
						}	
					}
				}
			}
		}

		// --------------------------------
		//  Allocate Memory with Appropriate Size
		// --------------------------------
		proc_data->phase_sync_wave_vecs_test = (int** )fftw_malloc(sizeof(int*) * 6);
		if (proc_data->phase_sync_wave_vecs_test == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
			exit(1);
		}	
		for (int n = 0; n < 6; ++n) {
			proc_data->phase_sync_wave_vecs_test[n] = (int* )fftw_malloc(sizeof(int) * test_data_size);
			if (proc_data->phase_sync_wave_vecs_test[n] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
				exit(1);
			}
		}
		// Initialize phase sync test data
		for (int n = 0; n < 6; ++n) {
			proc_data->num_triads_test[n] = 0;
			for (int i = 0; i < test_data_size; ++i) {
				proc_data->phase_sync_wave_vecs_test[n][i] = 0.0;
			}
		}
	}
	// --------------------------------
	//  Phase Sync: Data Size Search
	// --------------------------------
	else if (data_flag == SYNC_DATA_FLAG) {

		// Allocate k1 sector angles
		double k1_sector_angles[NUM_K1_SECTORS] = {-sys_vars->num_k3_sectors/2.0, -sys_vars->num_k3_sectors/3.0, -sys_vars->num_k3_sectors/4.0, -sys_vars->num_k3_sectors/6.0, sys_vars->num_k3_sectors/6.0, sys_vars->num_k3_sectors/4.0, sys_vars->num_k3_sectors/3.0, sys_vars->num_k3_sectors/2.0};
		proc_data->k1_sector_angles = (double* )fftw_malloc(sizeof(double) * NUM_K1_SECTORS);
		memcpy(proc_data->k1_sector_angles, k1_sector_angles, sizeof(k1_sector_angles));

		// --------------------------------
		//  Wavevector Data File Check
		// --------------------------------
		// Check if wavevector data file exist already and read in data from that
		// No size search or object allocation is needed if file exists
		
		//----------- Get Phase Sync wavevector data file path
		herr_t status;
		char dset_name[64];
		char wave_vec_file[128];
		strncpy(file_info->wave_vec_data_name, "./Data/PostProcess/PhaseSync", 1024);
		sprintf(wave_vec_file, "/Wavevector_Data_N[%d,%d]_SECTORS[%d,%d]_KFRAC[%1.2lf,%1.2lf].h5", (int)sys_vars->N[0], (int)sys_vars->N[1], sys_vars->num_k3_sectors, sys_vars->num_k1_sectors, sys_vars->kmin_sqr, sys_vars->kmax_frac);	
		strcat(file_info->wave_vec_data_name, wave_vec_file);
		
		////////////////////////////////////////////////////////////////////
		// Check if Wavector file exists: IF FILE DOES EXIST
		///////////////////////////////////////////////////////////////////
		if (access(file_info->wave_vec_data_name, F_OK) == 0) {
			printf("\n["YELLOW"NOTE"RESET"] --- Reading in wavevectors data for Phase Sync computation...");
			printf("\nNumber of k_3 Sectors: ["CYAN"%d"RESET"]\nNumber of k_1 Sectors: ["CYAN"%d"RESET"]\n\n", sys_vars->num_k3_sectors, sys_vars->num_k1_sectors);
			
			// --------------------------------
			//  Open Phase Sync Data File
			// --------------------------------
			//----------------------- Open file with default I/O access properties
			file_info->wave_vec_file_handle = H5Fopen(file_info->wave_vec_data_name, H5F_ACC_RDWR, H5P_DEFAULT);
			if (file_info->wave_vec_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->wave_vec_data_name);
				exit(1);
			}

			// --------------------------------
			//  Read in Num Triads Data
			// --------------------------------
			//------------ Allocate memory for the number of wavevector triads per sector
			proc_data->num_wave_vecs = (int** )fftw_malloc(sizeof(int*) * sys_vars->num_k3_sectors);
			if (proc_data->num_wave_vecs == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Phase Sync Wavevectors");
				exit(1);
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				proc_data->num_wave_vecs[a] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k1_sectors);
				if (proc_data->num_wave_vecs[a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Phase Sync Wavevectors");
					exit(1);
				}	
			}

			///----------------------- Read in Number of Triad Wavectors Data
			// Read in the number of wavevectors
			int* tmp_num_wavevecs = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors);
			if(H5LTread_dataset(file_info->wave_vec_file_handle, "NumWavevectors", H5T_NATIVE_INT, tmp_num_wavevecs) < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Wavevectors");
				exit(1);	
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					proc_data->num_wave_vecs[a][l] = tmp_num_wavevecs[a * sys_vars->num_k1_sectors + l];
				}
			}


			// --------------------------------
			//  Read in Wavevector Data
			// --------------------------------
			//------------ Allocate memory for the triad wavevectors per sector and their data 
			proc_data->phase_sync_wave_vecs = (int**** )fftw_malloc(sizeof(int***) * sys_vars->num_k3_sectors);
			if (proc_data->phase_sync_wave_vecs == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
				exit(1);
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				proc_data->phase_sync_wave_vecs[a] = (int*** )fftw_malloc(sizeof(int**) * sys_vars->num_k1_sectors);
				if (proc_data->phase_sync_wave_vecs[a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
					exit(1);
				}	
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					proc_data->phase_sync_wave_vecs[a][l] = (int** )fftw_malloc(sizeof(int*) * NUM_K_DATA);
					if (proc_data->phase_sync_wave_vecs[a][l] == NULL) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
						exit(1);
					}	
				}
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					for (int n = 0; n < NUM_K_DATA; ++n) {
						if (proc_data->num_wave_vecs[a][l] != 0) {
							proc_data->phase_sync_wave_vecs[a][l][n] = (int* )fftw_malloc(sizeof(int) * proc_data->num_wave_vecs[a][l]);
							if (proc_data->phase_sync_wave_vecs[a][l][n] == NULL) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
								exit(1);
							}
						}
					}	
				}
			}

			//----------- Read in wavevector data
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					int* tmp_wave_vec_data = (int*)fftw_malloc(sizeof(int) * NUM_K_DATA * proc_data->num_wave_vecs[a][l]);
					sprintf(dset_name, "WVData_Sector_%d_%d", a, l);

					if(H5LTread_dataset(file_info->wave_vec_file_handle, dset_name, H5T_NATIVE_INT, tmp_wave_vec_data) < 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", dset_name);
						exit(1);	
					}
			
					for (int k = 0; k < NUM_K_DATA; ++k) {
						for (int n = 0; n < proc_data->num_wave_vecs[a][l]; ++n) {
							proc_data->phase_sync_wave_vecs[a][l][k][n] = tmp_wave_vec_data[k * proc_data->num_wave_vecs[a][l] + n];
						}
					}
					fftw_free(tmp_wave_vec_data);
				}
			}
			// Free temporary memory
			fftw_free(tmp_num_wavevecs);

			// --------------------------------
			//  Open Phase Sync Data File
			// --------------------------------
			status = H5Fclose(file_info->wave_vec_file_handle);
			if (status < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close wavevector data file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->wave_vec_data_name);
				exit(1);		
			}
		}
		////////////////////////////////////////////////////////////////////
		// Check if Wavector file exists: IF FILE DOES NOT EXIST
		///////////////////////////////////////////////////////////////////
		else {
			int k1_sec_indx; 
			double C_theta_k3, C_theta_k1, C_theta_k2;
			double C_theta_k3_lwr, C_theta_k3_upr, C_theta_k1_upr, C_theta_k1_lwr;
			double k1_angle, k2_angle, k3_angle;
			double k1_angle_neg, k2_angle_neg, k3_angle_neg;
			long int pre_search_terms = 0;

			// --------------------------------
			//  Pre Search to Find Data Size
			// --------------------------------
			// Loop through the wavevectors to find the triad wavevector data
			// That falls into each sector
			// Print to screen that a pre computation search is needed for the phase sync wavevectors and begin timeing it
			printf("\n["YELLOW"NOTE"RESET"] --- Performing pre-search for number wavevectors for Phase Sync computation...\n");

			//------------ Allocate memory for the number of wavevector triads per sector
			proc_data->num_wave_vecs = (int** )fftw_malloc(sizeof(int*) * sys_vars->num_k3_sectors);
			if (proc_data->num_wave_vecs == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Phase Sync Wavevectors");
				exit(1);
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				proc_data->num_wave_vecs[a] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k1_sectors);
				if (proc_data->num_wave_vecs[a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Phase Sync Wavevectors");
					exit(1);
				}	
			}
			// Initialize number of triad terms array
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k3_sectors; ++l) {
					proc_data->num_wave_vecs[a][l] = 0;
				}
			}

			// Loop through the sectors for k3
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				
				// Get the angles for the current sector
				C_theta_k3 = proc_data->theta_k3[a];
				C_theta_k3_lwr = C_theta_k3 - proc_data->dtheta_k3 / 2.0;
				C_theta_k3_upr = C_theta_k3 + proc_data->dtheta_k3 / 2.0;

				// Loop through k1 sector choice
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {

					// Get the angles for the k1 sector
					if (sys_vars->REDUCED_K1_SEARCH_FLAG) {
						// Reduced k1 sector search
						C_theta_k1     = MyMod(C_theta_k3 + (k1_sector_angles[l] * proc_data->dtheta_k3/2.0) + M_PI, 2.0 * M_PI) - M_PI;
						C_theta_k1_lwr = C_theta_k1 - proc_data->dtheta_k3 / 2.0;
						C_theta_k1_upr = C_theta_k1 + proc_data->dtheta_k3 / 2.0;
					}
					else if (sys_vars->num_k1_sectors == 1){
						// When k1 can vary anywhere -> no restriction to sector
						C_theta_k1     = 0.0;
						C_theta_k1_lwr = C_theta_k1 - M_PI / 2.0 + 1e-10;
						C_theta_k1_upr = C_theta_k1 + M_PI / 2.0 + 1e-10;
					}
					else {
						// Full search over sectors
						C_theta_k1     = proc_data->theta_k1[(a + l) % sys_vars->num_k1_sectors];
						C_theta_k1_lwr = C_theta_k1 - proc_data->dtheta_k1 / 2.0;
						C_theta_k1_upr = C_theta_k1 + proc_data->dtheta_k1 / 2.0;
					}

					// Get the index for saving the k1 sector data in the right order -> 1d contribution should be on the diagonal
					k1_sec_indx = (l + a) % sys_vars->num_k1_sectors;
					
					// Loop through the k wavevector (k is the k3 wavevector)
					for (int tmp_k3_x = 0; tmp_k3_x <= sys_vars->N[0] - 1; ++tmp_k3_x) {
						
						// Get k3_x
						k3_x = tmp_k3_x - (int) (sys_vars->N[0] / 2) + 1;

						for (int tmp_k3_y = 0; tmp_k3_y <= sys_vars->N[1] - 1; ++tmp_k3_y) {
							
							// Get k3_y
							k3_y = tmp_k3_y - (int) (sys_vars->N[1] / 2) + 1;

							// Get polar coords for the k wavevector
							k3_sqr       = (double) (k3_x * k3_x + k3_y * k3_y);
							k3_angle     = atan2((double)k3_x, (double)k3_y);
							k3_angle_neg = atan2((double)-k3_x, (double)-k3_y);
							
							if ( (k3_sqr > sys_vars->kmax_C_sqr && k3_sqr <= sys_vars->kmax_sqr) 
								 && ((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr)) ) {  

								// Loop through the k1 wavevector
								for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
									
									// Get k1_x
									k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

									for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
										
										// Get k1_y
										k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

										// Get polar coords for k1
										k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
										k1_angle     = atan2((double) k1_x, (double) k1_y);
										k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

										if( ((k1_sqr > sys_vars->kmin_sqr && k1_sqr <= sys_vars->kmax_C_sqr) 
											&& ((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr)) 
											&& ((k1_angle >= C_theta_k1_lwr && k1_angle < C_theta_k1_upr) || (k1_angle_neg >= C_theta_k1_lwr && k1_angle_neg < C_theta_k1_upr)))
											|| 
											((k1_sqr > sys_vars->kmin_sqr && k1_sqr <= sys_vars->kmax_sqr) 
											&& ((k1_angle >= C_theta_k1_lwr && k1_angle < C_theta_k1_upr) || (k1_angle_neg >= C_theta_k1_lwr && k1_angle_neg < C_theta_k1_upr)) 
											&& !((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr))) ) { 
											
											// Find the k2 wavevector
											k2_x = k3_x - k1_x;
											k2_y = k3_y - k1_y;
											
											// Get polar coords for k2
											k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
											k2_angle     = atan2((double)k2_x, (double) k2_y);
											k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

											if ( (k2_sqr > sys_vars->kmin_sqr && k2_sqr <= sys_vars->kmax_sqr) 
												&& !((k2_sqr > sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_sqr) 
												&& ((k2_angle >= C_theta_k3_lwr && k2_angle < C_theta_k3_upr) || (k2_angle_neg >= C_theta_k3_lwr && k2_angle_neg < C_theta_k3_upr))) ) {

												// Increment data size counter
												proc_data->num_wave_vecs[a][k1_sec_indx]++;
											}
										}
									}
								}
							}
							else if ( ((k3_sqr > sys_vars->kmin_sqr && k3_sqr <= sys_vars->kmax_sqr) && 
								((k3_angle >= C_theta_k1_lwr && k3_angle < C_theta_k1_upr) || (k3_angle_neg >= C_theta_k1_lwr && k3_angle_neg < C_theta_k1_upr)) 
								&& !((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr))) 
								|| 
								((k3_sqr > sys_vars->kmin_sqr && k3_sqr <= sys_vars->kmax_C_sqr) 
								&& ((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr)) 
								&& ((k3_angle >= C_theta_k1_lwr && k3_angle < C_theta_k1_upr) || (k3_angle_neg >= C_theta_k1_lwr && k3_angle_neg < C_theta_k1_upr))) ) { 

								// Loop through the k1 wavevector
								for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
									
									// Get k1_x
									k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

									for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
										
										// Get k1_y
										k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

										// Get polar coords for k1
										k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
										k1_angle     = atan2((double) k1_x, (double) k1_y);
										k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

										if( (k1_sqr > sys_vars->kmax_C_sqr && k1_sqr <= sys_vars->kmax_sqr) 
											&& ((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr)) ) { 
											
											// Find the k2 wavevector
											k2_x = k3_x - k1_x;
											k2_y = k3_y - k1_y;
											
											// Get polar coords for k2
											k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
											k2_angle     = atan2((double)k2_x, (double) k2_y);
											k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

											if ( (k2_sqr > sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_sqr) 
												&& ((k2_angle >= C_theta_k3_lwr && k2_angle < C_theta_k3_upr) || (k2_angle_neg >= C_theta_k3_lwr && k2_angle_neg < C_theta_k3_upr)) ) {
												
												// Increment data size counter
												proc_data->num_wave_vecs[a][k1_sec_indx]++;
											}
										}
									}
								}
							}
						}
					}

					pre_search_terms += proc_data->num_wave_vecs[a][k1_sec_indx];
				}
			}
			printf("["YELLOW"NOTE"RESET"] --- Total number of terms presearch: %ld\n", pre_search_terms);

			// --------------------------------
			//  Allocate Wavevector Data
			// --------------------------------
			//------------ Allocate memory for the triad wavevectors per sector and their data 
			proc_data->phase_sync_wave_vecs = (int**** )fftw_malloc(sizeof(int***) * sys_vars->num_k3_sectors);
			if (proc_data->phase_sync_wave_vecs == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors: Dim 1");
				exit(1);
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				proc_data->phase_sync_wave_vecs[a] = (int*** )fftw_malloc(sizeof(int**) * sys_vars->num_k1_sectors);
				if (proc_data->phase_sync_wave_vecs[a] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors: Dim 2");
					exit(1);
				}	
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					proc_data->phase_sync_wave_vecs[a][l] = (int** )fftw_malloc(sizeof(int*) * NUM_K_DATA);
					if (proc_data->phase_sync_wave_vecs[a][l] == NULL) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors: Dim 3");
						exit(1);
					}	
				}
			}
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					for (int n = 0; n < NUM_K_DATA; ++n) {
						if (proc_data->num_wave_vecs[a][l] != 0) {
							proc_data->phase_sync_wave_vecs[a][l][n] = (int* )fftw_malloc(sizeof(int) * proc_data->num_wave_vecs[a][l]);
							if (proc_data->phase_sync_wave_vecs[a][l][n] == NULL) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors: Dim 4");
								exit(1);
							}
						}
					}	
				}
			}

			// --------------------------------
			//  Wavevector Search
			// --------------------------------
			int nn = 0;
			long int total_terms = 0;
			long int total_terms_per_sec = 0;
			long int total_1d_terms_per_sec = 0;
			long int total_2d_terms_per_sec = 0;
			double search_end, search_begin;

			//-------- Inialize pointer to txt file and open for writing the numbers of terms
			FILE *fptr;
			char num_terms_file[64];
			char num_terms_file_path[1024];
			sprintf(num_terms_file, "/NumFluxTerms_SECTORS[%d,%d]_KFRAC[%1.2lf,%1.2lf].txt", sys_vars->num_k3_sectors, sys_vars->num_k1_sectors, sys_vars->kmin_sqr, sys_vars->kmax_frac);
			strncpy(num_terms_file_path, file_info->output_dir, 1024);
			strcat(num_terms_file_path, num_terms_file);
			fptr = fopen(num_terms_file_path,"w");

			//----------- Start timer
			double loop_begin = omp_get_wtime();

			// Print to screen that a pre computation search is needed for the phase sync wavevectors and begin timeing it
			printf("\n["YELLOW"NOTE"RESET"] --- Performing search over wavevectors for Phase Sync computation...\n");
			printf("\nNumber of k_3 Sectors: ["CYAN"%d"RESET"]\nNumber of k_1 Sectors: ["CYAN"%d"RESET"]\n\n", sys_vars->num_k3_sectors, sys_vars->num_k1_sectors);

			//---------- Loop through the sectors for k3
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {

				// Initialize counter for terms per sector and start time of the current loop
				double search_begin = omp_get_wtime();
				total_terms_per_sec    = 0;
				total_1d_terms_per_sec = 0;
				total_2d_terms_per_sec = 0;
				
				// Get the angles for the current sector
				C_theta_k3 = proc_data->theta_k3[a];
				C_theta_k3_lwr = C_theta_k3 - proc_data->dtheta_k3 / 2.0;
				C_theta_k3_upr = C_theta_k3 + proc_data->dtheta_k3 / 2.0;

				// Loop through k1 sector choice
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {

					// Get the angles for the k1 sector
					if (sys_vars->REDUCED_K1_SEARCH_FLAG) {
						// Reduced k1 sector search
						C_theta_k1     = MyMod(C_theta_k3 + (k1_sector_angles[l] * proc_data->dtheta_k3/2.0) + M_PI, 2.0 * M_PI) - M_PI;
						C_theta_k1_lwr = C_theta_k1 - proc_data->dtheta_k3 / 2.0;
						C_theta_k1_upr = C_theta_k1 + proc_data->dtheta_k3 / 2.0;
					}
					else if (sys_vars->num_k1_sectors == 1){
						// When k1 can vary anywhere -> no restriction to sector
						C_theta_k1     = 0.0;
						C_theta_k1_lwr = C_theta_k1 - M_PI / 2.0 + 1e-10;
						C_theta_k1_upr = C_theta_k1 + M_PI / 2.0 + 1e-10;
					}
					else {
						// Full search over sectors
						C_theta_k1     = proc_data->theta_k1[(a + l) % sys_vars->num_k1_sectors];
						C_theta_k1_lwr = C_theta_k1 - proc_data->dtheta_k1 / 2.0;
						C_theta_k1_upr = C_theta_k1 + proc_data->dtheta_k1 / 2.0;
					}

					// Get the index for saving the k1 sector data in the right order -> 1d contribution should be on the diagonal
					k1_sec_indx = (l + a) % sys_vars->num_k1_sectors;

					// Initialize increment
					nn = 0;

					// Loop through the k wavevector (k is the k3 wavevector)
					for (int tmp_k3_x = 0; tmp_k3_x <= sys_vars->N[0] - 1; ++tmp_k3_x) {
						
						// Get k3_x
						k3_x = tmp_k3_x - (int) (sys_vars->N[0] / 2) + 1;

						for (int tmp_k3_y = 0; tmp_k3_y <= sys_vars->N[1] - 1; ++tmp_k3_y) {
							
							// Get k3_y
							k3_y = tmp_k3_y - (int) (sys_vars->N[1] / 2) + 1;

							// Get polar coords for the k wavevector
							k3_sqr       = (double) (k3_x * k3_x + k3_y * k3_y);
							k3_angle     = atan2((double)k3_x, (double)k3_y);
							k3_angle_neg = atan2((double)-k3_x, (double)-k3_y);
							
							if ( (k3_sqr > sys_vars->kmax_C_sqr && k3_sqr <= sys_vars->kmax_sqr) 
								 && ((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr)) ) {  

								// Loop through the k1 wavevector
								for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
									
									// Get k1_x
									k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

									for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
										
										// Get k1_y
										k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

										// Get polar coords for k1
										k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
										k1_angle     = atan2((double) k1_x, (double) k1_y);
										k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

										if( ((k1_sqr > sys_vars->kmin_sqr && k1_sqr <= sys_vars->kmax_C_sqr) 
											&& ((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr)) 
											&& ((k1_angle >= C_theta_k1_lwr && k1_angle < C_theta_k1_upr) || (k1_angle_neg >= C_theta_k1_lwr && k1_angle_neg < C_theta_k1_upr)))
											|| 
											((k1_sqr > sys_vars->kmin_sqr && k1_sqr <= sys_vars->kmax_sqr) 
											&& ((k1_angle >= C_theta_k1_lwr && k1_angle < C_theta_k1_upr) || (k1_angle_neg >= C_theta_k1_lwr && k1_angle_neg < C_theta_k1_upr)) 
											&& !((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr))) ) { 
											
											// Find the k2 wavevector
											k2_x = k3_x - k1_x;
											k2_y = k3_y - k1_y;
											
											// Get polar coords for k2
											k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
											k2_angle     = atan2((double)k2_x, (double) k2_y);
											k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

											if ( (k2_sqr > sys_vars->kmin_sqr && k2_sqr <= sys_vars->kmax_sqr) 
												&& !((k2_sqr > sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_sqr) 
												&& ((k2_angle >= C_theta_k3_lwr && k2_angle < C_theta_k3_upr) || (k2_angle_neg >= C_theta_k3_lwr && k2_angle_neg < C_theta_k3_upr))) ) {

												// Add k1 vector
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K1_X][nn] = k1_x;
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K1_Y][nn] = k1_y;
												// Add the k2 vector
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K2_X][nn] = k2_x;
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K2_Y][nn] = k2_y;
												// Add the k3 vector
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K3_X][nn] = k3_x;
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K3_Y][nn] = k3_y;
												// Indicate which flux term this data is in
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][FLUX_TERM][nn] = POS_FLUX_TERM;
												// Indicate which type of contribution to the flux
												if ( (k1_sqr > sys_vars->kmin_sqr && k1_sqr <= sys_vars->kmax_C_sqr) 
													&& ((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr)) 
													&& ((k1_angle >= C_theta_k1_lwr && k1_angle < C_theta_k1_upr) || (k1_angle_neg >= C_theta_k1_lwr && k1_angle_neg < C_theta_k1_upr)) ) {
													// 1d contribution
													proc_data->phase_sync_wave_vecs[a][k1_sec_indx][CONTRIB_TYPE][nn] = CONTRIB_1D;

													// Update count of 1d terms
													total_1d_terms_per_sec++;
												}
												else if ( (k1_sqr > sys_vars->kmin_sqr && k1_sqr <= sys_vars->kmax_sqr) 
													&& ((k1_angle >= C_theta_k1_lwr && k1_angle < C_theta_k1_upr) || (k1_angle_neg >= C_theta_k1_lwr && k1_angle_neg < C_theta_k1_upr)) 
													&& !((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr)) ) {
													// 2d contribution
													proc_data->phase_sync_wave_vecs[a][k1_sec_indx][CONTRIB_TYPE][nn] = CONTRIB_2D;

													// Update count for 2d terms
													total_2d_terms_per_sec++;
												}

												// Increment
												nn++;
												if (nn > proc_data->num_wave_vecs[a][k1_sec_indx]) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- The number of triads in ["CYAN"%s"RESET"] exceding the allocated number of ["CYAN"%d"RESET"] terms -- Need to allocate more memory!!\n-->> Exiting!!!\n", "Wavevector Triad Search", proc_data->num_wave_vecs[a][k1_sec_indx]);
													exit(1);
												}
											}
										}
									}
								}
							}
							else if ( ((k3_sqr > sys_vars->kmin_sqr && k3_sqr <= sys_vars->kmax_sqr) && 
								((k3_angle >= C_theta_k1_lwr && k3_angle < C_theta_k1_upr) || (k3_angle_neg >= C_theta_k1_lwr && k3_angle_neg < C_theta_k1_upr)) 
								&& !((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr))) 
								|| 
								((k3_sqr > sys_vars->kmin_sqr && k3_sqr <= sys_vars->kmax_C_sqr) 
								&& ((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr)) 
								&& ((k3_angle >= C_theta_k1_lwr && k3_angle < C_theta_k1_upr) || (k3_angle_neg >= C_theta_k1_lwr && k3_angle_neg < C_theta_k1_upr))) ) { 

								// Loop through the k1 wavevector
								for (int tmp_k1_x = 0; tmp_k1_x <= sys_vars->N[0] - 1; ++tmp_k1_x) {
									
									// Get k1_x
									k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

									for (int tmp_k1_y = 0; tmp_k1_y <= sys_vars->N[1] - 1; ++tmp_k1_y) {
										
										// Get k1_y
										k1_y = tmp_k1_y - (int) (sys_vars->N[1] / 2) + 1;

										// Get polar coords for k1
										k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
										k1_angle     = atan2((double) k1_x, (double) k1_y);
										k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

										if( (k1_sqr > sys_vars->kmax_C_sqr && k1_sqr <= sys_vars->kmax_sqr) 
											&& ((k1_angle >= C_theta_k3_lwr && k1_angle < C_theta_k3_upr) || (k1_angle_neg >= C_theta_k3_lwr && k1_angle_neg < C_theta_k3_upr)) ) { 
											
											// Find the k2 wavevector
											k2_x = k3_x - k1_x;
											k2_y = k3_y - k1_y;
											
											// Get polar coords for k2
											k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
											k2_angle     = atan2((double)k2_x, (double) k2_y);
											k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

											if ( (k2_sqr > sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_sqr) 
												&& ((k2_angle >= C_theta_k3_lwr && k2_angle < C_theta_k3_upr) || (k2_angle_neg >= C_theta_k3_lwr && k2_angle_neg < C_theta_k3_upr)) ) {
												
												// Add k1 vector
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K1_X][nn] = k1_x;
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K1_Y][nn] = k1_y;
												// Add the k2 vector
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K2_X][nn] = k2_x;
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K2_Y][nn] = k2_y;
												// Add the k3 vector
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K3_X][nn] = k3_x;
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][K3_Y][nn] = k3_y;
												// Indicate which flux term this data is in
												proc_data->phase_sync_wave_vecs[a][k1_sec_indx][FLUX_TERM][nn] = NEG_FLUX_TERM;
												// Indicate which type of contribution to the flux
												if ( (k3_sqr > sys_vars->kmin_sqr && k3_sqr <= sys_vars->kmax_C_sqr) 
													&& ((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr)) 
													&& ((k3_angle >= C_theta_k1_lwr && k3_angle < C_theta_k1_upr) || (k3_angle_neg >= C_theta_k1_lwr && k3_angle_neg < C_theta_k1_upr)) ) {
													// 1d contribution
													proc_data->phase_sync_wave_vecs[a][k1_sec_indx][CONTRIB_TYPE][nn] = CONTRIB_1D;

													// Update count of 1d terms
													total_1d_terms_per_sec++;
												}
												else if ( ((k3_sqr > sys_vars->kmin_sqr && k3_sqr <= sys_vars->kmax_sqr) 
													&& ((k3_angle >= C_theta_k1_lwr && k3_angle < C_theta_k1_upr) || (k3_angle_neg >= C_theta_k1_lwr && k3_angle_neg < C_theta_k1_upr)) 
													&& !((k3_angle >= C_theta_k3_lwr && k3_angle < C_theta_k3_upr) || (k3_angle_neg >= C_theta_k3_lwr && k3_angle_neg < C_theta_k3_upr))) ) {
													// 2d contribution
													proc_data->phase_sync_wave_vecs[a][k1_sec_indx][CONTRIB_TYPE][nn] = CONTRIB_2D;

													// Update count for 2d terms
													total_2d_terms_per_sec++;
												}

												// Increment
												nn++;
												if (nn > proc_data->num_wave_vecs[a][k1_sec_indx]) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- The number of triads in ["CYAN"%s"RESET"] exceding the allocated number of ["CYAN"%d"RESET"] terms -- Need to allocate more memory!!\n-->> Exiting!!!\n", "Wavevector Triad Search", proc_data->num_wave_vecs[a][k1_sec_indx]);
													exit(1);
												}
											}
										}
									}
								}
							}
						}
					}

					// Record the number of triad wavevectors
					total_terms += nn;
					total_terms_per_sec += nn;
				}

				// Write Update to Screen 
				double search_end = omp_get_wtime();
				printf("Sector: %d/%d\tNum Terms: %ld (Tot) -- %ld (1D) -- %ld (2D)\tTime: %g(s)\n", a + 1, sys_vars->num_k3_sectors, total_terms_per_sec, total_1d_terms_per_sec, total_2d_terms_per_sec, (search_end - search_begin));
				fprintf(fptr, "Sector: %d/%d\tNum Terms: %ld (Tot) -- %ld (1D) -- %ld (2D)\tTime: %g(s)\n", a + 1, sys_vars->num_k3_sectors, total_terms_per_sec, total_1d_terms_per_sec, total_2d_terms_per_sec, (search_end - search_begin));
			}


			// --------------------------------
			//  Write Wavevector Data To File
			// --------------------------------
			///------------------- Write Phase Sync Wavevector Data to File for Future Use
			static const hsize_t Dims2D = 2;
			hsize_t dset_dims_2d[Dims2D];   

			//-------------------- Create wavector data file
			file_info->wave_vec_file_handle = H5Fcreate(file_info->wave_vec_data_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			if (file_info->wave_vec_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create wavevector file name: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->wave_vec_data_name);
				exit(1);
			}	

			//------------------- Write the number of wavevectors per sector
			int* tmp_num_wavevecs = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors);
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a){
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l){
					tmp_num_wavevecs[a * sys_vars->num_k1_sectors + l] = proc_data->num_wave_vecs[a][l];
				}
			}

			dset_dims_2d[0] = sys_vars->num_k3_sectors;
			dset_dims_2d[1] = sys_vars->num_k1_sectors;
			status = H5LTmake_dataset(file_info->wave_vec_file_handle, "NumWavevectors", Dims2D, dset_dims_2d, H5T_NATIVE_INT, tmp_num_wavevecs);
			if (status < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to wavevector data file!!\n-->> Exiting...\n", "NumWavevectors");
				exit(1);
			}

			// Write the wavevector data
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a){
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l){
					int* tmp_wave_vec_data = (int* )fftw_malloc(sizeof(int) * NUM_K_DATA * proc_data->num_wave_vecs[a][l]);
					for (int k = 0; k < NUM_K_DATA; ++k) {
						for (int n = 0; n < proc_data->num_wave_vecs[a][l]; ++n) {
							tmp_wave_vec_data[k * proc_data->num_wave_vecs[a][l] + n] = proc_data->phase_sync_wave_vecs[a][l][k][n];
						}
					}

					dset_dims_2d[0] = NUM_K_DATA;
					dset_dims_2d[1] = proc_data->num_wave_vecs[a][l];
					sprintf(dset_name, "WVData_Sector_%d_%d", a, l);
					status = H5LTmake_dataset(file_info->wave_vec_file_handle, dset_name, Dims2D, dset_dims_2d, H5T_NATIVE_INT, tmp_wave_vec_data);
					if (status < 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to wavevector data file!!\n-->> Exiting...\n", dset_name);
						exit(1);
					}

					fftw_free(tmp_wave_vec_data);
				}
			}

			// Free temporary memory
			fftw_free(tmp_num_wavevecs);

			//------------------- Close Wavector Data File
			status = H5Fclose(file_info->wave_vec_file_handle);
			if (status < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close wavevector data file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->wave_vec_data_name);
				exit(1);		
			}

			printf("\n["YELLOW"NOTE"RESET"] --- Saved wavevector data to file at ["CYAN"%s"RESET"]...\n", file_info->wave_vec_data_name);

			// End timer 
			double loop_end = omp_get_wtime();
			printf("\n["YELLOW"NOTE"RESET"] --- Total Triad Terms: %ld\n", total_terms);
			printf("["YELLOW"NOTE"RESET"] --- Total Time for sector search: %g(s)\n", (loop_end - loop_begin));


			// Close number of flux terms file
			fprintf(fptr, "\n--- Total Triad Terms: %ld\n", total_terms);
			fprintf(fptr, "--- Total Time for sector search: %g(s)\n", (loop_end - loop_begin));
			fclose(fptr);
		}

	}
}
/**
 * Frees memory and GSL objects allocated to perform the Phase sync computation
 */
void FreePhaseSyncObjects(void) {

	// --------------------------------
	//  Free Memory
	// --------------------------------
	fftw_free(proc_data->theta_k3);
	fftw_free(proc_data->theta_k1);
	fftw_free(proc_data->mid_angle_sum);
	fftw_free(proc_data->phase_angle);
	fftw_free(proc_data->phase_order);
	fftw_free(proc_data->phase_R);
	fftw_free(proc_data->phase_Phi);
	fftw_free(proc_data->k1_sector_angles);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		fftw_free(proc_data->enst_flux[i]);
		fftw_free(proc_data->triad_phase_order[i]);
		fftw_free(proc_data->triad_R[i]);
		fftw_free(proc_data->triad_Phi[i]);
		fftw_free(proc_data->num_triads[i]);
		fftw_free(proc_data->enst_flux_1d[i]);
		fftw_free(proc_data->triad_phase_order_1d[i]);
		fftw_free(proc_data->phase_order_C_theta_triads[i]);
		fftw_free(proc_data->phase_order_C_theta_triads_1d[i]);
		fftw_free(proc_data->phase_order_C_theta_triads_unidirec[i]);
		fftw_free(proc_data->phase_order_C_theta_triads_unidirec_1d[i]);
		fftw_free(proc_data->triad_R_1d[i]);
		fftw_free(proc_data->triad_Phi_1d[i]);
		fftw_free(proc_data->num_triads_1d[i]);
		for (int l = 0; l < sys_vars->num_k3_sectors; ++l) {
			fftw_free(proc_data->enst_flux_2d[i][l]);
			fftw_free(proc_data->triad_phase_order_2d[i][l]);
			fftw_free(proc_data->triad_R_2d[i][l]);
			fftw_free(proc_data->triad_Phi_2d[i][l]);
			fftw_free(proc_data->num_triads_2d[i][l]);
			fftw_free(proc_data->phase_order_C_theta_triads_2d[i][l]);
			fftw_free(proc_data->phase_order_C_theta_triads_unidirec_2d[i][l]);
			fftw_free(proc_data->phase_order_norm_const[0][i][l]);
			fftw_free(proc_data->phase_order_norm_const[1][i][l]);
		}
		fftw_free(proc_data->phase_order_C_theta_triads_2d[i]);
		fftw_free(proc_data->phase_order_C_theta_triads_unidirec_2d[i]);
		fftw_free(proc_data->phase_order_norm_const[0][i]);
		fftw_free(proc_data->phase_order_norm_const[1][i]);
		fftw_free(proc_data->enst_flux_2d[i]);
		fftw_free(proc_data->triad_phase_order_2d[i]);
		fftw_free(proc_data->triad_R_2d[i]);
		fftw_free(proc_data->triad_Phi_2d[i]);
		fftw_free(proc_data->num_triads_2d[i]);
	}
	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l){
			if (proc_data->num_wave_vecs[a][l] != 0) {
				for (int n = 0; n < NUM_K_DATA; ++n) {
					fftw_free(proc_data->phase_sync_wave_vecs[a][l][n]);
				}	
			}
			fftw_free(proc_data->phase_sync_wave_vecs[a][l]);
		}
		fftw_free(proc_data->phase_sync_wave_vecs[a]);
	}
	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
		fftw_free(proc_data->num_wave_vecs[a]);
	}
	#if defined(__PHASE_SYNC)
	for (int n = 0; n < 6; ++n) {
		fftw_free(proc_data->phase_sync_wave_vecs_test[n]);
	}	
	#endif

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	#if defined (__SEC_PHASE_SYNC_FLUX_STATS) || defined (__SEC_PHASE_SYNC_STATS_IN_TIME) || defined (__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined (__SEC_PHASE_SYNC_STATS_IN_TIME_2D) || defined (__SEC_PHASE_SYNC_STATS_IN_TIME_1D) || defined(__SEC_PHASE_SYNC_COND_STATS)
	for (int triad_class = 0; triad_class < NUM_TRIAD_CLASS; ++triad_class) {
		for (int triad_type = 0; triad_type < NUM_TRIAD_TYPES + 1; ++triad_type) {
			#if defined(__SEC_PHASE_SYNC_COND_STATS)
			if (triad_class == 0) {
				fftw_free(proc_data->max_sync[triad_type]);
			}
			#endif
			#if defined (__SEC_PHASE_SYNC_FLUX_STATS)
			if (triad_type < 5) {
				gsl_histogram2d_free(proc_data->triads_wghtd_2d_pdf_t[triad_class][triad_type]);
				fftw_free(proc_data->max_enst_flux[triad_class][triad_type]);
			}
			#endif
			#if defined (__SEC_PHASE_SYNC_STATS_IN_TIME)
			gsl_histogram_free(proc_data->triads_all_pdf_t[triad_class][triad_type]);
			gsl_histogram_free(proc_data->triads_wghtd_all_pdf_t[triad_class][triad_type]);
			#endif
			#if defined (__SEC_PHASE_SYNC_STATS_IN_TIME_ALL) || defined (__SEC_PHASE_SYNC_STATS_IN_TIME_2D) || defined (__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
			for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
				gsl_histogram_free(proc_data->triads_sect_all_pdf_t[triad_class][triad_type][a]);
				gsl_histogram_free(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][triad_type][a]);
        	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
				gsl_histogram_free(proc_data->triads_sect_1d_pdf_t[triad_class][triad_type][a]);
				gsl_histogram_free(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][triad_type][a]);
				#endif
	       	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
				for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
					gsl_histogram_free(proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a][l]);
					gsl_histogram_free(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type][a][l]);
				}
				fftw_free(proc_data->triads_sect_2d_pdf_t[triad_class][triad_type][a]);
				fftw_free(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type][a]);
				#endif
			}
			fftw_free(proc_data->triads_sect_all_pdf_t[triad_class][triad_type]);
			fftw_free(proc_data->triads_sect_wghtd_all_pdf_t[triad_class][triad_type]);
       	    #if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_1D)
			fftw_free(proc_data->triads_sect_1d_pdf_t[triad_class][triad_type]);
			fftw_free(proc_data->triads_sect_wghtd_1d_pdf_t[triad_class][triad_type]);
			#endif
			#if defined(__SEC_PHASE_SYNC_STATS_IN_TIME_2D)
			fftw_free(proc_data->triads_sect_2d_pdf_t[triad_class][triad_type]);
			fftw_free(proc_data->triads_sect_wghtd_2d_pdf_t[triad_class][triad_type]);
			#endif
			#endif
		}
	}
	#endif
	#if defined (__SEC_PHASE_SYNC_COND_STATS)
	for (int i = 0; i < NUM_COND_TYPES; ++i) {
		for (int j = 0; j < INCR_TYPES; ++j) {
			for (int k = 0; k < NUM_INCR; ++k) {
				for (int l = 0; l < NUM_THRESH_TYPES; ++l) {
					gsl_histogram_free(proc_data->cond_t_w_incr_hist[i][j][k][l]);
					gsl_rstat_free(proc_data->cond_t_w_incr_stats[i][j][k][l]);
				}
			}
		}
	}
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		gsl_histogram2d_free(proc_data->joint_sync_enst_flux_hist[i]);
	}
	#endif
	#if defined(__SEC_PHASE_SYNC_STATS)
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
		gsl_histogram_free(proc_data->phase_sect_pdf[i]);
		gsl_histogram_free(proc_data->phase_sect_pdf_t[i]);
		gsl_histogram_free(proc_data->phase_sect_wghtd_pdf_t[i]);
		gsl_histogram_free(proc_data->triad_R_2d_pdf[l]);
		gsl_histogram_free(proc_data->triad_Phi_2d_pdf[l]);
		gsl_histogram_free(proc_data->enst_flux_2d_pdf[l]);
		gsl_rstat_free(proc_data->triad_R_2d_stats[i]);
		gsl_rstat_free(proc_data->triad_Phi_2d_stats[i]);
		gsl_rstat_free(proc_data->enst_flux_2d_stats[i]);
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			gsl_histogram_free(proc_data->triad_sect_pdf[j][i]);
			gsl_histogram_free(proc_data->triad_sect_pdf_t[j][i]);
			gsl_histogram_free(proc_data->triad_sect_wghtd_pdf_t[j][i]);
		}	
	}
	gsl_rstat_free(proc_data->triad_R_2d_stats);
	gsl_rstat_free(proc_data->triad_Phi_2d_stats);
	gsl_rstat_free(proc_data->enst_flux_2d_stats);
	gsl_histogram_free(proc_data->triad_R_2d_pdf);
	gsl_histogram_free(proc_data->triad_Phi_2d_pdf);
	gsl_histogram_free(proc_data->enst_flux_2d_pdf);
	#endif
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
