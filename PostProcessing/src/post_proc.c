/**
* @file utils.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the post processsing functions for the pseudospectral solver
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
 * Function to compute the real space stats both for the current snapshot and throughout the simulaion
 * @param s The index of the current snapshot
 */
void RealSpaceStats(int s) {

	// Initialize variables
	int tmp;
	int indx;
	int gsl_status;
	double w_max, w_min;
	double u_max, u_min;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];

	// --------------------------------
	// Get Histogram Limits
	// --------------------------------
	// Get min and max data for histogram limits
	w_max = 0.0;
	w_min = 1e8;
	gsl_stats_minmax(&w_min, &w_max, run_data->w, 1, Nx * Ny);

	u_max = 0.0; 
	u_min = 1e8;
	gsl_stats_minmax(&u_min, &u_max, run_data->u, 1, Nx * Ny * SYS_DIM);
	if (fabs(u_min) > u_max) {
		u_max = fabs(u_min);
	}

	// --------------------------------
	// Set Histogram Bin Ranges
	// --------------------------------
	// Set histogram ranges for the current snapshot
	gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_pdf, w_min - 0.5, w_max + 0.5);
	if (gsl_status != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
		exit(1);
	}
	gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_pdf, 0.0 - 0.1, u_max + 0.5);
	if (gsl_status != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
		exit(1);
	}

	// --------------------------------
	// Update Histogram Counts
	// --------------------------------
	// Update histograms with the data from the current snapshot
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Add current values to appropriate bins
			gsl_status = gsl_histogram_increment(stats_data->w_pdf, run_data->w[indx]);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_pdf, fabs(run_data->u[SYS_DIM * indx + 0]));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_pdf, fabs(run_data->u[SYS_DIM * indx + 1]));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
				exit(1);
			}
		}
	}
}
/**
 * Function used to fill the full field arrays
 */
void FullFieldData() {

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
	 				if (sqrt(k_sqr) < sys_vars->kmax) {
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
	 						// if (run_data->k[0][i] > 0 && run_data-> k[1][j] > 0) {
	 						// 	proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 0.0;	 							
	 						// }
	 						// if (run_data->k[0][i] < 0 && run_data-> k[1][j] > 0) {
	 						// 	proc_data->phases[tmp1 + sys_vars->kmax - 1 - run_data->k[1][j]] = 2.0 * M_PI;	 							
	 						// }
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
	 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
	 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
	 					}
	 					else {	
	 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = -50.0;
	 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = -50.0;
	 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
	 						proc_data->amps[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = -50.0;
	 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
	 						proc_data->enrg[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = 0.0;
	 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
	 						proc_data->enst[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = 0.0;
	 					}
	 				}
				}	
			}						
		}
	}
}
/**
 * Function to compute the phase synchroniztion data sector by sector in wavenumber space for the current snaphsot
 * @param s The index of the current snapshot
 */
void SectorPhaseOrder(int s) {

	// Initialize variables
	double r;
	int indx;
	double angle;
	int gsl_status;
	double flux_wght;
	int num_phases;
	int tmp, tmp1;
	double flux_pre_fac;
	int k_x, k1_x, k2_x, k2_y;
	int tmp_k, tmp_k1, tmp_k2;
	double phase, triad_phase, gen_triad_phase;
	double k_sqr, k_angle, k1_sqr, k1_angle, k2_sqr, k2_angle, k2_angle_neg;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;


	for (int i = 0; i < sys_vars->N[0]; ++i) {
		if (abs(run_data->k[0][i]) < sys_vars->kmax) {
			for (int j = 0; j < sys_vars->N[1] / 2 + 1; ++j) {
				if (abs(run_data->k[1][j]) < sys_vars->kmax) {
					k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
					angle = atan2((double) run_data->k[0][i], (double) run_data->k[1][j]); 
					if (k_sqr > 0 && k_sqr <= sys_vars->kmax_sqr && angle >= proc_data->theta[8] && angle < proc_data->theta[9]) {
						proc_data->phases[(sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1) + (sys_vars->kmax - 1 + run_data->k[1][j])] = 0.0;
						proc_data->phases[(sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1) + (sys_vars->kmax - 1 - run_data->k[1][j])] = 0.0;
					}
				}
			}
		}
	}

	printf("t1: %1.16lfa: %1.16lf\tt2: %1.16lf\n", proc_data->theta[8], angle, proc_data->theta[9]);


	// --------------------------------
	// Phase Sync Sector by Sector
	// --------------------------------
	// Loop through each sector
	// #pragma omp parallel for shared(run_data->k, proc_data->phases, proc_data->theta, proc_data->phase_order, proc_data->phase_sect_pdf, proc_data->phase_sect_pdf_t, proc_data->triad_phase_order, proc_data->ord_triad_phase_order) private(num_phases, num_ordered_phases, r, angle, phase)
	for (int a = 0; a < sys_vars->num_sect; ++a) {
	
		// --------------------------------
		// Individual Phases
		// --------------------------------
		// Initialize phase counter
		num_phases = 0;

		// Loop through the phases
		for (int i = 0; i < Nx; ++i) {
			tmp  = i * Ny_Fourier;	
			tmp1 = (sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1); // correct indexing for the phases -> for kx > 0 use -kx
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;
	
				// Compute the polar coords for the current mode
 				r     = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
 				angle = proc_data->phase_angle[tmp + j];
				
				// Update the phase order parameter for the current sector
 				if ((r < sys_vars->kmax_sqr) && (angle >= proc_data->theta[a] && angle < proc_data->theta[a + 1])) {
					// Pre-compute phase data
					phase = proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]];

					// Update the phase order sum
					proc_data->phase_order[a] += cexp(I * phase);
 					
 					// Update counter
 					num_phases++;

					// Update individual phases PDFs for this sector
					gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf[a], phase - M_PI);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF", a, s);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf_t[a], phase - M_PI);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s);
						exit(1);
					}
					gsl_status = gsl_histogram_accumulate(proc_data->phase_sect_wghtd_pdf_t[a], phase - M_PI, proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s);
						exit(1);
					}
 				}
	 		}	
		}

		///-------------------- Record the individual phase data
		// Normalize the phase order parameter
		proc_data->phase_order[a] /= num_phases;

		// Record the phase sync and average phase
		proc_data->phase_R[a]   = cabs(proc_data->phase_order[a]);
		proc_data->phase_Phi[a] = carg(proc_data->phase_order[a]);

		// --------------------------------
		// Triad Phases
		// --------------------------------
		// Initialize counters for each triad type
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			// Initialize the number of triads for each type
			proc_data->num_triads[i][a] = 0;
		}

		// Initialize the flux for the current sector
		proc_data->enst_flux[a] = 0.0;

		// Loop through the k wavevector (k is the k3 wavevector)
		for (int tmp_k_x = 0; tmp_k_x <= 2 * sys_vars->kmax - 1; ++tmp_k_x) {
			// Get k_x
			k_x = tmp_k_x - sys_vars->kmax + 1;

			for (int k_y = 0; k_y <= sys_vars->kmax; ++k_y) {
				// Get polar coords for the k wavevector
				k_sqr   = (double) (k_x * k_x + k_y * k_y);
				k_angle = proc_data->k_angle[tmp_k_x * (sys_vars->kmax + 1) + k_y]; 

				// Check if the k wavevector is in the current sector
				if ((k_sqr > 0 && k_sqr < sys_vars->kmax_sqr) && (k_angle >= proc_data->theta[a] && k_angle < proc_data->theta[a + 1])) {

					// Loop through the k1 wavevector
					for (int tmp_k1_x = 0; tmp_k1_x <= 2 * sys_vars->kmax - 1; ++tmp_k1_x) {
						// Get k1_x
						k1_x = tmp_k1_x - sys_vars->kmax + 1;

						for (int k1_y = 0; k1_y <= sys_vars->kmax; ++k1_y) {
							// Get polar coords for k1
							k1_sqr   = (double) (k1_x * k1_x + k1_y * k1_y);
							k1_angle = proc_data->k1_angle[tmp_k1_x * (sys_vars->kmax + 1) + k1_y];

							// Check if k1 wavevector is in the current sector
							if((k1_sqr > 0 && k1_sqr < sys_vars->kmax_sqr) && (k1_angle >= proc_data->theta[a] && k1_angle < proc_data->theta[a + 1])) {
								
								// Find the k2 wavevector
								k2_x = k_x - k1_x;
								k2_y = k_y - k1_y;
								
								// Get polar coords for k2
								k2_sqr   = (double) (k2_x * k2_x + k2_y * k2_y);
								k2_angle = proc_data->k2_angle[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y];

								if (k2_y < 0 || (k2_y == 0 && k2_x > 0)) {
									// Get the angle of the negative wavevector k2 i.e., -k2 --> for checking the case if k2 is in the negative sector
									k2_angle_neg = proc_data->k2_angle_neg[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y];
								}

								// Check if k2 is in the current positive sector or if k2 is in the negative sector
								if ((k2_sqr > 0 && k2_sqr < sys_vars->kmax_sqr) &&  ((k2_angle >= proc_data->theta[a] && k2_angle < proc_data->theta[a + 1] && k2_sqr > k1_sqr) || ((k2_y < 0 || (k2_y == 0 && k2_x > 0)) && k2_angle_neg >= proc_data->theta[a] && k2_angle_neg < proc_data->theta[a + 1]))) {
									// Get correct phase index -> recall that to access kx > 0, use -kx
									tmp_k  = (sys_vars->kmax - 1 - k_x) * (2 * sys_vars->kmax - 1);
									tmp_k1 = (sys_vars->kmax - 1 - k1_x) * (2 * sys_vars->kmax - 1);
									tmp_k2 = (sys_vars->kmax - 1 - k2_x) * (2 * sys_vars->kmax - 1);

									// Get the wavevector prefactor
									flux_pre_fac = (double)(k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

									// Get the weighting (modulus) of this term to the contribution to the flux
									flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax - 1 + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax - 1 + k2_y] * proc_data->amps[tmp_k + sys_vars->kmax - 1 + k_y]);

									// Get the triad phase and adjust to with the orientation of the phase - the generalized phase
									triad_phase = proc_data->phases[tmp_k1 + sys_vars->kmax - 1 + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax - 1 + k2_y] - proc_data->phases[tmp_k + sys_vars->kmax - 1 + k_y];
									// triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
									if (k_sqr > k1_sqr && k_sqr > k2_sqr) {
										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
									}
									else if (k_sqr < k1_sqr && k_sqr < k2_sqr) {
										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
									}
									else {
										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
									}

							
									///------------------ Determine which flux contribution we are dealing with -> the postive case (when k1 & k2 in C^' and k3 in C) or the negative case (when k3 in C^' and k1 and k2 in C)
									// The postive case
									if (k_sqr > k1_sqr && k_sqr > k2_sqr && flux_wght > 1e-5) {

										// Update the combined triad phase order parameter with the appropriate contribution
										proc_data->triad_phase_order[0][a] += cexp(I * gen_triad_phase) * GSL_SIGN(flux_pre_fac);
										
										// Update the triad counter for the combined triad type
										proc_data->num_triads[0][a]++;

										// Update the flux contribution
										proc_data->enst_flux[a] += flux_wght * cos(triad_phase);

										// printf("triad: %1.16lf\t gen: %1.16lf\t %1.16lf\t check = %d\t flux wght: %1.16lf, pre: %1.16lf\t k1_sqr: %1.0lf, k2_sqr: %1.0lf\t k1_x: %d, k1_y: %d \tk2_x: %d, k2_y: %d\n", triad_phase, gen_triad_phase, -M_PI, check, flux_wght, flux_pre_fac, k1_sqr, k2_sqr, k1_x, k1_y, k2_x, k2_y);

										// ------ Update the PDFs of the combined triads
										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s);
											exit(1);
										}
										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s);
											exit(1);
										}

										// Update the triad types and their PDFs
										if (GSL_SIGN(flux_pre_fac) < 0) {
											// TYPE 1 ---> Positive flux term and when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
											proc_data->triad_phase_order[1][a] += cexp(I * gen_triad_phase);
											proc_data->num_triads[1][a]++;

											// Update the PDFs
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[1][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[1][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[1][a], gen_triad_phase, fabs(flux_wght));
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s);
												exit(1);
											}
										}
										else if (GSL_SIGN(flux_pre_fac) > 0) {
											// TYPE 2 ---> Positive flux term and when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
											proc_data->triad_phase_order[2][a] += cexp(I * gen_triad_phase);
											proc_data->num_triads[2][a]++;

											// Update the PDFs
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[2][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Type 2 PDF", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[2][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 2 In Time", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[2][a], gen_triad_phase, fabs(flux_wght));
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 2 In Time", a, s);
												exit(1);
											}
										}

									}
									// The negative case
									else if (k_sqr < k1_sqr && k_sqr < k2_sqr && flux_wght > 1e-5) {

										// Update the combined triad phase order parameter with the appropriate contribution
										proc_data->triad_phase_order[0][a] += - 1.0 * cexp(I * gen_triad_phase) * GSL_SIGN(flux_pre_fac);

										// Update the triad counter for the combined triad type
										proc_data->num_triads[0][a]++;

										// Update the flux contribution
										proc_data->enst_flux[a] += -flux_wght * cos(triad_phase);

										// printf("triad: %1.16lf\t gen: %1.16lf\t %1.16lf\t check = %d\t flux wght: %1.16lf, pre: %1.16lf\t k1_sqr: %1.0lf, k2_sqr: %1.0lf\t k1_x: %d, k1_y: %d \tk2_x: %d, k2_y: %d\n", triad_phase, gen_triad_phase, -M_PI, check, flux_wght, flux_pre_fac, k1_sqr, k2_sqr, k1_x, k1_y, k2_x, k2_y);

										// ------ Update the PDFs of the combined triads
										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s);
											exit(1);
										}
										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s);
											exit(1);
										}
										gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(-flux_wght));
										if (gsl_status != 0) {
											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s);
											exit(1);
										}

										// Update the triad types and their PDFs
										if (GSL_SIGN(flux_pre_fac) < 0) {
											// TYPE 3 ---> Negative flux term and when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
											proc_data->triad_phase_order[3][a] += cexp(I * gen_triad_phase);
											proc_data->num_triads[3][a]++;

											// Update the PDFs
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[3][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[3][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3 In Time", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[3][a], gen_triad_phase, fabs(-flux_wght));
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 3 In Time", a, s);
												exit(1);
											}
										}
										else if (GSL_SIGN(flux_pre_fac) > 0) {
											// TYPE 4 ---> Negative flux term and when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
											proc_data->triad_phase_order[4][a] += cexp(I * gen_triad_phase);
											proc_data->num_triads[4][a]++;

											// Update the PDFs
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[4][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[4][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4 In Time", a, s);
												exit(1);
											}
											gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[4][a], gen_triad_phase, fabs(-flux_wght)); // phas
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 4 In Time", a, s);
												exit(1);
											}
										}
									}
									else {
										// When k1 and k2 = k3 the contribution to the flux is zero
										continue;
									}
								}									
							}
						}
					}
				}
			}				
		}

		//------------------- Record the data for the triads
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			// Normalize the phase order parameters
			proc_data->triad_phase_order[i][a] /= proc_data->num_triads[i][a];
			
			// Record the phase syncs and average phases
			proc_data->triad_R[i][a]   = cabs(proc_data->triad_phase_order[i][a]);
			proc_data->triad_Phi[i][a] = carg(proc_data->triad_phase_order[i][a]); 
			// printf("a: %d type: %d Num: %d\t triad_phase_order: %lf %lf I\t\t R: %lf, Phi %lf\n", a, i, num_triads[i], creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->triad_R[i][a], proc_data->triad_Phi[i][a]);
			// printf("Num Triads Type %d: %d\n", i, num_triads[i]);
		}
		printf("a: %d\tR: %1.5lf\ttheta: %1.16lf\t||\tR: %1.5lf\n", a,  proc_data->triad_R[0][a], proc_data->theta[a], proc_data->phase_R[a]);

		// printf("a: %d Num: %d\t triad_phase_order: %lf %lf I\t Num: %d\t triad_phase_order: %lf %lf I \t Num: %d\t triad_phase_order: %lf %lf I\n", a, num_triads[0], creal(proc_data->triad_phase_order[0][a]), cimag(proc_data->triad_phase_order[0][a]), num_triads[1], creal(proc_data->triad_phase_order[1][a]), cimag(proc_data->triad_phase_order[1][a]), num_triads[2], creal(proc_data->triad_phase_order[2][a]), cimag(proc_data->triad_phase_order[2][a]));
		// printf("a: %d | Num: %d R0: %lf Phi0: %lf |\t Num: %d R1: %lf Phi1: %lf |\t Num: %d R2: %lf Phi2: %lf\n", a, num_triads[0], proc_data->triad_R[0][a], proc_data->triad_Phi[0][a], num_triads[1], proc_data->triad_R[1][a], proc_data->triad_Phi[1][a], num_triads[2], proc_data->triad_R[2][a], proc_data->triad_Phi[2][a]);	
	}

	// --------------------------------
	// Reset Phase Order Parameters
	// --------------------------------
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		proc_data->phase_order[a] = 0.0 + 0.0 * I;
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			proc_data->triad_phase_order[i][a] = 0.0 + 0.0 * I;
		}	
	}
	printf("\n");
}
/**
 * Wrapper function used to allocate the nescessary data objects
 * @param N Array containing the dimensions of the system
 */
void AllocateMemory(const long int* N) {

	// Initialize variables
	int gsl_status;
	int tmp1, tmp2, tmp3;
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Compute maximum wavenumber
	sys_vars->kmax = (int) (Nx / 3);	

	// --------------------------------
	//  Allocate Field Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}

	// Allocate the Fourier stream funciton
	run_data->psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->psi_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Stream Function");
		exit(1);
	}

	#ifdef __REAL_STATS
	// Allocate current Fourier vorticity
	run_data->w = (double* )fftw_malloc(sizeof(double) * Nx * Ny);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity");
		exit(1);
	}

	// Allocate current Fourier vorticity
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}

	// Allocate current Fourier vorticity
	run_data->u = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}
	#endif
	
	#ifdef __FULL_FIELD
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
	#endif

	#ifdef __SPECTRA
	// Get the size of the spectra
	sys_vars->n_spec = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0)) + 1;

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

	// Allocate memory for the alternative enstrophy spectrum
	proc_data->enst_alt = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_alt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Enstrophy Spectrum");
		exit(1);
	}
    
    // Allocate memory for the alternative enstrophy spectrum
	proc_data->enrg_alt = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enrg_alt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Energy Spectrum");
		exit(1);
	}

	// Initialize arrays
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		proc_data->enst_spec[i] = 0.0;
        proc_data->enrg_spec[i] = 0.0;
        proc_data->enst_alt[i]  = 0.0;
        proc_data->enrg_alt[i]  = 0.0;
	}
	#endif

	// --------------------------------	
	//  Allocate Phase Sync Data
	// --------------------------------
	#ifdef __SEC_PHASE_SYNC
	// Get kmax ^2
	sys_vars->kmax_sqr = pow(sys_vars->kmax, 2.0);

	///--------------- Sector Angles
	// Allocate the array of sector angles
	proc_data->theta = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect + 1));
	if (proc_data->theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Sector Angles");
		exit(1);
	}

	///--------------- Enstrophy Flux
	proc_data->enst_flux = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect));
	if (proc_data->enst_flux == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
		exit(1);
	}	

	///--------------- Number of Triads
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		// Allocate memory for the phase order parameter for the triad phases
		proc_data->num_triads[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_sect);
		if (proc_data->num_triads[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}
	}
	

	///--------------- Precomputed wavevector arctangents
	// Allocate memory for the arctangent arrays
	proc_data->k_angle = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax) * (sys_vars->kmax + 1));
	if (proc_data->k_angle == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of k3");
		exit(1);
	}
	proc_data->k1_angle = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax) * (sys_vars->kmax + 1));
	if (proc_data->k1_angle == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of k1");
		exit(1);
	}
	proc_data->k2_angle = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax) * (sys_vars->kmax + 1) * (2 * sys_vars->kmax) * (sys_vars->kmax + 1));
	if (proc_data->k2_angle == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of k2");
		exit(1);
	}
	proc_data->k2_angle_neg = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax) * (sys_vars->kmax + 1) * (2 * sys_vars->kmax) * (sys_vars->kmax + 1));
	if (proc_data->k2_angle_neg == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of Negative k2");
		exit(1);
	}
	proc_data->phase_angle = (double* )fftw_malloc(sizeof(double) * Nx * Ny_Fourier);
	if (proc_data->phase_angle == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of Negative k2");
		exit(1);
	}

	// Fill the arrays with the pre-computed arctangents of the wavevectors
	int k_x, k1_x, k2_x, k2_y;
	for (int tmp_k_x = 0; tmp_k_x <= 2 * sys_vars->kmax - 1; ++tmp_k_x) {
		k_x = tmp_k_x - sys_vars->kmax + 1;
		for (int k_y = 0; k_y <= sys_vars->kmax; ++k_y) {
			// Fill the k3 array
			proc_data->k_angle[tmp_k_x * (sys_vars->kmax + 1) + k_y] = atan2((double) k_x, (double) k_y);

			// Loop over k1 to get k2
			for (int tmp_k1_x = 0; tmp_k1_x <= 2 * sys_vars->kmax - 1; ++tmp_k1_x) {
				k1_x = tmp_k1_x - sys_vars->kmax + 1;
				for (int k1_y = 0; k1_y <= sys_vars->kmax; ++k1_y) {	

					// Fill the k1 array
					proc_data->k1_angle[tmp_k1_x * (sys_vars->kmax + 1) + k1_y] = atan2((double) k1_x, (double) k1_y);				

					// Get k2
					k2_x = k_x - k1_x;
					k2_y = k_y - k1_y;

					// Fill the k2 arrays
					proc_data->k2_angle[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y]     = atan2((double) k2_x, (double) k2_y);
					proc_data->k2_angle_neg[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y] = atan2((double) -k2_x, (double) -k2_y);
				}
			}
		}
	}

	// Fill the array for the individual phases with the precomputed arctangents
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			proc_data->phase_angle[i * Ny_Fourier + j] = atan2((double) run_data->k[0][i], (double)run_data->k[1][j]);
		}
	}

	///--------------- Individual Phases
	// Allocate memory for the phase order parameter for the individual phases
	proc_data->phase_order = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
	if (proc_data->phase_order == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Phase Order Parameter");
		exit(1);
	}
	// Allocate the array of phase sync per sector for the individual phases
	proc_data->phase_R = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
	if (proc_data->phase_R == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Parameter");
		exit(1);
	}
	// Allocate the array of average phase per sector for the individual phases
	proc_data->phase_Phi = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
	if (proc_data->phase_Phi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Phase");
		exit(1);
	}

	///------------- Triad Phases
	// Allocate memory for each  of the triad types
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		// Allocate memory for the phase order parameter for the triad phases
		proc_data->triad_phase_order[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
		if (proc_data->triad_phase_order[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
		if (proc_data->triad_R[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
		if (proc_data->triad_Phi[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
			exit(1);
		}
	}
	
	///--------------- Phase Order Stats Objects
	// Allocate memory for the arrays stats objects
	proc_data->phase_sect_pdf         = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	proc_data->phase_sect_pdf_t       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	proc_data->phase_sect_wghtd_pdf_t = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		proc_data->triad_sect_pdf[i]         = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
		proc_data->triad_sect_pdf_t[i]       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
		proc_data->triad_sect_wghtd_pdf_t[i] = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	}
	

	// Allocate stats objects for each sector and set ranges
	for (int i = 0; i < sys_vars->num_sect; ++i) {
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
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_pdf[i], -M_PI - 0.05,  M_PI + 0.05);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases PDF");
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_pdf_t[i], -M_PI - 0.05,  M_PI + 0.05);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases PDF In Time");
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_wghtd_pdf_t[i], -M_PI - 0.05,  M_PI + 0.05);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases Weighted PDF In Time");
			exit(1);
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
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_pdf[j][i], -M_PI - 0.05,  M_PI + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_pdf_t[j][i], -M_PI - 0.05,  M_PI + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF In Time");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_wghtd_pdf_t[j][i], -M_PI - 0.05,  M_PI + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF In Time");
				exit(1);
			}
		}
	}
	
	// Initialize arrays
	double dtheta = M_PI / (double )sys_vars->num_sect;
	for (int i = 0; i < sys_vars->num_sect + 1; ++i) {
		proc_data->theta[i] = -M_PI / 2.0 + i * dtheta;
		if (i < sys_vars->num_sect){
			proc_data->phase_R[i]     = 0.0;
			proc_data->phase_Phi[i]   = 0.0;
			proc_data->enst_flux[i]   = 0.0;
			proc_data->phase_order[i] = 0.0 + 0.0 * I;
			for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
				proc_data->triad_R[j][i]     	   = 0.0;
				proc_data->triad_Phi[j][i]         = 0.0;
				proc_data->triad_phase_order[j][i] = 0.0 + 0.0 * I;
			}
		}
		printf("theta: %1.16lf\n", proc_data->theta[i]);
	}
	#endif

	// --------------------------------	
	//  Allocate Stats Data
	// --------------------------------
	#ifdef __REAL_STATS
	// Allocate vorticity histograms
	stats_data->w_pdf = gsl_histogram_alloc(N_BINS);
	if (stats_data->w_pdf == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Histogram");
		exit(1);
	}	

	// Allocate velocity histograms
	stats_data->u_pdf = gsl_histogram_alloc(N_BINS);
	if (stats_data->u_pdf == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity In Time Histogram");
		exit(1);
	}	
	#endif

	// --------------------------------
	//  Initialize Arrays
	// --------------------------------
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		tmp2 = i * (Ny_Fourier);
		tmp3 = i * (2 * sys_vars->kmax - 1);
		for (int j = 0; j < Ny; ++j) {
			if (j < Ny_Fourier) {
				run_data->w_hat[tmp2 + j] 				  = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 1] = 0.0 + 0.0 * I;
			}
			if ((i < 2 * sys_vars->kmax - 1) && (j < 2 * sys_vars->kmax - 1)) {
				proc_data->phases[tmp3 + j] = 0.0;
				proc_data->enst[tmp3 + j]   = 0.0;
				proc_data->enrg[tmp3 + j]   = 0.0;
			}
				run_data->w[tmp1 + j] = 0.0;
				run_data->u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
				run_data->u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
		}
	}
}
/**
 * Wrapper function for initializing FFTW plans
 * @param N Array containing the size of the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const int N_batch[SYS_DIM] = {Nx, Ny};

	#ifdef __REAL_STATS
	// Initialize Fourier Transforms
	sys_vars->fftw_2d_dft_c2r = fftw_plan_dft_c2r_2d(Nx, Ny, run_data->w_hat, run_data->w, FFTW_ESTIMATE);
	if (sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	
	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_c2r = fftw_plan_many_dft_c2r(SYS_DIM, N_batch, SYS_DIM, run_data->u_hat, NULL, SYS_DIM, 1, run_data->u, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	#endif
}
/**
* Function to compute 1D energy spectrum from the Fourier vorticity
*/
void EnergySpectrumAlt(void) {
    
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
        proc_data->enrg_alt[i] = 0.0;
    }

    // --------------------------------
	//  Get Psi
	// --------------------------------	
	for(int i = 0; i < Nx; ++i) {
	    tmp = i * Ny_Fourier;
	    for(int j = 0; j < Ny_Fourier; ++j) {
	        indx = tmp + j;

	        if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
	        	run_data->psi_hat[indx] = 0.0 + 0.0 * I;
	        }
	        else {
	       		k_sqr = 1.0 / ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

		        // Get Fourier stream function
		        run_data->psi_hat[indx] = run_data->w_hat[indx] * k_sqr;
	        }
	    }
	}

    // --------------------------------
	//  Compute spectrum
	// --------------------------------	
    for(int i = 0; i < Nx; ++i) {
        tmp = i * Ny_Fourier;
        for(int j = 0; j < Ny_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int ) round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
       		
       		// Compute the |k|^2
            k_sqr = ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
        
            if ((j == 0) || (j == Ny_Fourier - 1)) {
                proc_data->enrg_alt[spec_indx] += const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
            }
            else {
                proc_data->enrg_alt[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
            }
        }
    }
}
/**
* Function to compute 1D enstrophy spectrum from the Fourier vorticity
*/
void EnstrophySpectrumAlt(void) {
    
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
        proc_data->enst_alt[i] = 0.0;
    }


    // --------------------------------
	//  Get Psi
	// --------------------------------	
	for(int i = 0; i < Nx; ++i) {
	    tmp = i * Ny_Fourier;
	    for(int j = 0; j < Ny_Fourier; ++j) {
	        indx = tmp + j;

	        if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
	        	run_data->psi_hat[indx] = 0.0 + 0.0 * I;
	        }
	        else {
	       		k_sqr = 1.0 / ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

		        // Get Fourier stream function
		        run_data->psi_hat[indx] = run_data->w_hat[indx] * k_sqr;
	        }
	    }
	}

    // --------------------------------
	//  Compute spectrum
	// --------------------------------	
    for(int i = 0; i < Nx; ++i) {
        tmp = i * Ny_Fourier;
        for(int j = 0; j < Ny_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int ) round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
            
            // Compute |k|^2
            k_sqr = ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
        
            if ((j == 0) || (j == Ny_Fourier - 1)) {
                proc_data->enst_alt[spec_indx] += const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * pow(k_sqr, 2.0);
            }
            else {
                proc_data->enst_alt[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * pow(k_sqr, 2.0);
            }
        }
    }
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
 * Wrapper function to free memory and close any objects before exiting
 */
void FreeMemoryAndCleanUp(void) {


	// --------------------------------
	//  Free memory
	// --------------------------------
	fftw_free(run_data->w_hat);
	fftw_free(run_data->time);
	fftw_free(run_data->psi_hat);
	#ifdef __REAL_STATS
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	#endif
	#ifdef __FULL_FIELD
	fftw_free(proc_data->phases);
	fftw_free(proc_data->amps);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	#endif
	#ifdef __SPECTRA
	fftw_free(proc_data->enst_spec);
    fftw_free(proc_data->enrg_spec);
    fftw_free(proc_data->enst_alt);
    fftw_free(proc_data->enrg_alt);
	#endif
	#ifdef __SEC_PHASE_SYNC
	fftw_free(proc_data->theta);
	fftw_free(proc_data->k_angle);
	fftw_free(proc_data->k1_angle);
	fftw_free(proc_data->k2_angle);
	fftw_free(proc_data->k2_angle_neg);
	fftw_free(proc_data->phase_angle);
	fftw_free(proc_data->phase_order);
	fftw_free(proc_data->phase_R);
	fftw_free(proc_data->phase_Phi);
	fftw_free(proc_data->enst_flux);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		fftw_free(proc_data->triad_phase_order[i]);
		fftw_free(proc_data->triad_R[i]);
		fftw_free(proc_data->triad_Phi[i]);
		fftw_free(proc_data->num_triads[i]);
	}
	#endif
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	#ifdef __REAL_STATS
	// Free histogram structs
	gsl_histogram_free(stats_data->w_pdf);
	gsl_histogram_free(stats_data->u_pdf);
	#endif
	#ifdef __SEC_PHASE_SYNC
	for (int i = 0; i < sys_vars->num_sect; ++i) {
		gsl_histogram_free(proc_data->phase_sect_pdf[i]);
		gsl_histogram_free(proc_data->phase_sect_pdf_t[i]);
		gsl_histogram_free(proc_data->phase_sect_wghtd_pdf_t[i]);
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			gsl_histogram_free(proc_data->triad_sect_pdf[j][i]);
			gsl_histogram_free(proc_data->triad_sect_pdf_t[j][i]);
			gsl_histogram_free(proc_data->triad_sect_wghtd_pdf_t[j][i]);
		}	
	}	
	#endif

	// --------------------------------
	//  Free FFTW Plans
	// --------------------------------
	#ifdef __REAL_STATS
	// Destroy FFTW plans
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);
	#endif

	// --------------------------------
	//  Close HDF5 Objects
	// --------------------------------
	// Close HDF5 identifiers
	herr_t status = H5Tclose(file_info->COMPLEX_DTYPE);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close compound datatype for complex data\n-->> Exiting...\n");
		exit(1);		
	}
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
