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
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	#if defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
	int r;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double long_increment, trans_increment;
	double long_increment_abs, trans_increment_abs;
	int N_max_incr = (int) (GSL_MIN(Nx, Ny) / 2);
	int increment[NUM_INCR] = {1, N_max_incr};
	double norm_fac = 1.0 / (Nx * Ny);
	#endif


	// --------------------------------
	// Get In-Time Histogram Limits
	// --------------------------------
	#if defined(__REAL_STATS)
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
	#endif

	// --------------------------------
	// Compute Gradients
	// --------------------------------
	#if defined(__GRAD_STATS)
	// Compute the graident in Fourier space
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute the gradients of the vorticity
			proc_data->grad_w_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * run_data->w_hat[indx];
			proc_data->grad_w_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * run_data->w_hat[indx];

			// Compute the gradients of the vorticity
			proc_data->grad_u_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 0];
			proc_data->grad_u_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 1];
		}
	}

	// Perform inverse transform and normalize to get the gradients in real space
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_w_hat, proc_data->grad_w);
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_u_hat, proc_data->grad_u);
	double std_w_x = 0.0;
	double std_u_x = 0.0;
	double std_w_y = 0.0;
	double std_u_y = 0.0;
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Normalize the gradients
			proc_data->grad_w[SYS_DIM * indx + 0] *= norm_fac;
			proc_data->grad_w[SYS_DIM * indx + 1] *= norm_fac;
			proc_data->grad_u[SYS_DIM * indx + 0] *= norm_fac;
			proc_data->grad_u[SYS_DIM * indx + 1] *= norm_fac;

			if (s == 0) {
				// Update the sum for standard deviations
				std_u_x += pow(proc_data->grad_u[SYS_DIM * indx + 0], 2.0);
				std_u_y += pow(proc_data->grad_u[SYS_DIM * indx + 1], 2.0);
				std_w_x += pow(proc_data->grad_w[SYS_DIM * indx + 0], 2.0);
				std_w_y += pow(proc_data->grad_w[SYS_DIM * indx + 1], 2.0);
			}
		}
	}

	// If this is the first snapshot set the bin limits
	if (s == 0) {
		// Set the bin limits for the Gradients
		double std_w, std_u;
		for (int i = 0; i < INCR_TYPES; ++i) {
			if (i == 0)	{
				std_u = sqrt(std_u_x * norm_fac);
				std_w = sqrt(std_w_x * norm_fac);
			}
			else {
				std_u = sqrt(std_u_y * norm_fac);
				std_w = sqrt(std_w_y * norm_fac);
			}
			// Velocity gradients
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vel_grad[i], -BIN_LIM * std_u, BIN_LIM * std_u);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Increments");
				exit(1);
			}
			// Vorticity gradients
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vort_grad[i], -BIN_LIM * std_w, BIN_LIM * std_w);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Increments");
				exit(1);
			}
		}
	}
	#endif

	// --------------------------------
	// Update Histogram Counts
	// --------------------------------
	// Update histograms with the data from the current snapshot
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			#if defined(__REAL_STATS)
			///-------------------------------- Velocity & Vorticity Fields
			// Add current values to appropriate bins
			gsl_status = gsl_histogram_increment(stats_data->w_pdf, run_data->w[indx]);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s, gsl_status, run_data->w[indx]);
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_pdf, fabs(run_data->u[SYS_DIM * indx + 0]));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s, gsl_status, fabs(run_data->u[SYS_DIM * indx + 0]));
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_pdf, fabs(run_data->u[SYS_DIM * indx + 1]));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s, gsl_status, fabs(run_data->u[SYS_DIM * indx + 1]));
				exit(1);
			}
			#endif

			///-------------------------------- Gradient Fields
			#if defined(__GRAD_STATS)
			// Update Velocity gradient histograms
			gsl_status = gsl_histogram_increment(stats_data->vel_grad[0], proc_data->grad_u[SYS_DIM * indx + 0] * (norm_fac));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "X Velocity Gradient", s, gsl_status, proc_data->grad_u[SYS_DIM * indx + 0] * (norm_fac));
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->vel_grad[1], proc_data->grad_u[SYS_DIM * indx + 1] * (norm_fac));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Y Velocity Gradient", s, gsl_status, proc_data->grad_u[SYS_DIM * indx + 1] * (norm_fac));
				exit(1);
			}

			// Update Vorticity gradient histograms
			gsl_status = gsl_histogram_increment(stats_data->vort_grad[0], proc_data->grad_w[SYS_DIM * indx + 0] * (norm_fac));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "X Vorticity Gradient", s, gsl_status, proc_data->grad_w[SYS_DIM * indx + 0] * (norm_fac));
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->vort_grad[1], proc_data->grad_w[SYS_DIM * indx + 1] * (norm_fac));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Y Vorticity Gradient", s, gsl_status, proc_data->grad_w[SYS_DIM * indx + 1] * (norm_fac));
				exit(1);
			}
			#endif

			///-------------------------------- Velocity & Vorticity Incrments
			#if defined(__VEL_INC_STATS)
			// Compute velocity increments and update histograms
			for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
				// Get the current increment
				r = increment[r_indx];

				//------------- Get the longitudinal and transverse Velocity increments
				long_increment  = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0];
				trans_increment = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1];

				// Update the histograms
				gsl_status = gsl_histogram_increment(stats_data->vel_incr[0][r_indx], long_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment", s, gsl_status, long_increment);
					exit(1);
				}
				gsl_status = gsl_histogram_increment(stats_data->vel_incr[1][r_indx], trans_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Velocity Increment", s, gsl_status, trans_increment);
					exit(1);
				}

				//------------- Get the longitudinal and transverse Vorticity increments
				long_increment  = run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j];
				trans_increment = run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j];

				// Update the histograms
				gsl_status = gsl_histogram_increment(stats_data->w_incr[0][r_indx], long_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Vorticity Increment", s, gsl_status, long_increment);
					exit(1);
				}
				gsl_status = gsl_histogram_increment(stats_data->w_incr[1][r_indx], trans_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Vorticity Increment", s, gsl_status, trans_increment);
					exit(1);
				}
			}
			#endif
		}
	}

	// --------------------------------
	// Compute Structure Functions
	// --------------------------------
	#if defined(__STR_FUNC_STATS)
	for (int p = 2; p < STR_FUNC_MAX_POW; ++p) {
		for (int r_inc = 1; r_inc <= N_max_incr; ++r_inc) {
			// Initialize increments
			long_increment      = 0.0;
			trans_increment     = 0.0;
			long_increment_abs  = 0.0;
			trans_increment_abs = 0.0;
			for (int i = 0; i < Nx; ++i) {
				tmp = i * Ny;
				for (int j = 0; j < Ny; ++j) {
					indx = tmp + j;
				
					// Get increments
					long_increment      += pow(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0], p);
					trans_increment     += pow(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1], p);
					long_increment_abs  += pow(fabs(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0]), p);
					trans_increment_abs += pow(fabs(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1]), p);
				}
			}
			// Compute str function - normalize here
			stats_data->str_func[0][p - 2][r_inc - 1]     += long_increment * norm_fac;	
			stats_data->str_func[1][p - 2][r_inc - 1]     += trans_increment * norm_fac;
			stats_data->str_func_abs[0][p - 2][r_inc - 1] += long_increment_abs * norm_fac;	
			stats_data->str_func_abs[1][p - 2][r_inc - 1] += trans_increment_abs * norm_fac;
		}
	}
	#endif	
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
 * Function to compute the phase synchroniztion data sector by sector in wavenumber space for the current snaphsot
 * @param s The index of the current snapshot
 */
void SectorPhaseOrder(int s) {

	// Initialize variables
	double r_sqr;
	int indx;
	double angle;
	int gsl_status;
	double flux_wght;
	int num_phases;
	int tmp, tmp1;
	bool k2_neg, not_in_k1_sec, not_in_k3_sec;
	bool valid_wavevec, in_pos_sector, in_neg_sector;
	double flux_pre_fac;
	double k1_curl_k3;
	double sector_angle;
	double theta_lwr, theta_upr;
	double alt_theta_lwr, alt_theta_upr;
	int k_x, k1_x, k2_x, k2_y;
	int tmp_k, tmp_k1, tmp_k2;
	double phase, triad_phase, gen_triad_phase;
	double k_sqr, k_angle, k1_sqr, k1_angle, k2_sqr, k2_angle, neg_k2_angle;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// for (int i = 0; i < sys_vars->N[0]; ++i) {
	// 	if (abs(run_data->k[0][i]) < sys_vars->kmax) {
	// 		for (int j = 0; j < sys_vars->N[1] / 2 + 1; ++j) {
	// 			if (abs(run_data->k[1][j]) < sys_vars->kmax) {
	// 				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
	// 				angle = atan2((double) run_data->k[0][i], (double) run_data->k[1][j]); 
	// 				if (k_sqr > 0 && k_sqr <= sys_vars->kmax_sqr && angle >= proc_data->theta[25] && angle < proc_data->theta[26]) {
	// 					phase = 2.0;
	// 					proc_data->phases[(sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1) + (sys_vars->kmax - 1 + run_data->k[1][j])] = phase;
	// 					proc_data->phases[(sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1) + (sys_vars->kmax - 1 - run_data->k[1][j])] = -phase;
	// 				}
	// 			}
	// 		}
	// 	}
	// }

	// --------------------------------
	// Phase Sync Sector by Sector
	// --------------------------------
	// Loop through each sector
	// #pragma omp parallel for shared(run_data->k, proc_data->phases, proc_data->theta, proc_data->phase_order, proc_data->phase_sect_pdf, proc_data->phase_sect_pdf_t, proc_data->triad_phase_order, proc_data->ord_triad_phase_order) private(num_phases, num_ordered_phases, r, angle, phase)
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		
		// Get the angles for the current sector
		theta_lwr = proc_data->theta[a] - proc_data->dtheta / 2.0;
		theta_upr = proc_data->theta[a] + proc_data->dtheta / 2.0;

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
 				r_sqr     = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
 				angle = proc_data->phase_angle[tmp + j];
				
				// Update the phase order parameter for the current sector
 				if ((r_sqr < sys_vars->kmax_sqr) && (angle >= theta_lwr && angle < theta_upr)) {
					// Pre-compute phase data
					phase = proc_data->phases[tmp1 + (sys_vars->kmax - 1 + run_data->k[1][j])];

					// Update the phase order sum
					proc_data->phase_order[a] += cexp(I * phase);
 					
 					// Update counter
 					num_phases++;

					// Update individual phases PDFs for this sector
					gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf[a], phase - M_PI);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF", a, s, gsl_status, phase - M_PI);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf_t[a], phase - M_PI);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s, gsl_status, phase - M_PI);
						exit(1);
					}
					gsl_status = gsl_histogram_accumulate(proc_data->phase_sect_wghtd_pdf_t[a], phase - M_PI, proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s, gsl_status, phase - M_PI);
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

			// Initialize the flux for the current sector
			proc_data->enst_flux[i][a] = 0.0;
			for (int l = 0; l < sys_vars->num_sect; ++l) {
				// Initialize the number of triads for each type for across sectors
				proc_data->num_triads_across_sec[i][a][l] = 0;

				// Initialize the flux for the current sector for across sectors
				proc_data->enst_flux_across_sec[i][a][l] = 0.0;
			}
		}

		// Loop through second sector choice
		for (int l = 0; l < sys_vars->num_sect; ++l) {

			// Get the angles for the second sector
			alt_theta_lwr = proc_data->theta[(a + l) % sys_vars->num_sect] - proc_data->dtheta / 2.0;
			alt_theta_upr = proc_data->theta[(a + l) % sys_vars->num_sect] + proc_data->dtheta / 2.0;
			
			// Loop through the k wavevector (k is the k3 wavevector)
			for (int tmp_k_x = 0; tmp_k_x <= 2 * sys_vars->kmax - 1; ++tmp_k_x) {
				// Get k_x
				k_x = tmp_k_x - sys_vars->kmax + 1;

				for (int k_y = 0; k_y <= sys_vars->kmax; ++k_y) {
					// Get polar coords for the k wavevector
					k_sqr   = (double) (k_x * k_x + k_y * k_y);
					k_angle = proc_data->k_angle[tmp_k_x * (sys_vars->kmax + 1) + k_y]; 

					// Check if the k wavevector is in the current sector
					valid_wavevec = (k_sqr > 0 && k_sqr < sys_vars->kmax_sqr);
					in_pos_sector     = (k_angle >= theta_lwr && k_angle < theta_upr);
					if (valid_wavevec && in_pos_sector) {

						// Loop through the k1 wavevector
						for (int tmp_k1_x = 0; tmp_k1_x <= 2 * sys_vars->kmax - 1; ++tmp_k1_x) {
							// Get k1_x
							k1_x = tmp_k1_x - sys_vars->kmax + 1;

							for (int k1_y = 0; k1_y <= sys_vars->kmax; ++k1_y) {
								// Get polar coords for k1
								k1_sqr   = (double) (k1_x * k1_x + k1_y * k1_y);
								k1_angle = proc_data->k1_angle[tmp_k1_x * (sys_vars->kmax + 1) + k1_y];

								// Check if k1 wavevector is in the sector theta + l*dtheta
								valid_wavevec = (k1_sqr > 0 && k1_sqr < sys_vars->kmax_sqr);
								in_pos_sector     = (k1_angle >= alt_theta_lwr && k1_angle < alt_theta_upr);
								if(valid_wavevec && in_pos_sector) {
									
									// Find the k2 wavevector
									k2_x = k_x - k1_x;
									k2_y = k_y - k1_y;

									// Get k1 x k2 unit vectors
									k1_curl_k3 = (double)(k1_x * k_y - k1_y * k_x) / (sqrt(k1_sqr) * sqrt(k_sqr));
									
									// Get polar coords for k2
									k2_sqr   = (double) (k2_x * k2_x + k2_y * k2_y);
									k2_angle = proc_data->k2_angle[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y];

									// Get the value of the -k2 angle
									if (k2_y < 0 || (k2_y == 0 && k2_x > 0)) {
										// Get the angle of the negative wavevector k2 i.e., -k2 --> for checking the case if k2 is in the negative sector
										neg_k2_angle = proc_data->k2_angle_neg[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y];
									}

									// Get the appropriate sector angle for k2 -> k2 can either be in the same sector as k3 and k1 (pos or neg) or in the sector defined by the vector addition of k3 - k1
									if (l == 0) {
										sector_angle = proc_data->theta[a];
									}
									else {
										// (1.0 / (2.0 * I)) * (clog(k3 - k1) - clog(conj(k3) - conj(k1)));

										// Get the k2 sector angle from the precomputed angle sum
										if (k2_y < 0 && k2_x < 0) {
											// Case when k2 lands in ther lower left quadrant
											sector_angle = proc_data->mid_angle_sum[a * sys_vars->num_sect + l] - M_PI;	
											// sector_angle = creal(1.0 / (2.0 * I) * (clog(cexp(I * proc_data->theta[a]) - cexp(I * proc_data->theta[(a + l) % sys_vars->num_sect])) - clog(conj(cexp(I * proc_data->theta[a])) - conj(cexp(I * proc_data->theta[(a + l) % sys_vars->num_sect])))));	
										}
										else if (k2_y < 0 && k2_x > 0) {
											// Case when k2 lands in the upper left quadrant
											sector_angle = proc_data->mid_angle_sum[a * sys_vars->num_sect + l] + M_PI;
											// sector_angle = creal(1.0 / (2.0 * I) * (clog(cexp(I * proc_data->theta[a]) - cexp(I * proc_data->theta[(a + l) % sys_vars->num_sect])) - clog(conj(cexp(I * proc_data->theta[a])) - conj(cexp(I * proc_data->theta[(a + l) % sys_vars->num_sect])))));
										}
										else {
											// All the other cases
											sector_angle = proc_data->mid_angle_sum[a * sys_vars->num_sect + l];
											// sector_angle = creal(1.0 / (2.0 * I) * (clog(cexp(I * proc_data->theta[a]) - cexp(I * proc_data->theta[(a + l) % sys_vars->num_sect])) - clog(conj(cexp(I * proc_data->theta[a])) - conj(cexp(I * proc_data->theta[(a + l) % sys_vars->num_sect])))));
										}
									}

									// Check if k2 is in the current positive sector or if k2 is in the negative sector
									k2_neg        = (k2_y < 0 || (k2_y == 0 && k2_x > 0));
									in_pos_sector = (k2_angle >= sector_angle - proc_data->dtheta/2.0 && k2_angle < sector_angle + proc_data->dtheta/2.0 && k2_sqr > k1_sqr);
									valid_wavevec = (k2_sqr > 0 && k2_sqr < sys_vars->kmax_sqr);
									in_neg_sector = neg_k2_angle >= sector_angle - proc_data->dtheta/2.0 && neg_k2_angle < sector_angle + proc_data->dtheta/2.0;
									not_in_k1_sec = !(k2_angle >= alt_theta_lwr && k2_angle < alt_theta_upr) || !(k2_angle >= alt_theta_lwr + M_PI && k2_angle < alt_theta_upr + M_PI);
									not_in_k3_sec = !(k2_angle >= theta_lwr && k2_angle < theta_upr) || !(k2_angle >= theta_lwr + M_PI && k2_angle < theta_upr + M_PI);
									// if (a == 0 && l == 1) printf("l: %d\tsec_ang: %lf\tk2_ang: %lf\tin pos: %s\t in neg: %s\n", l, sector_angle, k2_angle, in_pos_sector ? "True" : "False", (k2_neg && in_neg_sector) ? "True" : "False");
									if (valid_wavevec &&  (in_pos_sector || (k2_neg && in_neg_sector)) && not_in_k1_sec && not_in_k3_sec) {
										// Get correct phase index -> recall that to access kx > 0, use -kx
										tmp_k  = (sys_vars->kmax - 1 - k_x) * (2 * sys_vars->kmax - 1);
										tmp_k1 = (sys_vars->kmax - 1 - k1_x) * (2 * sys_vars->kmax - 1);	
										tmp_k2 = (sys_vars->kmax - 1 - k2_x) * (2 * sys_vars->kmax - 1);

										// Get the wavevector prefactor -> the sign of the determines triad type
										flux_pre_fac = (double)(k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

										// Get the weighting (modulus) of this term to the contribution to the flux
										flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax - 1 + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax - 1 + k2_y] * proc_data->amps[tmp_k + sys_vars->kmax - 1 + k_y]);

										// Get the triad phase and adjust to with the orientation of the phase - the generalized phase
										triad_phase = proc_data->phases[tmp_k1 + sys_vars->kmax - 1 + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax - 1 + k2_y] - proc_data->phases[tmp_k + sys_vars->kmax - 1 + k_y];
										// triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
										
										///----------------- Get the generalized triads and triad case conditions
										// If the set C is the entire sector
										if (sys_vars->kmax_frac == 1.0) {
											// Get the generalize triad
											if (k_sqr > k1_sqr && k_sqr > k2_sqr) {
												gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
											}
											else if (k_sqr < k1_sqr && k_sqr < k2_sqr) {
												gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
											}
											else {
												gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
											}	

											// Get the wavevector conditions to both terms in the flux
											proc_data->pos_flux_term_cond = k_sqr > k1_sqr && k_sqr > k2_sqr;
											proc_data->neg_flux_term_cond = k_sqr < k1_sqr && k_sqr < k2_sqr;
										}
										// Else if the set C is some fraction of the sector
										else {
											// Get the generalize triad
											if (k_sqr > sys_vars->kmax_C_sqr && (k1_sqr <= sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_C_sqr)) {
												gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
											}
											else if (k_sqr <= sys_vars->kmax_C_sqr && (k1_sqr > sys_vars->kmax_C_sqr && k2_sqr > sys_vars->kmax_C_sqr)) {
												gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
											}
											else {
												gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
											}	

											// Get the wavevector conditions to both terms in the flux
											proc_data->pos_flux_term_cond = k_sqr > sys_vars->kmax_C_sqr && (k1_sqr <= sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_C_sqr);
											proc_data->neg_flux_term_cond = k_sqr <= sys_vars->kmax_C_sqr && (k1_sqr > sys_vars->kmax_C_sqr && k2_sqr > sys_vars->kmax_C_sqr);
										}

										///------------------ Determine which flux contribution we are dealing with -> the postive case (when k1 & k2 in C^' and k3 in C) or the negative case (when k3 in C^' and k1 and k2 in C)
										// The postive case
										if (proc_data->pos_flux_term_cond && fabs(flux_wght) > 1e-5) {  

											// Update the combined triad phase order parameter with the appropriate contribution
											proc_data->triad_phase_order[0][a] += cexp(I * gen_triad_phase);
											proc_data->triad_phase_order_across_sec[0][a][l] += cexp(I * gen_triad_phase);

											// Update the triad counter for the combined triad type
											proc_data->num_triads[0][a]++;
											proc_data->num_triads_across_sec[0][a][l]++;

											// Update the flux contribution for type 0
											proc_data->enst_flux[0][a] += 2.0 * flux_wght * cos(triad_phase);
											proc_data->enst_flux_across_sec[0][a][l] += 2.0 * flux_wght * cos(triad_phase);

											// ------ Update the PDFs of the combined triads
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
												exit(1);
											}
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
												exit(1);
											}
											gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(flux_wght));
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
												exit(1);
											}

											// Update the triad types and their PDFs
											if (flux_pre_fac < 0) {
												// TYPE 1 ---> Positive flux term and when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
												proc_data->triad_phase_order[1][a] += cexp(I * gen_triad_phase);
												proc_data->triad_phase_order_across_sec[1][a][l] += cexp(I * gen_triad_phase);
												proc_data->num_triads[1][a]++;		
												proc_data->num_triads_across_sec[1][a][l]++;		

												// Update the flux contribution for tpye 1
												proc_data->enst_flux[1][a] += 2.0 *flux_wght * cos(triad_phase);
												proc_data->enst_flux_across_sec[1][a][l] += 2.0 *flux_wght * cos(triad_phase);

												// Update the PDFs
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[1][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[1][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[1][a], gen_triad_phase, fabs(flux_wght));
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
											}
											else if (flux_pre_fac > 0) {
												// TYPE 2 ---> Positive flux term and when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
												proc_data->triad_phase_order[2][a] += cexp(I * gen_triad_phase);
												proc_data->triad_phase_order_across_sec[2][a][l] += cexp(I * gen_triad_phase);
												proc_data->num_triads[2][a]++;
												proc_data->num_triads_across_sec[2][a][l]++;

												// Update the flux contribution for type 2
												proc_data->enst_flux[2][a] += 2.0 * flux_wght * cos(triad_phase);
												proc_data->enst_flux_across_sec[2][a][l] += 2.0 * flux_wght * cos(triad_phase);

												// Update the PDFs
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[2][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Type 2 PDF", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[2][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[2][a], gen_triad_phase, fabs(flux_wght));
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
											}

										}
										// The negative case
										else if (proc_data->neg_flux_term_cond && fabs(flux_wght) > 1e-5) {

											// Update the combined triad phase order parameter with the appropriate contribution
											proc_data->triad_phase_order[0][a] += cexp(I * gen_triad_phase);
											proc_data->triad_phase_order_across_sec[0][a][l] += cexp(I * gen_triad_phase);

											// Update the triad counter for the combined triad type
											proc_data->num_triads[0][a]++;
											proc_data->num_triads_across_sec[0][a][l]++;

											// Update the flux contribution for type 0
											proc_data->enst_flux[0][a] -= 2.0 * flux_wght * cos(triad_phase);
											proc_data->enst_flux_across_sec[0][a][l] -= 2.0 * flux_wght * cos(triad_phase);

											// ------ Update the PDFs of the combined triads
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
												exit(1);
											}
											gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
												exit(1);
											}
											gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(-flux_wght));
											if (gsl_status != 0) {
												fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
												exit(1);
											}

											// Update the triad types and their PDFs
											if (flux_pre_fac < 0) {
												// TYPE 3 ---> Negative flux term and when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
												proc_data->triad_phase_order[3][a] += cexp(I * gen_triad_phase);
												proc_data->triad_phase_order_across_sec[3][a][l] += cexp(I * gen_triad_phase);
												proc_data->num_triads[3][a]++;
												proc_data->num_triads_across_sec[3][a][l]++;

												// Update the flux contribution for type 3
												proc_data->enst_flux[3][a] -= 2.0 * flux_wght * cos(triad_phase);
												proc_data->enst_flux_across_sec[3][a][l] -= 2.0 * flux_wght * cos(triad_phase);

												// Update the PDFs
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[3][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[3][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[3][a], gen_triad_phase, fabs(-flux_wght));
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
											}
											else if (flux_pre_fac > 0) {
												// TYPE 4 ---> Negative flux term and when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
												proc_data->triad_phase_order[4][a] += cexp(I * gen_triad_phase);
												proc_data->triad_phase_order_across_sec[4][a][l] += cexp(I * gen_triad_phase);
												proc_data->num_triads[4][a]++;
												proc_data->num_triads_across_sec[4][a][l]++;

												// Update the flux contribution for type 4
												proc_data->enst_flux[4][a] -= 2.0 * flux_wght * cos(triad_phase);
												proc_data->enst_flux_across_sec[4][a][l] -= 2.0 * flux_wght * cos(triad_phase);

												// Update the PDFs
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[4][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[4][a], gen_triad_phase);
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
												gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[4][a], gen_triad_phase, fabs(-flux_wght)); // phas
												if (gsl_status != 0) {
													fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
													exit(1);
												}
											}
										}
										else if (fabs(flux_wght) <= 1e-5 || k1_sqr == k2_sqr || k1_angle == k2_angle){
											// TYPE 5 ---> When the modulus of the flux term is 0 but not necessarily the triad
											proc_data->triad_phase_order[5][a] += cexp(I * gen_triad_phase);
											proc_data->triad_phase_order_across_sec[5][a][l] += cexp(I * gen_triad_phase);
											proc_data->num_triads[5][a]++;
											proc_data->num_triads_across_sec[5][a][l]++;
										}
										else {
											// When k1 and k2 = k3 the contribution to the flux is zero
											continue;
										}
									}	
									// if (a == 19 && l == 1 && k_sqr <= sys_vars->kmax_C_sqr + 3 &&  k_sqr > sys_vars->kmax_C_sqr) {
									// 	printf("a: %d\ttype: %d\tl = %d\tnum_t: %d\tk1 = (%d, %d)\tk2 = (%d, %d)\tk3 = (%d, %d)\n",a, 1, l, proc_data->num_triads_across_sec[1][a][l], k1_x, k1_y, k2_x, k2_y, k_x, k_y);
									// 	// printf("a: %d\ttype: %d\tl = %d\tnum_t: %d\tord: %lf %lf I\t\tR: %lf\tPhi: %lf\n", a, i, l, proc_data->num_triads_across_sec[i][a][l], creal(proc_data->triad_phase_order_across_sec[i][a][l]), cimag(proc_data->triad_phase_order_across_sec[i][a][l]), proc_data->triad_R_across_sec[i][a][l], proc_data->triad_Phi_across_sec[i][a][l]);
									// }								
								}
							}
						}
					}
				}		

			}
		// if(a == 19 && l == 1) printf("\n");	
		}

		

		//------------------- Record the data for the triads
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			// Normalize the phase order parameters
			if (proc_data->num_triads[i][a] != 0) {
				proc_data->triad_phase_order[i][a] /= proc_data->num_triads[i][a];
			}
			
			// Record the phase syncs and average phases
			proc_data->triad_R[i][a]   = cabs(proc_data->triad_phase_order[i][a]);
			proc_data->triad_Phi[i][a] = carg(proc_data->triad_phase_order[i][a]);
			for (int l = 0; l < sys_vars->num_sect; ++l) {
				if (proc_data->num_triads_across_sec[i][a][l] != 0) {
					proc_data->triad_phase_order_across_sec[i][a][l] /= proc_data->num_triads_across_sec[i][a][l];
				}
				
				// Record the phase syncs and average phases
				proc_data->triad_R_across_sec[i][a][l]   = cabs(proc_data->triad_phase_order_across_sec[i][a][l]);
				proc_data->triad_Phi_across_sec[i][a][l] = carg(proc_data->triad_phase_order_across_sec[i][a][l]); 

			}
			// if (a == 0)printf("a: %d type: %d Num: %d\t triad_phase_order: %lf %lf I\t\t R: %lf, Phi %lf\n", a, i, proc_data->num_triads[i][a], creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->triad_R[i][a], proc_data->triad_Phi[i][a]);
			
			// printf("Num Triads Type %d: %d\tord: %lf %lf I\tR: %lf\tPhi: %lf\n", i, proc_data->num_triads[i][a], creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->triad_R[i][a], proc_data->triad_Phi[i][a]);
			// printf("a = %d\t ord: %1.16lf %1.16lf I\t num: %d\tR: %lf\n", a, creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->num_triads[i][a], proc_data->triad_R[i][a]);
		}
		// printf("\n");
		// printf("a: %d Num: %d\t triad_phase_order: %lf %lf I\t Num: %d\t triad_phase_order: %lf %lf I \t Num: %d\t triad_phase_order: %lf %lf I\n", a, num_triads[0], creal(proc_data->triad_phase_order[0][a]), cimag(proc_data->triad_phase_order[0][a]), num_triads[1], creal(proc_data->triad_phase_order[1][a]), cimag(proc_data->triad_phase_order[1][a]), num_triads[2], creal(proc_data->triad_phase_order[2][a]), cimag(proc_data->triad_phase_order[2][a]));
		// printf("a: %d | Num: %d R0: %lf Phi0: %lf |\t Num: %d R1: %lf Phi1: %lf |\t Num: %d R2: %lf Phi2: %lf\n", a, num_triads[0], proc_data->triad_R[0][a], proc_data->triad_Phi[0][a], num_triads[1], proc_data->triad_R[1][a], proc_data->triad_Phi[1][a], num_triads[2], proc_data->triad_R[2][a], proc_data->triad_Phi[2][a]);	
		// 
		
		// ----------------------------------
		// Reset Phase Order Parameters
		// ----------------------------------
		// Set parameters to 0 for next snapshot
		for (int a = 0; a < sys_vars->num_sect; ++a) {
			proc_data->phase_order[a] = 0.0 + 0.0 * I;
			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
				proc_data->triad_phase_order[i][a] = 0.0 + 0.0 * I;
				for (int l = 0; l < sys_vars->num_sect; ++l) {
					proc_data->triad_phase_order_across_sec[i][a][l] = 0.0 + 0.0 * I;
				}
			}	
		}
	}
}
/**
 * Function to compute the phase synchroniztion data sector by sector in wavenumber space for the current snaphsot
 * @param s The index of the current snapshot
 */
// void SectorPhaseOrderPar(int s) {

// 	// Initialize variables
// 	double r_sqr;
// 	int indx;
// 	double angle;
// 	int gsl_status;
// 	double flux_wght;
// 	double amp;
// 	int num_phases;
// 	int tmp, tmp1;
// 	bool k2_neg, not_in_k1_sec, not_in_k3_sec;
// 	bool valid_wavevec, in_pos_sector, in_neg_sector;
// 	double flux_pre_fac;
// 	double k1_curl_k3;
// 	double sector_angle;
// 	double theta_lwr, theta_upr;
// 	double alt_theta_lwr, alt_theta_upr;
// 	int k_x, k1_x, k2_x, k2_y;
// 	int tmp_k, tmp_k1, tmp_k2;
// 	double phase, triad_phase, gen_triad_phase;
// 	double k_sqr, k_angle, k1_sqr, k1_angle, k2_sqr, k2_angle, neg_k2_angle;
// 	const long int Nx 		  = sys_vars->N[0];
// 	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

// 	// for (int i = 0; i < sys_vars->N[0]; ++i) {
// 	// 	if (abs(run_data->k[0][i]) < sys_vars->kmax) {
// 	// 		for (int j = 0; j < sys_vars->N[1] / 2 + 1; ++j) {
// 	// 			if (abs(run_data->k[1][j]) < sys_vars->kmax) {
// 	// 				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
// 	// 				angle = atan2((double) run_data->k[0][i], (double) run_data->k[1][j]); 
// 	// 				if (k_sqr > 0 && k_sqr <= sys_vars->kmax_sqr && angle >= proc_data->theta[25] && angle < proc_data->theta[26]) {
// 	// 					phase = 2.0;
// 	// 					proc_data->phases[(sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1) + (sys_vars->kmax - 1 + run_data->k[1][j])] = phase;
// 	// 					proc_data->phases[(sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1) + (sys_vars->kmax - 1 - run_data->k[1][j])] = -phase;
// 	// 				}
// 	// 			}
// 	// 		}
// 	// 	}
// 	// }

// 	// --------------------------------
// 	// Phase Sync Sector by Sector
// 	// --------------------------------
// 	// Loop through each sector
// 	#pragma omp parallel for shared(proc_data, sys_vars, run_data) private(phase, num_phases, theta_lwr, theta_upr, tmp, tmp1, angle, r_sqr, amp, gsl_status)
// 	for (int a = 0; a < sys_vars->num_sect; ++a) {
		

// 		int t_id = omp_get_thread_num();

// 		// Get the angles for the current sector
// 		theta_lwr = proc_data->theta[a] - proc_data->dtheta / 2.0;
// 		theta_upr = proc_data->theta[a] + proc_data->dtheta / 2.0;
		
// 		// --------------------------------
// 		// Individual Phases
// 		// --------------------------------
// 		// Initialize phase counter
// 		num_phases = 0;

// 		int kx, ky;

// 		// Loop through the phases
// 		for (int i = 0; i < Nx; ++i) {
// 			kx = (i <= Nx / 2) ? i : i - Nx;

// 			if (abs(kx) < sys_vars->kmax) {
// 				tmp  = i * Ny_Fourier;	
// 				tmp1 = (sys_vars->kmax - 1 - kx) * (2 * sys_vars->kmax - 1); // correct indexing for the phases -> for kx > 0 use -kx

// 				for (int j = 0; j < Ny_Fourier; ++j) {
// 					indx = tmp + j;

// 					ky = j;

// 					if (abs(ky) < sys_vars->kmax) {
		
// 						// Compute the polar coords for the current mode
// 		 				r_sqr = (double) (kx * kx + ky * ky);
// 		 				angle = proc_data->phase_angle[tmp + j];
						

// 						// Update the phase order parameter for the current sector
// 		 				if ((r_sqr < sys_vars->kmax_sqr) && (angle >= theta_lwr && angle < theta_upr)) {
// 							// Pre-compute phase data
// 							phase = proc_data->phases[tmp1 + (sys_vars->kmax - 1 + ky)];
// 							printf("Phase: %lf\n", phase);
// 							if (phase <= -50.0) printf("ky: %d\t kx: %d\tkmax: %d\n", ky, kx, (int)sqrt(sys_vars->kmax_sqr));
// 							if (phase < 0.0 || phase >= 2.0 * M_PI) printf("id: %d\tky: %d\t kx: %d\tkmax: %d\tr_sqr: %lf\tr_max: %lf\ta: %lf\tphase: %lf\tp: %lf\n", t_id, ky, kx, (int)sqrt(sys_vars->kmax_sqr), r_sqr, sys_vars->kmax_sqr, angle, phase, proc_data->phases[tmp1 + (sys_vars->kmax - 1 + ky)]);
							
// 							// Update the phase order sum
// 							proc_data->phase_order[a] += cexp(I * phase);
		 					
// 		 					// Update counter
// 		 					num_phases++;

// 		 					// Get the amplitude
// 		 					amp = proc_data->amps[tmp1 + sys_vars->kmax - 1 + ky];
		 					

// 							// Update individual phases PDFs for this sector
// 							gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf[a], phase - M_PI);
// 							if (gsl_status != 0) {
// 								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"] %d\n-->> Exiting!!!\n", "Sector Phase PDF", a, s, gsl_status, phase - M_PI,  t_id);
// 								exit(1);
// 							}
// 							gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf_t[a], phase - M_PI);
// 							if (gsl_status != 0) {
// 								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"] %d\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s, gsl_status, phase - M_PI,  t_id);
// 								exit(1);
// 							}
// 							gsl_status = gsl_histogram_accumulate(proc_data->phase_sect_wghtd_pdf_t[a], phase - M_PI, amp);
// 							if (gsl_status != 0) {
// 								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"] %d\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s, gsl_status, phase - M_PI,  t_id);
// 								exit(1);
// 							}
// 		 				}
// 		 			}
// 		 		}
// 		 	}	
// 		}

// 		///-------------------- Record the individual phase data
// 		// Normalize the phase order parameter
// 		proc_data->phase_order[a] /= num_phases;

// 		// Record the phase sync and average phase
// 		proc_data->phase_R[a]   = cabs(proc_data->phase_order[a]);
// 		proc_data->phase_Phi[a] = carg(proc_data->phase_order[a]);

// 		// // --------------------------------
// 		// // Triad Phases
// 		// // --------------------------------
// 		// // Initialize counters for each triad type
// 		// for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
// 		// 	// Initialize the number of triads for each type
// 		// 	proc_data->num_triads[i][a] = 0;

// 		// 	// Initialize the flux for the current sector
// 		// 	proc_data->enst_flux[i][a] = 0.0;
// 		// }

// 		// // Loop through second sector choice
// 		// for (int l = 0; l < sys_vars->num_sect; ++l) {

// 		// 	// Get the angles for the second sector
// 		// 	alt_theta_lwr = proc_data->theta[(a + l) % sys_vars->num_sect] - proc_data->dtheta / 2.0;
// 		// 	alt_theta_upr = proc_data->theta[(a + l) % sys_vars->num_sect] + proc_data->dtheta / 2.0;
			
// 		// 	// Loop through the k wavevector (k is the k3 wavevector)
// 		// 	for (int tmp_k_x = 0; tmp_k_x <= 2 * sys_vars->kmax - 1; ++tmp_k_x) {
// 		// 		// Get k_x
// 		// 		k_x = tmp_k_x - sys_vars->kmax + 1;

// 		// 		for (int k_y = 0; k_y <= sys_vars->kmax; ++k_y) {
// 		// 			// Get polar coords for the k wavevector
// 		// 			k_sqr   = (double) (k_x * k_x + k_y * k_y);
// 		// 			k_angle = proc_data->k_angle[tmp_k_x * (sys_vars->kmax + 1) + k_y]; 

// 		// 			// Check if the k wavevector is in the current sector
// 		// 			valid_wavevec = (k_sqr > 0 && k_sqr < sys_vars->kmax_sqr);
// 		// 			in_pos_sector     = (k_angle >= theta_lwr && k_angle < theta_upr);
// 		// 			if (valid_wavevec && in_pos_sector) {

// 		// 				// Loop through the k1 wavevector
// 		// 				for (int tmp_k1_x = 0; tmp_k1_x <= 2 * sys_vars->kmax - 1; ++tmp_k1_x) {
// 		// 					// Get k1_x
// 		// 					k1_x = tmp_k1_x - sys_vars->kmax + 1;

// 		// 					for (int k1_y = 0; k1_y <= sys_vars->kmax; ++k1_y) {
// 		// 						// Get polar coords for k1
// 		// 						k1_sqr   = (double) (k1_x * k1_x + k1_y * k1_y);
// 		// 						k1_angle = proc_data->k1_angle[tmp_k1_x * (sys_vars->kmax + 1) + k1_y];

// 		// 						// Check if k1 wavevector is in the sector theta + l*dtheta
// 		// 						valid_wavevec = (k1_sqr > 0 && k1_sqr < sys_vars->kmax_sqr);
// 		// 						in_pos_sector     = (k1_angle >= alt_theta_lwr && k1_angle < alt_theta_upr);
// 		// 						if(valid_wavevec && in_pos_sector) {
									
// 		// 							// Find the k2 wavevector
// 		// 							k2_x = k_x - k1_x;
// 		// 							k2_y = k_y - k1_y;

// 		// 							// Get k1 x k2 unit vectors
// 		// 							k1_curl_k3 = (double)(k1_x * k_y - k1_y * k_x) / (sqrt(k1_sqr) * sqrt(k_sqr));
									
// 		// 							// Get polar coords for k2
// 		// 							k2_sqr   = (double) (k2_x * k2_x + k2_y * k2_y);
// 		// 							k2_angle = proc_data->k2_angle[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y];

// 		// 							// Get the value of the -k2 angle
// 		// 							if (k2_y < 0 || (k2_y == 0 && k2_x > 0)) {
// 		// 								// Get the angle of the negative wavevector k2 i.e., -k2 --> for checking the case if k2 is in the negative sector
// 		// 								neg_k2_angle = proc_data->k2_angle_neg[(sys_vars->kmax + 1) * ((2 * sys_vars->kmax) * (tmp_k_x * (sys_vars->kmax + 1) + k_y) + tmp_k1_x) + k1_y];
// 		// 							}

// 		// 							// Get the appropriate sector angle for k2 -> k2 can either be in the same sector as k3 and k1 (pos or neg) or in the sector defined by the vector addition of k3 - k1
// 		// 							if (l == 0) {
// 		// 								sector_angle = proc_data->theta[a];
// 		// 							}
// 		// 							else {
// 		// 								// Get the k2 sector angle from the precomputed angle sum
// 		// 								if (k2_y < 0 && k2_x < 0) {
// 		// 									// Case when k2 lands in ther lower left quadrant
// 		// 									sector_angle = proc_data->mid_angle_sum[a * sys_vars->num_sect + l] + M_PI;	
// 		// 								}
// 		// 								else if (k2_y < 0 && k2_x > 0) {
// 		// 									// Case when k2 lands in the upper left quadrant
// 		// 									sector_angle = proc_data->mid_angle_sum[a * sys_vars->num_sect + l] - M_PI;
// 		// 								}
// 		// 								else {
// 		// 									// All the other cases
// 		// 									sector_angle = proc_data->mid_angle_sum[a * sys_vars->num_sect + l];
// 		// 								}
										
// 		// 							}

// 		// 							// Check if k2 is in the current positive sector or if k2 is in the negative sector
// 		// 							k2_neg        = (k2_y < 0 || (k2_y == 0 && k2_x > 0));
// 		// 							in_pos_sector = (k2_angle >= sector_angle - proc_data->dtheta/2.0 && k2_angle < sector_angle + proc_data->dtheta/2.0 && k2_sqr > k1_sqr);
// 		// 							valid_wavevec = (k2_sqr > 0 && k2_sqr < sys_vars->kmax_sqr);
// 		// 							in_neg_sector = neg_k2_angle >= sector_angle - proc_data->dtheta/2.0 && neg_k2_angle < sector_angle + proc_data->dtheta/2.0;
// 		// 							not_in_k1_sec = !(k2_angle >= alt_theta_lwr && k2_angle < alt_theta_upr);
// 		// 							not_in_k3_sec = !(k2_angle >= theta_lwr && k2_angle < theta_upr);
// 		// 							if (valid_wavevec &&  (in_pos_sector || (k2_neg && in_neg_sector)) && not_in_k1_sec && not_in_k3_sec) {
// 		// 								// Get correct phase index -> recall that to access kx > 0, use -kx
// 		// 								tmp_k  = (sys_vars->kmax - 1 - k_x) * (2 * sys_vars->kmax - 1);
// 		// 								tmp_k1 = (sys_vars->kmax - 1 - k1_x) * (2 * sys_vars->kmax - 1);	
// 		// 								tmp_k2 = (sys_vars->kmax - 1 - k2_x) * (2 * sys_vars->kmax - 1);

// 		// 								// Get the wavevector prefactor -> the sign of the determines triad type
// 		// 								flux_pre_fac = (double)(k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

// 		// 								// Get the weighting (modulus) of this term to the contribution to the flux
// 		// 								flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax - 1 + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax - 1 + k2_y] * proc_data->amps[tmp_k + sys_vars->kmax - 1 + k_y]);

// 		// 								// Get the triad phase and adjust to with the orientation of the phase - the generalized phase
// 		// 								triad_phase = proc_data->phases[tmp_k1 + sys_vars->kmax - 1 + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax - 1 + k2_y] - proc_data->phases[tmp_k + sys_vars->kmax - 1 + k_y];
// 		// 								// triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
										
// 		// 								///----------------- Get the generalized triads and triad case conditions
// 		// 								// If the set C is the entire sector
// 		// 								if (sys_vars->kmax_frac == 1.0) {
// 		// 									// Get the generalize triad
// 		// 									if (k_sqr > k1_sqr && k_sqr > k2_sqr) {
// 		// 										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
// 		// 									}
// 		// 									else if (k_sqr < k1_sqr && k_sqr < k2_sqr) {
// 		// 										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
// 		// 									}
// 		// 									else {
// 		// 										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
// 		// 									}	

// 		// 									// Get the wavevector conditions to both terms in the flux
// 		// 									proc_data->pos_flux_term_cond = k_sqr > k1_sqr && k_sqr > k2_sqr;
// 		// 									proc_data->neg_flux_term_cond = k_sqr < k1_sqr && k_sqr < k2_sqr;
// 		// 								}
// 		// 								// Else if the set C is some fraction of the sector
// 		// 								else {
// 		// 									// Get the generalize triad
// 		// 									if (k_sqr > sys_vars->kmax_C_sqr && (k1_sqr <= sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_C_sqr)) {
// 		// 										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
// 		// 									}
// 		// 									else if (k_sqr <= sys_vars->kmax_C_sqr && (k1_sqr > sys_vars->kmax_C_sqr && k2_sqr > sys_vars->kmax_C_sqr)) {
// 		// 										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
// 		// 									}
// 		// 									else {
// 		// 										gen_triad_phase = fmod(triad_phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;
// 		// 									}	

// 		// 									// Get the wavevector conditions to both terms in the flux
// 		// 									proc_data->pos_flux_term_cond = k_sqr > sys_vars->kmax_C_sqr && (k1_sqr <= sys_vars->kmax_C_sqr && k2_sqr <= sys_vars->kmax_C_sqr);
// 		// 									proc_data->neg_flux_term_cond = k_sqr <= sys_vars->kmax_C_sqr && (k1_sqr > sys_vars->kmax_C_sqr && k2_sqr > sys_vars->kmax_C_sqr);
// 		// 								}

// 		// 								///------------------ Determine which flux contribution we are dealing with -> the postive case (when k1 & k2 in C^' and k3 in C) or the negative case (when k3 in C^' and k1 and k2 in C)
// 		// 								// The postive case
// 		// 								if (proc_data->pos_flux_term_cond && fabs(flux_wght) > 1e-5) {  

// 		// 									// Update the combined triad phase order parameter with the appropriate contribution
// 		// 									proc_data->triad_phase_order[0][a] += cexp(I * gen_triad_phase);

// 		// 									// Update the triad counter for the combined triad type
// 		// 									proc_data->num_triads[0][a]++;

// 		// 									// Update the flux contribution for type 0
// 		// 									proc_data->enst_flux[0][a] += 2.0 * flux_wght * cos(triad_phase);

// 		// 									// ------ Update the PDFs of the combined triads
// 		// 									gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
// 		// 									if (gsl_status != 0) {
// 		// 										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
// 		// 										exit(1);
// 		// 									}
// 		// 									gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
// 		// 									if (gsl_status != 0) {
// 		// 										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 										exit(1);
// 		// 									}
// 		// 									gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(flux_wght));
// 		// 									if (gsl_status != 0) {
// 		// 										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 										exit(1);
// 		// 									}

// 		// 									// Update the triad types and their PDFs
// 		// 									if (flux_pre_fac < 0) {
// 		// 										// TYPE 1 ---> Positive flux term and when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
// 		// 										proc_data->triad_phase_order[1][a] += cexp(I * gen_triad_phase);
// 		// 										proc_data->num_triads[1][a]++;		

// 		// 										// Update the flux contribution for tpye 1
// 		// 										proc_data->enst_flux[1][a] += 2.0 *flux_wght * cos(triad_phase);

// 		// 										// Update the PDFs
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[1][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[1][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[1][a], gen_triad_phase, fabs(flux_wght));
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 									}
// 		// 									else if (flux_pre_fac > 0) {
// 		// 										// TYPE 2 ---> Positive flux term and when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
// 		// 										proc_data->triad_phase_order[2][a] += cexp(I * gen_triad_phase);
// 		// 										proc_data->num_triads[2][a]++;

// 		// 										// Update the flux contribution for type 2
// 		// 										proc_data->enst_flux[2][a] += 2.0 * flux_wght * cos(triad_phase);

// 		// 										// Update the PDFs
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[2][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Type 2 PDF", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[2][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[2][a], gen_triad_phase, fabs(flux_wght));
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 									}

// 		// 								}
// 		// 								// The negative case
// 		// 								else if (proc_data->neg_flux_term_cond && fabs(flux_wght) > 1e-5) {

// 		// 									// Update the combined triad phase order parameter with the appropriate contribution
// 		// 									proc_data->triad_phase_order[0][a] += cexp(I * gen_triad_phase);

// 		// 									// Update the triad counter for the combined triad type
// 		// 									proc_data->num_triads[0][a]++;

// 		// 									// Update the flux contribution for type 0
// 		// 									proc_data->enst_flux[0][a] -= 2.0 * flux_wght * cos(triad_phase);

// 		// 									// ------ Update the PDFs of the combined triads
// 		// 									gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
// 		// 									if (gsl_status != 0) {
// 		// 										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
// 		// 										exit(1);
// 		// 									}
// 		// 									gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
// 		// 									if (gsl_status != 0) {
// 		// 										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 										exit(1);
// 		// 									}
// 		// 									gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(-flux_wght));
// 		// 									if (gsl_status != 0) {
// 		// 										fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 										exit(1);
// 		// 									}

// 		// 									// Update the triad types and their PDFs
// 		// 									if (flux_pre_fac < 0) {
// 		// 										// TYPE 3 ---> Negative flux term and when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
// 		// 										proc_data->triad_phase_order[3][a] += cexp(I * gen_triad_phase);
// 		// 										proc_data->num_triads[3][a]++;

// 		// 										// Update the flux contribution for type 3
// 		// 										proc_data->enst_flux[3][a] -= 2.0 * flux_wght * cos(triad_phase);

// 		// 										// Update the PDFs
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[3][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[3][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[3][a], gen_triad_phase, fabs(-flux_wght));
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 									}
// 		// 									else if (flux_pre_fac > 0) {
// 		// 										// TYPE 4 ---> Negative flux term and when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
// 		// 										proc_data->triad_phase_order[4][a] += cexp(I * gen_triad_phase);
// 		// 										proc_data->num_triads[4][a]++;

// 		// 										// Update the flux contribution for type 4
// 		// 										proc_data->enst_flux[4][a] -= 2.0 * flux_wght * cos(triad_phase);

// 		// 										// Update the PDFs
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[4][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[4][a], gen_triad_phase);
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 										gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[4][a], gen_triad_phase, fabs(-flux_wght)); // phas
// 		// 										if (gsl_status != 0) {
// 		// 											fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
// 		// 											exit(1);
// 		// 										}
// 		// 									}
// 		// 								}
// 		// 								else {
// 		// 									// When k1 and k2 = k3 the contribution to the flux is zero
// 		// 									continue;
// 		// 								}
// 		// 							}									
// 		// 						}
// 		// 					}
// 		// 				}
// 		// 			}
// 		// 		}				
// 		// 	}
// 		// }

		

// 		// //------------------- Record the data for the triads
// 		// for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
// 		// 	// Normalize the phase order parameters
// 		// 	if (proc_data->num_triads[i][a] != 0) {
// 		// 		proc_data->triad_phase_order[i][a] /= proc_data->num_triads[i][a];
// 		// 	}
			
// 		// 	// Record the phase syncs and average phases
// 		// 	proc_data->triad_R[i][a]   = cabs(proc_data->triad_phase_order[i][a]);
// 		// 	proc_data->triad_Phi[i][a] = carg(proc_data->triad_phase_order[i][a]); 
// 		// 	// printf("a: %d type: %d Num: %d\t triad_phase_order: %lf %lf I\t\t R: %lf, Phi %lf\n", a, i, num_triads[i], creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->triad_R[i][a], proc_data->triad_Phi[i][a]);
// 		// 	// printf("Num Triads Type %d: %d\tord: %lf %lf I\tR: %lf\tPhi: %lf\n", i, proc_data->num_triads[i][a], creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->triad_R[i][a], proc_data->triad_Phi[i][a]);
// 		// 	// printf("a = %d\t ord: %1.16lf %1.16lf I\t num: %d\tR: %lf\n", a, creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->num_triads[i][a], proc_data->triad_R[i][a]);
// 		// }
// 		// // printf("\n");
// 		// // printf("a: %d Num: %d\t triad_phase_order: %lf %lf I\t Num: %d\t triad_phase_order: %lf %lf I \t Num: %d\t triad_phase_order: %lf %lf I\n", a, num_triads[0], creal(proc_data->triad_phase_order[0][a]), cimag(proc_data->triad_phase_order[0][a]), num_triads[1], creal(proc_data->triad_phase_order[1][a]), cimag(proc_data->triad_phase_order[1][a]), num_triads[2], creal(proc_data->triad_phase_order[2][a]), cimag(proc_data->triad_phase_order[2][a]));
// 		// // printf("a: %d | Num: %d R0: %lf Phi0: %lf |\t Num: %d R1: %lf Phi1: %lf |\t Num: %d R2: %lf Phi2: %lf\n", a, num_triads[0], proc_data->triad_R[0][a], proc_data->triad_Phi[0][a], num_triads[1], proc_data->triad_R[1][a], proc_data->triad_Phi[1][a], num_triads[2], proc_data->triad_R[2][a], proc_data->triad_Phi[2][a]);	
// 		// // 
		
// 		// ----------------------------------
// 		// Reset Phase Order Parameters
// 		// ----------------------------------
// 		// Set parameters to 0 for next snapshot
// 		for (int a = 0; a < sys_vars->num_sect; ++a) {
// 			proc_data->phase_order[a] = 0.0 + 0.0 * I;
// 			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
// 				proc_data->triad_phase_order[i][a] = 0.0 + 0.0 * I;
// 			}	
// 		}
// 	}
//}
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
			// printf("-k_sqr[%d, %d]: %1.16lf %1.16lf I \t kx[%d]: %1.16lf \t wh[%d, %d]: %1.16lf %1.16lf I \n", i, j, creal(-1.0 * k_sqr), cimag(-1.0 * k_sqr), i, ((double) run_data->k[0][i]), i, j, creal(w_hat[indx]), cimag(w_hat[indx]));
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
 * Function to compute the enstrophy flux spectrum and enstorphy flux in C for the current snapshot of the simulation
 * The results are gathered on the master rank before being written to file
 * @param snap    The current snapshot of the simulation
 */
void EnstrophyFluxSpectrum(int snap) {

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

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		proc_data->enst_flux_spec[i] = 0.0;
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
				#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// Both Hyperviscosity and Ekman drag
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// No hyperviscosity but we have Ekman drag
				pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
				// Hyperviscosity only
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
				#else 
				// No hyper viscosity or no ekman drag -> just normal viscosity
				pre_fac = sys_vars->NU * k_sqr; 
				#endif

				// Get the spectrum index
				spec_indx = (int) ceil(sqrt(k_sqr));

				// Update spectrum bin
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the current bin sum 
					proc_data->enst_flux_spec[spec_indx] += 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]);
				}
				else {
					// Update the running sum for the flux of energy
					proc_data->enst_flux_spec[spec_indx] += 2.0 * 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(proc_data->dw_hat_dt[indx]) + conj(run_data->w_hat[indx]) * proc_data->dw_hat_dt[indx]);
				}

			}
		}
	}

	// -------------------------------------
	// Accumulate The Flux
	// -------------------------------------
	// Accumulate the flux for each k
	for (int i = 1; i < sys_vars->n_spec; ++i) {
		proc_data->enst_flux_spec[i] += proc_data->enst_flux_spec[i - 1];
		if (i == (int)(sys_vars->kmax_frac * sys_vars->kmax)) {
			// Record the enstrophy flux out of the set C
			proc_data->enst_flux_C[snap] = proc_data->enst_flux_spec[i];
		}
	}
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

	// Initialize arrays
	for (int i = 0; i < Nx; ++i) {
		tmp2 = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			run_data->w_hat[tmp2 + j]   = 0.0 + 0.0 * I;
			run_data->psi_hat[tmp2 + j] = 0.0 + 0.0 * I;
		}
	}

	#if defined(__REAL_STATS) || defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
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

	#if defined(__GRAD_STATS)
	// Allocate Fourier gradient arrays
	proc_data->grad_u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	if (proc_data->grad_u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	proc_data->grad_w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	if (proc_data->grad_w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	// Allocate Real Space gradient arrays
	proc_data->grad_u = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (proc_data->grad_u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}
	proc_data->grad_w = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (proc_data->grad_w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}

	// Initialize the gradient histograms
	for (int i = 0; i < INCR_TYPES; ++i) {
		stats_data->vel_grad[i]  = gsl_histogram_alloc(N_BINS);
		stats_data->vort_grad[i] = gsl_histogram_alloc(N_BINS);
	}
	#endif
	#if defined(__VEL_INC_STATS)
	// Initialize a histogram objects for each increments for each direction	
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			stats_data->w_incr[i][j]   = gsl_histogram_alloc(N_BINS);
			stats_data->vel_incr[i][j] = gsl_histogram_alloc(N_BINS);
		}
	}

	// Set the bin limits for the velocity increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			// Velocity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vel_incr[i][j], -BIN_LIM - 0.5, BIN_LIM + 0.5);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Increments");
				exit(1);
			}
			// Vorticity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_incr[i][j], -BIN_LIM - 0.5, BIN_LIM + 0.5);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Increments");
				exit(1);
			}
		}
	}
	#endif
	#if defined(__STR_FUNC_STATS)
	// Allocate memory for each structure function for each of the increment directions
	int N_max_incr = (int) GSL_MIN(Nx, Ny) / 2;
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 2; j < STR_FUNC_MAX_POW; ++j) {
			stats_data->str_func[i][j - 2] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->str_func[i][j - 2] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Structure Functions");
				exit(1);
			}
			stats_data->str_func_abs[i][j - 2] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->str_func_abs[i][j - 2] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Structure Functions Absolute");
				exit(1);
			}

			// Initialize array
			for (int r = 0; r < N_max_incr; ++r) {
				stats_data->str_func[i][j - 2][r]     = 0.0;
				stats_data->str_func_abs[i][j - 2][r] = 0.0;
			}
		}
	}
	#endif

	// Initialize the arrays
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		tmp2 = i * (Ny_Fourier);
		for (int j = 0; j < Ny; ++j) {
			if (j < Ny_Fourier) {
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 1] = 0.0 + 0.0 * I;
			}
			run_data->w[tmp2 + j] = 0.0;
			run_data->u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
			run_data->u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
		}
	}
	#endif
	
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

	#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
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
	// The enstrophy flux spectrum
	proc_data->enst_flux_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_flux_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}
	// The enstrophy flux out of the set C
	proc_data->enst_flux_C = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if (proc_data->enst_flux_C == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Out of C");
		exit(1);
	}

	// Initialize arrays
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny; ++j) {
			if(j < Ny_Fourier) {
				proc_data->dw_hat_dt[SYS_DIM * (i * Ny_Fourier + j) + 0] = 0.0 + 0.0 * I;
				proc_data->dw_hat_dt[SYS_DIM * (i * Ny_Fourier + j) + 1] = 0.0 + 0.0 * I;
			}
			proc_data->nabla_w[SYS_DIM * (i * Ny + j) + 0] = 0.0;
			proc_data->nabla_w[SYS_DIM * (i * Ny + j) + 1] = 0.0;
			proc_data->nabla_psi[SYS_DIM * (i * Ny + j) + 0] = 0.0;
			proc_data->nabla_psi[SYS_DIM * (i * Ny + j) + 1] = 0.0;
			proc_data->nonlinterm[i * Ny + j] = 0.0;
		}
	}
	for (int i = 0; i < sys_vars->num_snaps; ++i) {
		proc_data->enst_flux_C[i] = 0.0;
	}
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		proc_data->enst_flux_spec[i] = 0.0;
	}
	#endif

	// --------------------------------	
	//  Allocate Phase Sync Data
	// --------------------------------
	#if defined(__SEC_PHASE_SYNC)
	// Get the various kmax variables
	sys_vars->kmax_sqr   = pow(sys_vars->kmax, 2.0);
	sys_vars->kmax_C   	 = (int) ceil(sys_vars->kmax_frac * sys_vars->kmax);
	sys_vars->kmax_C_sqr = pow(sys_vars->kmax_C, 2.0);

	///--------------- Sector Angles
	// Allocate the array of sector angles
	proc_data->theta = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect));
	if (proc_data->theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Sector Angles");
		exit(1);
	}

	///------------------ Number of Triads & Enstrophy Flux
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		///--------------- Number of Triads
		// Allocate memory for the phase order parameter for the triad phases
		proc_data->num_triads[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_sect);
		if (proc_data->num_triads[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}

		///--------------- Enstrophy Flux
		proc_data->enst_flux[i] = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect));
		if (proc_data->enst_flux[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
			exit(1);
		}

		///-------------- Number of Triads and Enstrophy Flux across sectors
		// Allocate memory for the number of triads per sector
		proc_data->num_triads_across_sec[i] = (int** )fftw_malloc(sizeof(int* ) * sys_vars->num_sect);
		if (proc_data->num_triads_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}
		// Allocate memory for the flux of enstrophy across sectors
		proc_data->enst_flux_across_sec[i] = (double** )fftw_malloc(sizeof(double* ) * sys_vars->num_sect);
		if (proc_data->enst_flux_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
			exit(1);
		}	
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			proc_data->num_triads_across_sec[i][l] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_sect);
			if (proc_data->num_triads_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
				exit(1);
			}
			proc_data->enst_flux_across_sec[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
			if (proc_data->enst_flux_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
				exit(1);
			}
		}
	}
	// Allocate memory for the precomputed sector midpoint angle sums -> this is used to determine which sector k2 is
	proc_data->mid_angle_sum = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect) * (sys_vars->num_sect));
	if (proc_data->mid_angle_sum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Midpoint Sector Angles");
		exit(1);
	}
	fftw_complex k1, k3;
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			k3 = 1.0 * cexp(I * proc_data->theta[a]);
			k1 = 1.0 * cexp(I * proc_data->theta[a + l % sys_vars->num_sect]);

			// Compute the midpoint angle of the vector 
			proc_data->mid_angle_sum[a * sys_vars->num_sect + l] = creal(1.0 / (2.0 * I) * (clog(k3 - k1) - clog(conj(k3) - conj(k1))));
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

		///------------- Allocate memory for arrays across sectors
		/// Allocate memory for the phase order parameter for the triad phases
		proc_data->triad_phase_order_across_sec[i] = (fftw_complex** )fftw_malloc(sizeof(fftw_complex*) * sys_vars->num_sect);
		if (proc_data->triad_phase_order_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R_across_sec[i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_sect);
		if (proc_data->triad_R_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi_across_sec[i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_sect);
		if (proc_data->triad_Phi_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
			exit(1);
		}
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			/// Allocate memory for the phase order parameter for the triad phases
			proc_data->triad_phase_order_across_sec[i][l] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
			if (proc_data->triad_phase_order_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
				exit(1);
			}
			// Allocate the array of phase sync per sector for the triad phases
			proc_data->triad_R_across_sec[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
			if (proc_data->triad_R_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
				exit(1);
			}
			// Allocate the array of average phase per sector for the triad phases
			proc_data->triad_Phi_across_sec[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
			if (proc_data->triad_Phi_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
				exit(1);
			}
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
	proc_data->dtheta = M_PI / (double )sys_vars->num_sect;
	for (int i = 0; i < sys_vars->num_sect; ++i) {
		proc_data->theta[i] = -M_PI / 2.0 + i * proc_data->dtheta;
		proc_data->phase_R[i]     = 0.0;
		proc_data->phase_Phi[i]   = 0.0;
		proc_data->phase_order[i] = 0.0 + 0.0 * I;
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			proc_data->num_triads[j][i]        = 0;
			proc_data->enst_flux[j][i]   	   = 0.0;
			proc_data->triad_R[j][i]     	   = 0.0;
			proc_data->triad_Phi[j][i]         = 0.0;
			proc_data->triad_phase_order[j][i] = 0.0 + 0.0 * I;
			for (int k = 0; k < sys_vars->num_sect; ++k) {
				proc_data->num_triads_across_sec[j][i][k]        = 0;
				proc_data->enst_flux_across_sec[j][i][k]   	  	 = 0.0;
				proc_data->triad_R_across_sec[j][i][k]     	  	 = 0.0;
				proc_data->triad_Phi_across_sec[j][i][k]         = 0.0;
				proc_data->triad_phase_order_across_sec[j][i][k] = 0.0 + 0.0 * I;
			}
		}
	}
	#endif

	// --------------------------------	
	//  Allocate Stats Data
	// --------------------------------
	#if defined(__REAL_STATS)
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

	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
	// Initialize Fourier Transforms
	sys_vars->fftw_2d_dft_c2r = fftw_plan_dft_c2r_2d(Nx, Ny, run_data->w_hat, run_data->w, FFTW_ESTIMATE);
	if (sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	#endif
	#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
	sys_vars->fftw_2d_dft_r2c = fftw_plan_dft_r2c_2d(Nx, Ny, run_data->w, run_data->w_hat, FFTW_ESTIMATE);
	if (sys_vars->fftw_2d_dft_r2c == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	#endif

	#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__REAL_STATS) || defined(__GRAD_STATS)
	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_c2r = fftw_plan_many_dft_c2r(SYS_DIM, N_batch, SYS_DIM, run_data->u_hat, NULL, SYS_DIM, 1, run_data->u, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	#endif

	#if defined(__GRAD_STATS)
	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_r2c = fftw_plan_many_dft_r2c(SYS_DIM, N_batch, SYS_DIM, run_data->u, NULL, SYS_DIM, 1, run_data->u_hat, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_batch_r2c == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	#endif
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
	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__GRAD_STATS)
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	#if defined(__STR_FUNC_STATS)
	for (int i = 2; i < STR_FUNC_MAX_POW; ++i) {
		fftw_free(stats_data->str_func[0][i - 2]);
		fftw_free(stats_data->str_func[1][i - 2]);
		fftw_free(stats_data->str_func_abs[0][i - 2]);
		fftw_free(stats_data->str_func_abs[1][i - 2]);
	}
	#endif
	#if defined(__GRAD_STATS)
	fftw_free(proc_data->grad_u_hat);
	fftw_free(proc_data->grad_w_hat);
	fftw_free(proc_data->grad_u);
	fftw_free(proc_data->grad_w);	
	#endif
	#endif
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
	#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
	fftw_free(proc_data->nabla_w);
	fftw_free(proc_data->nabla_psi);
	fftw_free(proc_data->dw_hat_dt);
	fftw_free(proc_data->nonlinterm);
	fftw_free(proc_data->enst_flux_C);
	fftw_free(proc_data->enst_flux_spec);
	#endif
	#if defined(__SEC_PHASE_SYNC)
	fftw_free(proc_data->theta);
	fftw_free(proc_data->k_angle);
	fftw_free(proc_data->k1_angle);
	fftw_free(proc_data->k2_angle);
	fftw_free(proc_data->mid_angle_sum);
	fftw_free(proc_data->k2_angle_neg);
	fftw_free(proc_data->phase_angle);
	fftw_free(proc_data->phase_order);
	fftw_free(proc_data->phase_R);
	fftw_free(proc_data->phase_Phi);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		fftw_free(proc_data->enst_flux[i]);
		fftw_free(proc_data->triad_phase_order[i]);
		fftw_free(proc_data->triad_R[i]);
		fftw_free(proc_data->triad_Phi[i]);
		fftw_free(proc_data->num_triads[i]);
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			fftw_free(proc_data->enst_flux_across_sec[i][l]);
			fftw_free(proc_data->triad_phase_order_across_sec[i][l]);
			fftw_free(proc_data->triad_R_across_sec[i][l]);
			fftw_free(proc_data->triad_Phi_across_sec[i][l]);
			fftw_free(proc_data->num_triads_across_sec[i][l]);
		}
		fftw_free(proc_data->enst_flux_across_sec[i]);
		fftw_free(proc_data->triad_phase_order_across_sec[i]);
		fftw_free(proc_data->triad_R_across_sec[i]);
		fftw_free(proc_data->triad_Phi_across_sec[i]);
		fftw_free(proc_data->num_triads_across_sec[i]);
	}
	#endif
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}
	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	#if defined(__REAL_STATS)
	gsl_histogram_free(stats_data->w_pdf);
	gsl_histogram_free(stats_data->u_pdf);
	#endif
	#if defined(__VEL_INC_STATS)
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			gsl_histogram_free(stats_data->vel_incr[j][i]);
			gsl_histogram_free(stats_data->w_incr[j][i]);
		}	
	}
	#endif
	#if defined(__GRAD_STATS)
	for (int i = 0; i < INCR_TYPES; ++i) {
		gsl_histogram_free(stats_data->vel_grad[i]);
		gsl_histogram_free(stats_data->vort_grad[i]);
	}
	#endif
	#if defined(__SEC_PHASE_SYNC)
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
	// Destroy FFTW plans
	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	#endif
	#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
	fftw_destroy_plan(sys_vars->fftw_2d_dft_r2c);
	#endif
	#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__REAL_STATS) || defined(__GRAD_STATS)
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);
	#endif
	#if defined(__GRAD_STATS)
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_r2c);
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
