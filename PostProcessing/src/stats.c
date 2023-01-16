/**
* @file stats.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the stats functions for the pseudospectral solver data
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
#include "utils.h"


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
	const long int Ny = sys_vars->N[0];
	const long int Nx = sys_vars->N[1];
	#if defined(__VEL_INC_STATS) || defined(__VORT_INC_STATS) || defined(__VORT_STR_FUNC_STATS) || defined(__VORT_RADIAL_STR_FUNC_STATS) || defined(__VEL_STR_FUNC_STATS) || defined(__VEL_GRAD_STATS) || defined(__VORT_GRAD_STATS)  || defined(__MIXED_VEL_STR_FUNC) || defined(__MIXED_VORT_STR_FUNC)
	int r;
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	double vel_long_increment, vel_trans_increment, mixed_vel_increment;
	double vel_long_increment_abs, vel_trans_increment_abs;
	double vort_long_increment, vort_trans_increment, mixed_vort_increment;
	double vort_long_increment_abs, vort_trans_increment_abs;
	int N_max_incr = (int) (GSL_MIN(Ny, Nx) / 2);
	double norm_fac = 1.0 / (Ny * Nx);
	double radial_pow[STR_FUNC_MAX_POW] = {0.1, 0.5, 1.0, 1.5, 2.0, 2.5};
	double delta_x = 2.0 * M_PI / Nx;
	double delta_y = 2.0 * M_PI / Ny;
	#endif
	#if defined(__VORT_RADIAL_STR_FUNC_STATS)
	int tmp_r, indx_r;
	double vort_rad_increment, vort_rad_increment_abs;
	#endif

	// --------------------------------
	// Get In-Time Histogram Limits
	// --------------------------------
	#if defined(__REAL_STATS)
	// Get min and max data for histogram limits
	double w_max = 0.0;
	double w_min = 1e8;
	gsl_stats_minmax(&w_min, &w_max, run_data->w, 1, Ny * Nx);

	double u_max = 0.0; 
	double u_min = 1e8;
	gsl_stats_minmax(&u_min, &u_max, run_data->u, 1, Ny * Nx * SYS_DIM);
	if (fabs(u_min) > u_max) {
		u_max = fabs(u_min);
	}

	// Set histogram ranges for the current snapshot
	gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_hist, w_min - 0.5, w_max + 0.5);
	if (gsl_status != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
		exit(1);
	}
	gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_hist, 0.0 - 0.1, u_max + 0.5);
	if (gsl_status != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
		exit(1);
	}
	#endif


	// --------------------------------
	// Compute Gradients
	// --------------------------------
	#if defined(__VEL_GRAD_STATS) || defined(__VORT_GRAD_STATS)
	// Compute the graident in Fourier space
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			#if defined(__VORT_GRAD_STATS)
			// Compute the gradients of the vorticity
			proc_data->grad_w_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * run_data->w_hat[indx];
			proc_data->grad_w_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * run_data->w_hat[indx];
			#endif

			#if defined(__VORT_GRAD_STATS)
			// Compute the gradients of the vorticity
			proc_data->grad_u_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 0];
			proc_data->grad_u_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 1];
			#endif
		}
	}

	// Perform inverse transform and normalize to get the gradients in real space - no need to presave grad_w_hat & grad_u_hat, wont be used again
	#if defined(__VORT_GRAD_STATS)
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_w_hat, proc_data->grad_w);
	#endif
	#if defined(__VORT_GRAD_STATS)
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_u_hat, proc_data->grad_u);
	#endif
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Normalize the gradients
			#if defined(__VORT_GRAD_STATS)
			proc_data->grad_w[SYS_DIM * indx + 0] *= 1.0; // norm_fac;
			proc_data->grad_w[SYS_DIM * indx + 1] *= 1.0; // norm_fac;
			#endif
			#if defined(__VEL_GRAD_STATS)
			proc_data->grad_u[SYS_DIM * indx + 0] *= 1.0; // norm_fac;
			proc_data->grad_u[SYS_DIM * indx + 1] *= 1.0; // norm_fac;
			#endif
		}
	}
	#endif

	// --------------------------------
	// Update Histogram Counts
	// --------------------------------
	// Update histograms with the data from the current snapshot
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx;
		for (int j = 0; j < Nx; ++j) {
			indx = tmp + j;

			#if defined(__REAL_STATS)
			///-------------------------------- Velocity & Vorticity Fields
			// Add current values to appropriate bins
			gsl_status = gsl_histogram_increment(stats_data->w_hist, run_data->w[indx]);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s, gsl_status, run_data->w[indx]);
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_hist, fabs(run_data->u[SYS_DIM * indx + 0]));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s, gsl_status, fabs(run_data->u[SYS_DIM * indx + 0]));
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_hist, fabs(run_data->u[SYS_DIM * indx + 1]));
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s, gsl_status, fabs(run_data->u[SYS_DIM * indx + 1]));
				exit(1);
			}

			// Update stats accumulators
			gsl_rstat_add(run_data->w[indx], stats_data->w_stats);			
			for (int i = 0; i < SYS_DIM + 1; ++i) {
				if (i < SYS_DIM) {
					// Add to the individual directions accumulators
					gsl_rstat_add(run_data->u[SYS_DIM * indx + i], stats_data->u_stats[i]);
				}
				else {
					// Add to the combined accumulator
					for (int j = 0; j < SYS_DIM; ++j) {
						gsl_rstat_add(run_data->u[SYS_DIM * indx + j], stats_data->u_stats[SYS_DIM]);	
					}
				}
			}
			#endif

			///-------------------------------- Gradient Fields
			#if defined(__VEL_GRAD_STATS)
			// Update Velocity gradient histograms
			gsl_status = gsl_histogram_increment(stats_data->u_grad_hist[0], proc_data->grad_u[SYS_DIM * indx + 0] * delta_x);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "X Velocity Gradient", s, gsl_status, proc_data->grad_u[SYS_DIM * indx + 0] * delta_x);
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->u_grad_hist[1], proc_data->grad_u[SYS_DIM * indx + 1] * delta_y);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Y Velocity Gradient", s, gsl_status, proc_data->grad_u[SYS_DIM * indx + 1] * delta_y);
				exit(1);
			}
			#endif

			#if defined(__VORT_GRAD_STATS)
			// Update Vorticity gradient histograms
			gsl_status = gsl_histogram_increment(stats_data->w_grad_hist[0], proc_data->grad_w[SYS_DIM * indx + 0] * delta_x);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "X Vorticity Gradient", s, gsl_status, proc_data->grad_w[SYS_DIM * indx + 0] * delta_x);
				exit(1);
			}
			gsl_status = gsl_histogram_increment(stats_data->w_grad_hist[1], proc_data->grad_w[SYS_DIM * indx + 1] * delta_y);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Y Vorticity Gradient", s, gsl_status, proc_data->grad_w[SYS_DIM * indx + 1] * delta_y);
				exit(1);
			}
			#endif

			///-------------------------------- Velocity & Vorticity Incrments
			#if defined(__VEL_INC_STATS) || defined(__VORT_INC_STATS)
			// Compute velocity increments and update histograms
			for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
				// Get the current increment
				r = stats_data->increments[r_indx];

				#if defined(__VEL_INC_STATS)
				//------------- Get the longitudinal and transverse Velocity increments
				vel_long_increment  = run_data->u[SYS_DIM * (i * Nx + (j + r) % Nx) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0];
				vel_trans_increment = run_data->u[SYS_DIM * (i * Nx + (j + r) % Nx) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1];

				// Update the histograms
				gsl_status = gsl_histogram_increment(stats_data->u_incr_hist[0][r_indx], vel_long_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment", s, gsl_status, vel_long_increment);
					exit(1);
				}
				gsl_status = gsl_histogram_increment(stats_data->u_incr_hist[1][r_indx], vel_trans_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Velocity Increment", s, gsl_status, vel_trans_increment);
					exit(1);
				}
				#endif

				#if defined(__VORT_INC_STATS)
				//------------- Get the longitudinal and transverse Vorticity increments
				vort_long_increment  = run_data->w[i * Nx + (j + r) % Nx] - run_data->w[i * Nx + j];
				vort_trans_increment = run_data->w[((i + r) % Ny)* Nx + j] - run_data->w[i * Nx + j];

				// Update the histograms
				gsl_status = gsl_histogram_increment(stats_data->w_incr_hist[0][r_indx], vort_long_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Vorticity Increment", s, gsl_status, vort_long_increment);
					exit(1);
				}
				gsl_status = gsl_histogram_increment(stats_data->w_incr_hist[1][r_indx], vort_trans_increment);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Vorticity Increment", s, gsl_status, vort_trans_increment);
					exit(1);
				}			
				#endif
			}
			#endif
		}
	}

	// --------------------------------
	// Compute Structure Functions
	// --------------------------------
	#if defined(__VEL_STR_FUNC_STATS) || defined(__VORT_STR_FUNC_STATS) || defined(__VORT_RAD_STR_FUNC_STATS)
	#pragma omp parallel num_threads(sys_vars->num_threads) shared(run_data, stats_data) private(tmp, indx, tmp_r, indx_r)
	{
		#pragma omp single 
		{	
			for (int p = 1; p <= STR_FUNC_MAX_POW; ++p) {

				#pragma omp taskloop reduction (+:vel_long_increment, vel_long_increment_abs, vel_trans_increment, vel_trans_increment_abs, vort_long_increment, vort_long_increment_abs, vort_trans_increment, vort_trans_increment_abs, mixed_vel_increment, mixed_vort_increment) grainsize(N_max_incr / sys_vars->num_threads)
				for (int r_inc = 1; r_inc <= N_max_incr; ++r_inc) {
					
					// Initialize velocity increments
					#if defined(__VEL_STR_FUNC_STATS)
					vel_long_increment      = 0.0;
					vel_trans_increment     = 0.0;
					vel_long_increment_abs  = 0.0;
					vel_trans_increment_abs = 0.0;
					#endif
					// Initialize vorticit increments
					#if defined(__VORT_STR_FUNC_STATS)
					vort_long_increment      = 0.0;
					vort_trans_increment     = 0.0;
					vort_long_increment_abs  = 0.0;
					vort_trans_increment_abs = 0.0;
					#endif
					if (p == 3) {
						#if defined(__MIXED_VEL_STR_FUNC_STATS)
						mixed_vel_increment = 0.0;
						#endif
						#if defined(__MIXED_VORT_STR_FUNC_STATS)
						mixed_vort_increment = 0.0;
						#endif	
					}

					// Loop over space and average 
					for (int i = 0; i < Ny; ++i) {
						for (int j = 0; j < Nx; ++j) {				
							// Get velocity increments
							#if defined(__VEL_STR_FUNC_STATS)
							vel_long_increment      += pow(sgn(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0]), 2.0 * radial_pow[p - 1]) * pow(fabs(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0]), 2.0 * radial_pow[p - 1]);
							vel_trans_increment     += pow(sgn(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1]), 2.0 * radial_pow[p - 1]) * pow(fabs(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1]), 2.0 * radial_pow[p - 1]);
							vel_long_increment_abs  += pow(fabs(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0]), 2.0 * radial_pow[p - 1]);
							vel_trans_increment_abs += pow(fabs(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1]), 2.0 * radial_pow[p - 1]);
							#endif
							// Get vorticity increments
							#if defined(__VORT_STR_FUNC_STATS)
							vort_long_increment      += pow(sgn(run_data->w[i * Nx + ((j + r_inc) % Nx)] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]) * pow(fabs(run_data->w[i * Nx + ((j + r_inc) % Nx)] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]);
							vort_trans_increment     += pow(sgn(run_data->w[((i + r_inc) % Ny) * Nx + j] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]) * pow(fabs(run_data->w[((i + r_inc) % Ny) * Nx + j] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]);
							vort_long_increment_abs  += pow(fabs(run_data->w[i * Nx + ((j + r_inc) % Nx)] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]);
							vort_trans_increment_abs += pow(fabs(run_data->w[((i + r_inc) % Ny) * Nx + j] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]);
							#endif
							if (p == 3) {
								#if defined(__MIXED_VEL_STR_FUNC)
								mixed_vel_increment += (run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0]) * pow(run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1], 2.0);
								#endif
								#if defined(__MIXED_VORT_STR_FUNC)
								mixed_vort_increment += (run_data->u[SYS_DIM * (i * Nx + ((j + r_inc) % Nx)) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0]) * pow(run_data->w[i * Nx + ((j + r_inc) % Nx)] - run_data->w[i * Nx + j], 2.0);
								#endif	
							}
						}
					}
					// Compute velocity str function - normalize here
					#if defined(__VEL_STR_FUNC_STATS)
					stats_data->u_str_func[0][p - 1][r_inc - 1]     += vel_long_increment * norm_fac;	
					stats_data->u_str_func[1][p - 1][r_inc - 1]     += vel_trans_increment * norm_fac;
					stats_data->u_str_func_abs[0][p - 1][r_inc - 1] += vel_long_increment_abs * norm_fac;	
					stats_data->u_str_func_abs[1][p - 1][r_inc - 1] += vel_trans_increment_abs * norm_fac;
					#endif
					// Compute vorticity str function - normalize here
					#if defined(__VORT_STR_FUNC_STATS)
					stats_data->w_str_func[0][p - 1][r_inc - 1]     += vort_long_increment * norm_fac;	
					stats_data->w_str_func[1][p - 1][r_inc - 1]     += vort_trans_increment * norm_fac;
					stats_data->w_str_func_abs[0][p - 1][r_inc - 1] += vort_long_increment_abs * norm_fac;	
					stats_data->w_str_func_abs[1][p - 1][r_inc - 1] += vort_trans_increment_abs * norm_fac;
					#endif
					if (p == 3) {
						#if defined(__MIXED_VEL_STR_FUNC)
						stats_data->vel_mixed_str_func[r_inc - 1] += mixed_vel_increment * norm_fac;
						#endif
						#if defined(__MIXED_VORT_STR_FUNC)
						stats_data->vel_mixed_str_func[r_inc - 1] += mixed_vort_increment * norm_fac;
						#endif
					}
				}

				#if defined(__VORT_RADIAL_STR_FUNC_STATS)
				// Compute the radial structure functions
				#pragma omp taskloop reduction (+:vort_rad_increment, vort_rad_increment_abs) collapse(2) grainsize(N_max_incr * N_max_incr / sys_vars->num_threads)
				for (int r_y = 1; r_y <= N_max_incr; ++r_y) {
					for (int r_x = 1; r_x <= N_max_incr; ++r_x) {
						tmp_r = (r_y - 1) * N_max_incr;
						indx_r = tmp_r + (r_x - 1);

						vort_rad_increment     = 0.0;
						vort_rad_increment_abs = 0.0;

						// Loop over space and average 
						for (int i = 0; i < Ny; ++i) {
							for (int j = 0; j < Nx; ++j) {
								vort_rad_increment     += pow(sgn(run_data->w[((i + r_y) % Ny) * Nx + ((j + r_x) % Nx)] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]) * pow(fabs(run_data->w[((i + r_y) % Ny) * Nx + ((j + r_x) % Nx)] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]);
								vort_rad_increment_abs += pow(fabs(run_data->w[((i + r_y) % Ny) * Nx + ((j + r_x) % Nx)] - run_data->w[i * Nx + j]), 2.0 * radial_pow[p - 1]);
							}
						}

						// Update radial vorticity the structure funcitons
						stats_data->w_radial_str_func[p - 1][indx_r]     += vort_rad_increment * norm_fac;	
						stats_data->w_radial_str_func_abs[p - 1][indx_r] += vort_rad_increment_abs * norm_fac;	
					}
				}
				#endif
			}
		}
	}
	#endif	
}
/**
 * Allocates memory and initializes GSL stats objects for the stats computations
 * @param N Array containing the size of each dimension
 */
void AllocateStatsMemory(const long int* N) {

	// Initialize variables
	const long int Ny = N[0];
	const long int Nx = N[1];
	const long int Nx_Fourier = N[1] / 2 + 1;
	stats_data->N_max_incr = (int) (GSL_MIN(Ny, Nx) / 2);
	int N_max_incr = stats_data->N_max_incr;


	// Set up increments array
	stats_data->increments = (int* )fftw_malloc(sizeof(int) * NUM_INCR);
	int increment[NUM_INCR] = {1, 2, 4, 16, N_max_incr};
	memcpy(stats_data->increments, increment, sizeof(increment));

	// --------------------------------	
	//  Initialize Real Space Stats
	// --------------------------------
	#if defined(__REAL_STATS)
	// Allocate vorticity histograms
	stats_data->w_hist = gsl_histogram_alloc(N_BINS);
	if (stats_data->w_hist == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Histogram");
		exit(1);
	}	

	// Allocate velocity histograms
	stats_data->u_hist = gsl_histogram_alloc(N_BINS);
	if (stats_data->u_hist == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity In Time Histogram");
		exit(1);
	}

	// Initialize the running stats for the velocity and vorticity fields
	stats_data->w_stats = gsl_rstat_alloc();
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		stats_data->u_stats[i] = gsl_rstat_alloc();
	}
	#endif
	
	// --------------------------------
	//  Initialize Graient Stats
	// --------------------------------
	#if defined(__VEL_GRAD_STATS)
	// Allocate Fourier gradient arrays
	proc_data->grad_u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier * SYS_DIM);
	if (proc_data->grad_u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	// Allocate Real Space gradient arrays
	proc_data->grad_u = (double* )fftw_malloc(sizeof(double) * Ny * Nx * SYS_DIM);
	if (proc_data->grad_u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Gradient");
		exit(1);
	}

	for (int i = 0; i < SYS_DIM + 1; ++i) {
		// Initialize the gradient histograms
		stats_data->u_grad_hist[i] = gsl_histogram_alloc(N_BINS);

		// Initialize the running stats for the gradients
		stats_data->u_grad_stats[i] = gsl_rstat_alloc();
	}
	#endif

	#if defined(__VORT_GRAD_STATS)
	proc_data->grad_w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier * SYS_DIM);
	if (proc_data->grad_w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	proc_data->grad_w = (double* )fftw_malloc(sizeof(double) * Ny * Nx * SYS_DIM);
	if (proc_data->grad_w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Gradient");
		exit(1);
	}
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		// Initialize the gradient histograms
		stats_data->w_grad_hist[i] = gsl_histogram_alloc(N_BINS);

		// Initialize the running stats for the gradients
		stats_data->w_grad_stats[i] = gsl_rstat_alloc();
	}
	#endif


	// --------------------------------	
	//  Initialize Increment Stats
	// --------------------------------
	// Initialize GSL objects for the increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			// Initialize the vorticity increment objects	
			#if defined(__VEL_INC_STATS)
			stats_data->w_incr_hist[i][j]  = gsl_histogram_alloc(N_BINS);
			stats_data->w_incr_stats[i][j] = gsl_rstat_alloc();
			#endif
			
			// Initialize the velocity incrment objects
			#if defined(__VORT_INC_STATS)
			stats_data->u_incr_hist[i][j]  = gsl_histogram_alloc(N_BINS);
			stats_data->u_incr_stats[i][j] = gsl_rstat_alloc();
			#endif
		}
	}


	// --------------------------------	
	//  Initialize Str Func Stats
	// --------------------------------
	#if defined(__VEL_STR_FUNC_STATS) || defined(__VORT_STR_FUNC_STATS)	|| defined(__VORT_RAD_STR_FUNC_STATS)
	// Allocate memory for each structure function for each of the increment directions
	for (int p = 1; p <= STR_FUNC_MAX_POW; ++p) {
		for (int i = 0; i < INCR_TYPES; ++i) {

			///----------------------------------- Velocity Structure functions
			#if defined(__VEL_STR_FUNC_STATS)
			stats_data->u_str_func[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->u_str_func[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Structure Functions");
				exit(1);
			}
			stats_data->u_str_func_abs[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->u_str_func_abs[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Structure Functions Absolute");
				exit(1);
			}

			// Initialize array
			for (int r = 0; r < N_max_incr; ++r) {
				stats_data->u_str_func[i][p - 1][r]     = 0.0;
				stats_data->u_str_func_abs[i][p - 1][r] = 0.0;
			}
			#endif

			///----------------------------------- Vorticity Structure functions
			#if defined(__VORT_STR_FUNC_STATS)
			stats_data->w_str_func[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->w_str_func[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Structure Functions");
				exit(1);
			}
			stats_data->w_str_func_abs[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->w_str_func_abs[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Structure Functions Absolute");
				exit(1);
			}

			// Initialize array
			for (int r = 0; r < N_max_incr; ++r) {
				stats_data->w_str_func[i][p - 1][r]     = 0.0;
				stats_data->w_str_func_abs[i][p - 1][r] = 0.0;
			}
			#endif

		}
		///----------------------------------- Radial Vorticity Structure functions
		#if defined(__VORT_RADIAL_STR_FUNC_STATS)
		stats_data->w_radial_str_func[p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr) * (N_max_incr));
		if (stats_data->w_radial_str_func[p - 1] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Radial Vorticity Structure Functions");
			exit(1);
		}
		stats_data->w_radial_str_func_abs[p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr) * (N_max_incr));
		if (stats_data->w_radial_str_func_abs[p - 1] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Radial Vorticity Structure Functions Absolute");
			exit(1);
		}

		// Initialize array
		for (int r = 0; r < N_max_incr * N_max_incr; ++r) {
			stats_data->w_radial_str_func[p - 1][r]     = 0.0;
			stats_data->w_radial_str_func_abs[p - 1][r] = 0.0;
		}
		#endif
	}
	#endif

	#if defined(__MIXED_VEL_STR_FUNC_STATS)
	stats_data->mxd_u_str_func = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
	if (stats_data->mxd_u_str_func == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Mixed Velocity Structure Functions");
		exit(1);
	}
	for (int r = 0; r < N_max_incr; ++r) {
		stats_data->mxd_u_str_func[r] = 0.0;
	}
	#endif
	#if defined(__MIXED_VORT_STR_FUNC_STATS)
	stats_data->mxd_w_str_func = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
	if (stats_data->mxd_w_str_func == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Mixed Vorticity Structure Functions");
		exit(1);
	}
	for (int r = 0; r < N_max_incr; ++r) {
		stats_data->mxd_w_str_func[r] = 0.0;
	}
	#endif
}
/**
 * Wrapper function to write All the stats data to file at the end of postprocessing
 */
void WriteStatsToFile(void) {

    // Initialize variables
    int N_max_incr = (int ) GSL_MIN(sys_vars->N[0], sys_vars->N[1]) / 2;
    herr_t status;
    static const hsize_t Dims1D = 1;
    hsize_t dset_dims_1d[Dims1D];        // array to hold dims of the dataset to be created
    static const hsize_t Dims2D = 2;
    hsize_t dset_dims_2d[Dims2D];
    static const hsize_t Dims3D = 3;
    hsize_t dset_dims_3d[Dims3D];


    // -------------------------------
    // Write Real Space Stats
    // -------------------------------
	///----------------------------------- Write the Real Space Statistics
	#if defined(__REAL_STATS)
    double w_field_stats[6];
    double u_field_stats[SYS_DIM + 1][6];
    for (int i = 0; i < SYS_DIM + 1; ++i) {
    	// Velocity stats
    	u_field_stats[i][0] = gsl_rstat_min(stats_data->u_stats[i]);
    	u_field_stats[i][1] = gsl_rstat_max(stats_data->u_stats[i]);
    	u_field_stats[i][2] = gsl_rstat_mean(stats_data->u_stats[i]);
    	u_field_stats[i][3] = gsl_rstat_sd(stats_data->u_stats[i]);
    	u_field_stats[i][4] = gsl_rstat_skew(stats_data->u_stats[i]);
    	u_field_stats[i][5] = gsl_rstat_kurtosis(stats_data->u_stats[i]);
    	// Vorticity stats
    	w_field_stats[0] = gsl_rstat_min(stats_data->w_stats);
    	w_field_stats[1] = gsl_rstat_max(stats_data->w_stats);
    	w_field_stats[2] = gsl_rstat_mean(stats_data->w_stats);
    	w_field_stats[3] = gsl_rstat_sd(stats_data->w_stats);
    	w_field_stats[4] = gsl_rstat_skew(stats_data->w_stats);
    	w_field_stats[5] = gsl_rstat_kurtosis(stats_data->w_stats);
    }

    dset_dims_2d[0] = SYS_DIM + 1;
   	dset_dims_2d[1] = 6;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityStats", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, u_field_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Stats");
        exit(1);
    }
    dset_dims_1d[0] = 6;
    status = H5LTmake_dataset(file_info->output_file_handle, "VorticityStats", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, w_field_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Stats");
        exit(1);
    }
    #endif


    // -------------------------------
    // Write Increment Stats
    // -------------------------------
    ///----------------------------------- Write the Velocity Increments
	#if defined(__VEL_INC_STATS)
	// Allocate temporary memory to record the histogram data contiguously
    double* vel_inc_range  = (double*) fftw_malloc(sizeof(double) * NUM_INCR * (N_BINS + 1));
    double* vel_inc_counts = (double*) fftw_malloc(sizeof(double) * NUM_INCR * (N_BINS));

    //-------------- Write the longitudinal increments
   	for (int r = 0; r < NUM_INCR; ++r) {
   		for (int b = 0; b < N_BINS + 1; ++b) {
	   		vel_inc_range[r * (N_BINS + 1) + b] = stats_data->u_incr_hist[0][r]->range[b];
	   		if (b < N_BINS) {
	   			vel_inc_counts[r * (N_BINS) + b] = stats_data->u_incr_hist[0][r]->bin[b];	   			
	   		}
   		}
   	}
   	dset_dims_2d[0] = NUM_INCR;
   	dset_dims_2d[1] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "LongitudinalVelIncrements_BinRanges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_inc_range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Longitudinal Velocity Increment PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_2d[1] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "LongitudinalVelIncrements_BinCounts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_inc_counts);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Longitudinal Velocity Increment PDF Bin Counts");
        exit(1);
    }
    
    //--------------- Write the transverse increments
    for (int r = 0; r < NUM_INCR; ++r) {
   		for (int b = 0; b < N_BINS + 1; ++b) {
	   		vel_inc_range[r * (N_BINS + 1) + b] = stats_data->u_incr_hist[1][r]->range[b];
	   		if (b < N_BINS) {
	   			vel_inc_counts[r * (N_BINS) + b] = stats_data->u_incr_hist[1][r]->bin[b];	   			
	   		}
   		}
   	}
   	dset_dims_2d[1] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "TransverseVelIncrements_BinRanges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_inc_range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Transverse Velocity Increment PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_2d[1] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "TransverseVelIncrements_BinCounts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_inc_counts);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Transverse Velocity Increment PDF Bin Counts");
        exit(1);
    }

    // Free temp memory
    fftw_free(vel_inc_range);
    fftw_free(vel_inc_counts);
    #endif


	#if defined(__VORT_INC_STATS)
	// Allocate temporary memory to record the histogram data contiguously
    double* vort_inc_range  = (double*) fftw_malloc(sizeof(double) * NUM_INCR * (N_BINS + 1));
    double* vort_inc_counts = (double*) fftw_malloc(sizeof(double) * NUM_INCR * (N_BINS));

    //-------------- Write the longitudinal increments
    for (int r = 0; r < NUM_INCR; ++r) {
   		for (int b = 0; b < N_BINS + 1; ++b) {
	   		vort_inc_range[r * (N_BINS + 1) + b] = stats_data->w_incr_hist[0][r]->range[b];
	   		if (b < N_BINS) {
	   			vort_inc_counts[r * (N_BINS) + b] = stats_data->w_incr_hist[0][r]->bin[b];	   			
	   		}
   		}
   	}
   	dset_dims_2d[0] = NUM_INCR;
   	dset_dims_2d[1] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "LongitudinalVortIncrements_BinRanges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_inc_range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Longitudinal Vorticity Increment PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_2d[1] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "LongitudinalVortIncrements_BinCounts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_inc_counts);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Longitudinal Vorticity Increment PDF Bin Counts");
        exit(1);
    }

    //-------------- Write the Transverse increments
    for (int r = 0; r < NUM_INCR; ++r) {
   		for (int b = 0; b < N_BINS + 1; ++b) {
	   		vort_inc_range[r * (N_BINS + 1) + b] = stats_data->w_incr_hist[1][r]->range[b];
	   		if (b < N_BINS) {
	   			vort_inc_counts[r * (N_BINS) + b] = stats_data->w_incr_hist[1][r]->bin[b];	   			
	   		}
   		}
   	}
   	dset_dims_2d[1] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "TransverseVortIncrements_BinRanges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_inc_range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Transverse Vorticity Increment PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_2d[1] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "TransverseVortIncrements_BinCounts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_inc_counts);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Transverse Vorticity Increment PDF Bin Counts");
        exit(1);
    }
    
    // Free temporary memory
    fftw_free(vort_inc_range);
    fftw_free(vort_inc_counts);
    #endif

    //--------------- Write increment statistics
    #if defined(__VEL_INC_STATS)
    double vel_incr_stats[INCR_TYPES][NUM_INCR][6];
    #endif
    #if defined(__VORT_INC_STATS)
    double vort_incr_stats[INCR_TYPES][NUM_INCR][6];
    #endif
    for (int i = 0; i < INCR_TYPES; ++i) {
    	for (int j = 0; j < NUM_INCR; ++j) {
    		#if defined(__VEL_INC_STATS)
			// Velocity increment stats
			vel_incr_stats[i][j][0] = gsl_rstat_min(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][1] = gsl_rstat_max(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][2] = gsl_rstat_mean(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][3] = gsl_rstat_sd(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][4] = gsl_rstat_skew(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][5] = gsl_rstat_kurtosis(stats_data->u_incr_stats[i][j]);
			#endif

    		#if defined(__VORT_INC_STATS)
			// Vorticity increment stats
			vort_incr_stats[i][j][0] = gsl_rstat_min(stats_data->w_incr_stats[i][j]);
			vort_incr_stats[i][j][1] = gsl_rstat_max(stats_data->w_incr_stats[i][j]);
			vort_incr_stats[i][j][2] = gsl_rstat_mean(stats_data->w_incr_stats[i][j]);
			vort_incr_stats[i][j][3] = gsl_rstat_sd(stats_data->w_incr_stats[i][j]);
			vort_incr_stats[i][j][4] = gsl_rstat_skew(stats_data->w_incr_stats[i][j]);
			vort_incr_stats[i][j][5] = gsl_rstat_kurtosis(stats_data->w_incr_stats[i][j]);
			#endif
	    }
    }
	#if defined(__VEL_INC_STATS)
    dset_dims_3d[0] = INCR_TYPES;
   	dset_dims_3d[1] = NUM_INCR;
   	dset_dims_3d[2] = 6;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityIncrementStats", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, vel_incr_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Increment Stats");
        exit(1);
    }
    #endif
	#if defined(__VORT_INC_STATS)
    dset_dims_3d[0] = INCR_TYPES;
   	dset_dims_3d[1] = NUM_INCR;
   	dset_dims_3d[2] = 6;
    status = H5LTmake_dataset(file_info->output_file_handle, "VorticityIncrementStats", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, vort_incr_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Increment Stats");
        exit(1);
    }
	#endif



    // -------------------------------
    // Write Gradient Stats
    // -------------------------------
    ///----------------------------------- Velocity Gradient Statistics
	#if defined(__VEL_GRAD_STATS)
	//----------------- Write the x gradients
   	dset_dims_1d[0] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityGradient_x_BinRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_grad_hist[0]->range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Gradient X PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_1d[0] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityGradient_x_BinCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_grad_hist[0]->bin);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Gradient X PDF Bin Counts");
        exit(1);
    }

    //--------------- Write the y gradients
   	dset_dims_1d[0] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityGradient_y_BinRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_grad_hist[1]->range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Gradient Y PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_1d[0] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityGradient_y_BinCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_grad_hist[1]->bin);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Gradient Y PDF Bin Counts");
        exit(1);
    }
    #endif

    ///----------------------------------- Vorticity Gradient Statistics
	#if defined(__VORT_GRAD_STATS)
	//----------------- Write the x gradients
   	dset_dims_1d[0] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VorticityGradient_x_BinRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_grad_hist[0]->range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Gradient X PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_1d[0] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "VorticityGradient_x_BinCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_grad_hist[0]->bin);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Gradient X PDF Bin Counts");
        exit(1);
    }

    //--------------- Write the y gradients
   	dset_dims_1d[0] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VorticityGradient_y_BinRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_grad_hist[1]->range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Gradient Y PDF Bin Ranges");
        exit(1);
    }		
	dset_dims_1d[0] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "VorticityGradient_y_BinCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_grad_hist[1]->bin);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Gradient Y PDF Bin Counts");
        exit(1);
    }
    #endif

    //--------------- Write the gradient statistics
	#if defined(__VEL_GRAD_STATS)
    double grad_u_stats[SYS_DIM + 1][6];
   	#endif
	#if defined(__VORT_GRAD_STATS)
    double grad_w_stats[SYS_DIM + 1][6];
   	#endif
    for (int i = 0; i < SYS_DIM + 1; ++i) {

		#if defined(__VEL_GRAD_STATS)
    	// Velocity gradient stats
    	grad_u_stats[i][0] = gsl_rstat_min(stats_data->u_grad_stats[i]);
    	grad_u_stats[i][1] = gsl_rstat_max(stats_data->u_grad_stats[i]);
    	grad_u_stats[i][2] = gsl_rstat_mean(stats_data->u_grad_stats[i]);
    	grad_u_stats[i][3] = gsl_rstat_sd(stats_data->u_grad_stats[i]);
    	grad_u_stats[i][4] = gsl_rstat_skew(stats_data->u_grad_stats[i]);
    	grad_u_stats[i][5] = gsl_rstat_kurtosis(stats_data->u_grad_stats[i]);
    	#endif
    
    	#if defined(__VORT_GRAD_STATS)
		// Vorticity gradient stats
    	grad_w_stats[i][0] = gsl_rstat_min(stats_data->w_grad_stats[i]);
    	grad_w_stats[i][1] = gsl_rstat_max(stats_data->w_grad_stats[i]);
    	grad_w_stats[i][2] = gsl_rstat_mean(stats_data->w_grad_stats[i]);
    	grad_w_stats[i][3] = gsl_rstat_sd(stats_data->w_grad_stats[i]);
    	grad_w_stats[i][4] = gsl_rstat_skew(stats_data->w_grad_stats[i]);
    	grad_w_stats[i][5] = gsl_rstat_kurtosis(stats_data->w_grad_stats[i]);
    	#endif
    }
	#if defined(__VEL_GRAD_STATS)
    dset_dims_2d[0] = SYS_DIM + 1;
   	dset_dims_2d[1] = 6;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityGradientStats", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, grad_u_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Gradient Stats");
        exit(1);
    }
    #endif
    
	#if defined(__VORT_GRAD_STATS)
    dset_dims_2d[0] = SYS_DIM + 1;
    dset_dims_2d[1] = 6;
    status = H5LTmake_dataset(file_info->output_file_handle, "VorticityGradientStats", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, grad_w_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Gradient Stats");
        exit(1);
    }
	#endif


    // -------------------------------
    // Write Structure Function Stats
    // -------------------------------
	///----------------------------------- Write the Velocity Structure Functions
    #if defined(__VEL_STR_FUNC_STATS)
    // Allocate temporary memory to record the histogram data contiguously
    double* vel_str_funcs = (double*) fftw_malloc(sizeof(double) * STR_FUNC_MAX_POW * (N_max_incr));

    //----------------------- Write the longitudinal structure functions
    // Normal Structure functions
   	for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vel_str_funcs[p * (N_max_incr) + r] = stats_data->u_str_func[0][p][r] / sys_vars->num_snaps;
   		}
   	}
   	dset_dims_2d[0] = STR_FUNC_MAX_POW;
   	dset_dims_2d[1] = N_max_incr;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityLongitudinalStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Longitudinal Structure Functions");
        exit(1);
    }			
    // Absolute Structure functions
   	for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vel_str_funcs[p * (N_max_incr) + r] = stats_data->u_str_func_abs[0][p][r] / sys_vars->num_snaps;
   		}
   	}
	status = H5LTmake_dataset(file_info->output_file_handle, "AbsoluteVelocityLongitudinalStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Absolute Velocity Longitudinal Structure Functions");
        exit(1);
    }	

    //----------------------- Write the transverse structure functions
    // Normal structure functions
    for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vel_str_funcs[p * (N_max_incr) + r] = stats_data->u_str_func[1][p][r] / sys_vars->num_snaps;
   		}
   	}
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityTransverseStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Velocity Transverse Structure Funcitons");
        exit(1);
    }		
    // Absolute structure functions
    for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vel_str_funcs[p * (N_max_incr) + r] = stats_data->u_str_func_abs[1][p][r] / sys_vars->num_snaps;
   		}
   	}
   	status = H5LTmake_dataset(file_info->output_file_handle, "AbsoluteVelocityTransverseStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Absolute Velocity Transverse Structure Funcitons");
        exit(1);
    }		
	
    // Free temporary memory
    fftw_free(vel_str_funcs);
    #endif

	///----------------------------------- Write the Velocity Structure Functions
    #if defined(__VORT_STR_FUNC_STATS)
    // Allocate temporary memory to record the histogram data contiguously
    double* vort_str_funcs = (double*) fftw_malloc(sizeof(double) * STR_FUNC_MAX_POW * (N_max_incr));

    //----------------------- Write the longitudinal structure functions
    // Normal Structure functions
   	for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vort_str_funcs[p * (N_max_incr) + r] = stats_data->w_str_func[0][p][r] / sys_vars->num_snaps;
   		}
   	}
   	dset_dims_2d[0] = STR_FUNC_MAX_POW;
   	dset_dims_2d[1] = N_max_incr;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VorticityLongitudinalStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Longitudinal Structure Functions");
        exit(1);
    }			
    // Absolute Structure functions
   	for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vort_str_funcs[p * (N_max_incr) + r] = stats_data->w_str_func_abs[0][p][r] / sys_vars->num_snaps;
   		}
   	}
	status = H5LTmake_dataset(file_info->output_file_handle, "AbsoluteVorticityLongitudinalStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Absolute Vorticity Longitudinal Structure Functions");
        exit(1);
    }	

    //----------------------- Write the transverse structure functions
    // Normal structure functions
    for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vort_str_funcs[p * (N_max_incr) + r] = stats_data->w_str_func[1][p][r] / sys_vars->num_snaps;
   		}
   	}
   	status = H5LTmake_dataset(file_info->output_file_handle, "VorticityTransverseStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Vorticity Transverse Structure Funcitons");
        exit(1);
    }		
    // Absolute structure functions
    for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r = 0; r < N_max_incr; ++r) {
	   		vort_str_funcs[p * (N_max_incr) + r] = stats_data->w_str_func_abs[1][p][r] / sys_vars->num_snaps;
   		}
   	}
   	status = H5LTmake_dataset(file_info->output_file_handle, "AbsoluteVorticityTransverseStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Absolute Vorticity Transverse Structure Funcitons");
        exit(1);
    }		
	
    // Free temporary memory
    fftw_free(vort_str_funcs);
    #endif

    ///----------------------------------- Write the Velocity Structure Functions
    #if defined(__VORT_RADIAL_STR_FUNC_STATS)
    // Get the max shell index for the radial averaging
    int shell_indx, indx, tmp;
    int max_shell_indx = (int) round(sqrt((N_max_incr) * (N_max_incr) + (N_max_incr) * (N_max_incr)));

    // Allocate temporary memory to record the histogram data contiguously
    double* vort_radial_str_funcs = (double*) fftw_malloc(sizeof(double) * STR_FUNC_MAX_POW * (max_shell_indx));
    for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
    	for (int i = 0; i < max_shell_indx; ++i) {
    		vort_radial_str_funcs[p * (max_shell_indx) + i] = 0.0;
    	}
    }

    //----------------------- Write the longitudinal structure functions
    // Normal Structure functions
   	for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r_x = 0; r_x < N_max_incr; ++r_x) {
   			tmp = r_x * N_max_incr;
   			for (int r_y = 0; r_y < N_max_incr; ++r_y) {
   				indx = tmp + r_y;

   				shell_indx = (int) round(sqrt(r_x * r_x + r_y * r_y));

		   		vort_radial_str_funcs[p * (max_shell_indx) + shell_indx] += stats_data->w_radial_str_func[p][indx] / sys_vars->num_snaps / (2.0 * M_PI * shell_indx);
		   	}
   		}
   	}

   	// Save the radially averaged raidal structure function
   	dset_dims_2d[0] = STR_FUNC_MAX_POW;
   	dset_dims_2d[1] = max_shell_indx;
   	status = H5LTmake_dataset(file_info->output_file_handle, "RadialVorticityStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_radial_str_funcs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Radial Vorticity  Structure Functions");
        exit(1);
    }
    // Save the raidal structure function field
   	dset_dims_3d[0] = STR_FUNC_MAX_POW;
   	dset_dims_3d[1] = N_max_incr;
   	dset_dims_3d[2] = N_max_incr;
   	status = H5LTmake_dataset(file_info->output_file_handle, "RadialVorticityStructureFunctionsField", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, stats_data->w_radial_str_func[0]);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Radial Vorticity  Structure Functions Field");
        exit(1);
    }

    // Free memory
    fftw_free(vort_radial_str_funcs);

    // Allocate temporary memory to record the histogram data contiguously
    double* vort_radial_str_funcs_abs = (double*) fftw_malloc(sizeof(double) * STR_FUNC_MAX_POW * (max_shell_indx));
    for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
    	for (int i = 0; i < max_shell_indx; ++i) {
    		vort_radial_str_funcs_abs[p * (max_shell_indx) + i] = 0.0;
    	}
    }
		
    // Absolute Structure functions
   	for (int p = 0; p < STR_FUNC_MAX_POW; ++p) {
   		for (int r_x = 0; r_x < N_max_incr; ++r_x) {
   			tmp = r_x * N_max_incr;
   			for (int r_y = 0; r_y < N_max_incr; ++r_y) {
   				indx = tmp + r_y;

   				shell_indx = (int) round(sqrt(r_x * r_x + r_y * r_y));

		   		vort_radial_str_funcs_abs[p * (max_shell_indx) + shell_indx] += stats_data->w_radial_str_func_abs[p][indx] / sys_vars->num_snaps / (2.0 * M_PI * shell_indx);
		   	}
   		}
   	}
	status = H5LTmake_dataset(file_info->output_file_handle, "AbsoluteRadialVorticityStructureFunctions", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vort_radial_str_funcs_abs);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Absolute Radial Vorticity  Structure Functions");
        exit(1);
    }	

    // Save the raidal structure function field
   	dset_dims_3d[0] = STR_FUNC_MAX_POW;
   	dset_dims_3d[1] = N_max_incr;
   	dset_dims_3d[2] = N_max_incr;
   	status = H5LTmake_dataset(file_info->output_file_handle, "AbsoluteRadialVorticityStructureFunctionsField", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, stats_data->w_radial_str_func_abs[0]);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Absolute Radial Vorticity  Structure Functions Field");
        exit(1);
    }	
	
    // Free temporary memory
    fftw_free(vort_radial_str_funcs_abs);
    #endif

	///----------------------------------- Write Mixed Structure Functions
    #if defined(__MIXED_VEL_STR_FUNC_STATS) 
    // Define the dimensions of the mixed velocity structure funciton array
    dset_dims_1d[0] = N_max_incr;
    if ( (H5LTmake_dataset(file_info->output_file_handle, "MixedVelocityStructureFunctions", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->mxd_u_str_func)) < 0) {
    	printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "MixedVelocityStructureFunctions");
    }
    #endif
    #if defined(__MIXED_VORT_STR_FUNC_STATS) 
    // Define the dimensions of the mixed vorticity structure funciton array
    dset_dims_1d[0] = N_max_incr;
    if ( (H5LTmake_dataset(file_info->output_file_handle, "MixedVorticityStructureFunctions", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->mxd_w_str_func)) < 0) {
    	printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "MixedVorticityStructureFunctions");
    }
    #endif
}
/**
 * Frees memory and GSL objects allocated to perform the stats computations.
 */
void FreeStatsObjects(void) {

	// --------------------------------
	//  Free memory
	// --------------------------------
	for (int p = 1; p <= STR_FUNC_MAX_POW; ++p) {
		#if defined(__VEL_STR_FUNC_STATS)
		fftw_free(stats_data->u_str_func[0][p - 1]);
		fftw_free(stats_data->u_str_func[1][p - 1]);
		fftw_free(stats_data->u_str_func_abs[0][p - 1]);
		fftw_free(stats_data->u_str_func_abs[1][p - 1]);
		#endif
		#if defined(__VORT_STR_FUNC_STATS)
		fftw_free(stats_data->w_str_func[0][p - 1]);
		fftw_free(stats_data->w_str_func[1][p - 1]);
		fftw_free(stats_data->w_str_func_abs[0][p - 1]);
		fftw_free(stats_data->w_str_func_abs[1][p - 1]);
		#endif
		#if defined(__VORT_STR_FUNC_STATS)
		fftw_free(stats_data->w_radial_str_func[p - 1]);
		fftw_free(stats_data->w_radial_str_func_abs[p - 1]);
		#endif
	}
	#if defined(__MIXED_VEL_STR_FUNC)
	fftw_free(stats_data->mxd_u_str_func);
	#endif
	#if defined(__MIXED_VORT_STR_FUNC)
	fftw_free(stats_data->mxd_w_str_func);
	#endif
	#if defined(__VEL_GRAD_STATS)
	fftw_free(proc_data->grad_u_hat);
	fftw_free(proc_data->grad_u);
	#endif
	#if defined(__VORT_GRAD_STATS)
	fftw_free(proc_data->grad_w_hat);
	fftw_free(proc_data->grad_w);
	#endif
	fftw_free(stats_data->increments);


	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	#if defined(__REAL_STATS)
	gsl_histogram_free(stats_data->w_hist);
	gsl_histogram_free(stats_data->u_hist);
	gsl_rstat_free(stats_data->w_stats);
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		gsl_rstat_free(stats_data->u_stats[i]);
	}
	#endif
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			#if defined(__VEL_INC_STATS)
			gsl_histogram_free(stats_data->u_incr_hist[i][j]);
			gsl_rstat_free(stats_data->u_incr_stats[i][j]);
			#endif
			#if defined(__VEL_INC_STATS)
			gsl_histogram_free(stats_data->w_incr_hist[i][j]);
			gsl_rstat_free(stats_data->w_incr_stats[i][j]);
			#endif
		}	
	}
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		#if defined(__VEL_GRAD_STATS)
		gsl_histogram_free(stats_data->u_grad_hist[i]);
		gsl_rstat_free(stats_data->u_grad_stats[i]);
		#endif
		#if defined(__VORT_GRAD_STATS)
		gsl_histogram_free(stats_data->w_grad_hist[i]);
		gsl_rstat_free(stats_data->w_grad_stats[i]);
		#endif
	}
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
