/**
* @file utils.c  
* @author Enda Carroll
* @date Sept 2022
* @brief File containing the stats functions for the solver
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
 * Function to compute the real space stats of the vorticity and velocity fields
 * @param iters The index of the current iteration
 */
void ComputeStats(int iters) {

	// Initialize variables
	int tmp, indx;	
	int gsl_status;
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	#if defined(__VEL_INC) || defined(__VORT_INC) || defined(__VEL_STR_FUNC)
	int r;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double long_increment, trans_increment;
	double long_increment_abs, trans_increment_abs;
	int N_max_incr = (int) (GSL_MIN(Nx, Ny) / 2);
	int increment[NUM_INCR] = {1, N_max_incr};
	double norm_fac = 1.0 / (Nx * Ny);
	double std
	#endif

	// ------------------------------------
    // Check System for Stationary
    // ------------------------------------
    ///---------------------------------- System is not stationary yet -> Compute histogram limits
	if (iters < sys_vars->trans_iters) {

		// Loop over the field and compute the increments for adding to the running stats cal
		#if defined(__VEL_INC) || defined(__VORT_INC)
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Compute velocity increments and update histograms
				for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
					
					#if defined(__VEL_INC)
					// Get the current increment
					r = increment[r_indx];

					//------------- Get the longitudinal and transverse Velocity increments
					long_increment  = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0];
					trans_increment = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1];

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(long_increment, stats_data->vel_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(trans_increment, stats_data->vel_inc_stats[1][r_indx]);
					gsl_status = gsl_rstat_add(long_increment + trans_increment, stats_data->vel_inc_stats[2][r_indx]);
					#endif
					
					#if defined(__VORT_INC)
					// Get the current increment
					r = increment[r_indx];

					//------------- Get the longitudinal and transverse Vorticity increments
					long_increment  = run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j];
					trans_increment = run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j];

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(long_increment, stats_data->vort_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(trans_increment, stats_data->vort_inc_stats[1][r_indx]);
					gsl_status = gsl_rstat_add(long_increment + trans_increment, stats_data->vort_inc_stats[2][r_indx]);
					#endif
				}
			}
		}
		#endif
	}
	///---------------------------------- System is stationary -> calculate stats
	else {

		// ---------------------------------------------
	    // Update Stats Counter & Reset Stats Objects
	    // --------------------------------------------
	    // Update counter
		stats_data->num_stats_steps++;

		// Set histogram bin limits then reset running stats counters
		if (stats_data->set_stats_flag) {

			for (int type = 0; type < INCR_TYPES + 1; ++type) {
				for (int r = 0; r < NUM_INCR; ++r) {
					#if defined(__VEL_INC)
					// Get the standard deviation / rms
					std = gsl_rstat_sd(stats_data->vel_inc_stats[type][r]);

					// Set bin ranges for the histograms -> Set (in units) of standard deviations
					gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vel_inc_hist[type][r], -VEL_BIN_LIM * std, VEL_BIN_LIM * std);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for: ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Velocity Increment Histogram");
						exit(1);
					}
					#endif

					#if defined(__VORT_INC)
					// Get the standard deviation / rms
					std = gsl_rstat_sd(stats_data->vort_inc_stats[type][r]);

					// Set bin ranges for the histograms -> Set (in units) of standard deviations
					gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vort_inc_hist[type][r], -VEL_BIN_LIM * std, VEL_BIN_LIM * std);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for: ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Velocity Increment Histogram");
						exit(1);
					}
					#endif

					// Reset the stats counters
					gsl_status = gsl_rstat_reset(stats_data->vel_inc_stats[type][r]);
					gsl_status = gsl_rstat_reset(stats_data->vort_inc_stats[type][r]);
				}
			}

			// Reset set stats flag
			stats_data->set_stats_flag = 0;
		}

		// ---------------------------------------------
	    // Compute Stats
	    // --------------------------------------------
	    // Loop over the field and compute the increments, str funcs etc
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				///--------------------------------- Compute velocity and vorticity increments and update histograms
				for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
					
					#if defined(__VEL_INC)
					// Get the current increment
					r = increment[r_indx];

					// Get the longitudinal and transverse Velocity increments
					long_increment  = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0];
					trans_increment = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1];

					// Update the histograms
					gsl_status = gsl_histogram_increment(stats_data->vel_inc_hist[0][r_indx], long_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment", s, gsl_status, long_increment);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(stats_data->vel_inc_hist[1][r_indx], trans_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Velocity Increment", s, gsl_status, trans_increment);
						exit(1);
					}

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(long_increment, stats_data->vel_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(trans_increment, stats_data->vel_inc_stats[1][r_indx]);
					gsl_status = gsl_rstat_add(long_increment + trans_increment, stats_data->vel_inc_stats[2][r_indx]);
					#endif
					
					#if defined(__VORT_INC)
					// Get the current increment
					r = increment[r_indx];

					// Get the longitudinal and transverse Vorticity increments
					long_increment  = run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j];
					trans_increment = run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j];

					// Update the histograms
					gsl_status = gsl_histogram_increment(stats_data->vort_inc_hist[0][r_indx], long_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Vorticity Increment", s, gsl_status, long_increment);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(stats_data->vort_inc_hist[1][r_indx], trans_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Vorticity Increment", s, gsl_status, trans_increment);
						exit(1);
					}		

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(long_increment, stats_data->vort_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(trans_increment, stats_data->vort_inc_stats[1][r_indx]);
					gsl_status = gsl_rstat_add(long_increment + trans_increment, stats_data->vort_inc_stats[2][r_indx]);
					#endif
				}
			}
		}

		///--------------------------------- Compute velocity structure functions
		#if defined(__VEL_STR_FUNC)
		for (int p = 1; p < NUM_POW; ++p) {
			for (int r_inc = 1; r_inc <= N_max_incr; ++r_inc) {
				// Initialize increments
				long_increment      = 0.0;
				trans_increment     = 0.0;
				long_increment_abs  = 0.0;
				trans_increment_abs = 0.0;
				
				// Loop over field
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
				stats_data->str_func[0][p - 1][r_inc - 1]     += long_increment * norm_fac;	
				stats_data->str_func[1][p - 1][r_inc - 1]     += trans_increment * norm_fac;
				stats_data->str_func_abs[0][p - 1][r_inc - 1] += long_increment_abs * norm_fac;	
				stats_data->str_func_abs[1][p - 1][r_inc - 1] += trans_increment_abs * norm_fac;
			}
		}
		#endif	
	}
}
/**
 * Function to allocate and initialize the stats objects
 */
void InitializeStats(void) {

	// Initialize variables
	int tmp, indx;	
	int gsl_status;
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;


	// --------------------------------	
	//  Initialize Increment Stats
	// --------------------------------
	#if defined(__VEL_INC)
	// Initialize GSL objects for the increments
	for (int i = 0; i < INCR_TYPES + 1; ++i) {
		for (int r = 0; r < NUM_INCR; ++r) {
			// Initialize histogram objects for each increments for each direction	
			stats_data->vel_inc_hist[i][r]  = gsl_histogram_alloc(N_BINS);
			stats_data->vort_inc_hist[i][r] = gsl_histogram_alloc(N_BINS);
			
			// Initialize running stats for the increments
			stats_data->vel_inc_stats[i][r]  = gsl_rstat_alloc();
			stats_data->vort_inc_stats[i][r] = gsl_rstat_alloc();
		}
	}
	#endif

	// --------------------------------	
	//  Initialize Str Func Stats
	// --------------------------------
	#if defined(__STR_FUNC_STATS)
	// Allocate memory for each structure function for each of the increment directions
	int N_max_incr = (int) GSL_MIN(Nx, Ny) / 2;
	for (int i = 0; i < INCR_TYPES + 1; ++i) {
		for (int p = 1; p < NUM_POW; ++p) {
			stats_data->vel_str_func[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->vel_str_func[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Structure Functions");
				exit(1);
			}
			stats_data->vel_str_func_abs[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->velstr_func_abs[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Absolute Velocity Structure Functions");
				exit(1);
			}

			// Initialize arrays
			for (int r = 0; r < N_max_incr; ++r) {
				stats_data->vel_str_func[i][p - 1][r]     = 0.0;
				stats_data->vel_str_func_abs[i][p - 1][r] = 0.0;
			}
		}
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
	#if defined(__STR_FUNC_STATS)
	for (int p = 1; p < NUM_POW; ++p) {
		fftw_free(stats_data->vel_str_func[0][p - 1]);
		fftw_free(stats_data->vel_str_func[1][p - 1]);
		fftw_free(stats_data->vel_str_func_abs[0][p - 1]);
		fftw_free(stats_data->vel_str_func_abs[1][p - 1]);
	}
	#endif

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	#if defined(__VEL_INC_STATS)
	for (int i = 0; i < INCR_TYPES + 1; ++i) {
		for (int r = 0; r < NUM_INCR; ++r) {
			gsl_histogram_free(stats_data->vel_inc_hist[i][r]);
			gsl_histogram_free(stats_data->vort_inc_hist[i][r]);
			gsl_rstat_free(stats_data->vel_inc_stats[i][r]);
			gsl_rstat_free(stats_data->vort_inc_stats[i][r]);
		}	
	}
	#endif
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------