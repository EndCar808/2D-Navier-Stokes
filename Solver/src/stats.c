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
	#if defined(__VEL_INC) || defined(__VORT_INC) || defined(__VEL_STR_FUNC) || defined(__VORT_STR_FUNC) || defined(__MIXED_VEL_STR_FUNC) || defined(__MIXED_VORT_STR_FUNC)
	int r;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double vel_long_increment, vel_trans_increment, mixed_vel_increment;
	double vel_long_increment_abs, vel_trans_increment_abs;
	double vort_long_increment, vort_trans_increment, mixed_vort_increment;
	double vort_long_increment_abs, vort_trans_increment_abs;
	int N_max_incr = (int) (GSL_MIN(Nx, Ny) / 2);
	int increment[NUM_INCR] = {1, N_max_incr};
	double norm_fac = 1.0 / pow((Nx * Ny), 2.0);
	double rms;
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
					vel_long_increment  = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0];
					vel_trans_increment = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1];

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(vel_long_increment, stats_data->vel_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(vel_trans_increment, stats_data->vel_inc_stats[1][r_indx]);
					stats_data->vel_inc_sqr[0][r] += vel_long_increment * vel_long_increment;
					stats_data->vel_inc_sqr[1][r] += vel_trans_increment * vel_trans_increment;
					stats_data->vel_inc_num_data[0][r]++;
					stats_data->vel_inc_num_data[1][r]++;
					#endif
					
					#if defined(__VORT_INC)
					// Get the current increment
					r = increment[r_indx];

					//------------- Get the longitudinal and transverse Vorticity increments
					vort_long_increment  = run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j];
					vort_trans_increment = run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j];

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(vort_long_increment, stats_data->vort_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(vort_trans_increment, stats_data->vort_inc_stats[1][r_indx]);
					stats_data->vort_inc_sqr[0][r] += vort_long_increment * vort_long_increment;
					stats_data->vort_inc_sqr[1][r] += vort_trans_increment * vort_trans_increment;
					stats_data->vort_inc_num_data[0][r]++;
					stats_data->vort_inc_num_data[1][r]++;
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

			for (int type = 0; type < INCR_TYPES; ++type) {
				for (int r = 0; r < NUM_INCR; ++r) {
					#if defined(__VEL_INC)
					MPI_Allreduce(MPI_IN_PLACE, &(stats_data->vel_inc_sqr[type][r]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce(MPI_IN_PLACE, &(stats_data->vel_inc_num_data[type][r]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
					rms = sqrt(stats_data->vel_inc_sqr[type][r] / stats_data->vel_inc_num_data[type][r]);

					// Set bin ranges for the histograms -> Set (in units) of standard deviations
					gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vel_inc_hist[type][r], -VEL_BIN_LIM * rms, VEL_BIN_LIM * rms);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for: ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Velocity Increment Histogram");
						exit(1);
					}
					#endif

					#if defined(__VORT_INC)
					MPI_Allreduce(MPI_IN_PLACE, &(stats_data->vort_inc_sqr[type][r]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce(MPI_IN_PLACE, &(stats_data->vort_inc_num_data[type][r]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
					rms = sqrt(stats_data->vort_inc_sqr[type][r] / stats_data->vort_inc_num_data[type][r]);

					// Set bin ranges for the histograms -> Set (in units) of standard deviations
					gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vort_inc_hist[type][r], -VORT_BIN_LIM * rms, VORT_BIN_LIM * rms);
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
					vel_long_increment  = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0];
					vel_trans_increment = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1];

					// Update the histograms
					gsl_status = gsl_histogram_increment(stats_data->vel_inc_hist[0][r_indx], vel_long_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment", iters, gsl_status, vel_long_increment);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(stats_data->vel_inc_hist[1][r_indx], vel_trans_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Velocity Increment", iters, gsl_status, vel_trans_increment);
						exit(1);
					}

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(vel_long_increment, stats_data->vel_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(vel_trans_increment, stats_data->vel_inc_stats[1][r_indx]);
					#endif
					
					#if defined(__VORT_INC)
					// Get the current increment
					r = increment[r_indx];

					// Get the longitudinal and transverse Vorticity increments
					vort_long_increment  = run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j];
					vort_trans_increment = run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j];

					// Update the histograms
					gsl_status = gsl_histogram_increment(stats_data->vort_inc_hist[0][r_indx], vort_long_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Vorticity Increment", iters, gsl_status, vort_long_increment);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(stats_data->vort_inc_hist[1][r_indx], vort_trans_increment);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Transverse Vorticity Increment", iters, gsl_status, vort_trans_increment);
						exit(1);
					}		

					// Update to vel inc running stats
					gsl_status = gsl_rstat_add(vort_long_increment, stats_data->vort_inc_stats[0][r_indx]);
					gsl_status = gsl_rstat_add(vort_trans_increment, stats_data->vort_inc_stats[1][r_indx]);
					#endif
				}
			}
		}

		///--------------------------------- Compute velocity structure functions
		#if defined(__VEL_STR_FUNC) || defined(__VORT_STR_FUNC)
		for (int p = 1; p < NUM_POW; ++p) {
			for (int r_inc = 1; r_inc <= N_max_incr; ++r_inc) {
				// Initialize increments
				#if defined(__VEL_STR_FUNC)
				vel_long_increment      = 0.0;
				vel_trans_increment     = 0.0;
				vel_long_increment_abs  = 0.0;
				vel_trans_increment_abs = 0.0;
				#endif
				#if defined(__VORT_STR_FUNC)
				vort_long_increment      = 0.0;
				vort_trans_increment     = 0.0;
				vort_long_increment_abs  = 0.0;
				vort_trans_increment_abs = 0.0;
				#endif
				if (p == 3) {
					#if defined(__MIXED_VORT_STR_FUNC)
					mixed_vel_increment = 0.0;
					#endif
					#if defined(__MIXED_VORT_STR_FUNC)
					mixed_vort_increment = 0.0;
					#endif	
				}
							
				// Loop over field
				for (int i = 0; i < Nx; ++i) {
					tmp = i * Ny;
					for (int j = 0; j < Ny; ++j) {
						indx = tmp + j;
					
						// Get increments
						#if defined(__VEL_STR_FUNC) || defined(__MIXED_VEL_STR_FUNC) || defined(__MIXED_VORT_STR_FUNC)
						vel_long_increment      += pow(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0], p);
						vel_trans_increment     += pow(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1], p);
						vel_long_increment_abs  += pow(fabs(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0]), p);
						vel_trans_increment_abs += pow(fabs(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1]), p);
						#endif
						#if defined(__VORT_STR_FUNC) || defined(__MIXED_VORT_STR_FUNC)
						vort_long_increment      += pow(run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j], p);
						vort_trans_increment     += pow(run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j], p);
						vort_long_increment_abs  += pow(fabs(run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j]), p);
						vort_trans_increment_abs += pow(fabs(run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j]), p);
						#endif
						if (p == 3) {
							#if defined(__MIXED_VORT_STR_FUNC)
							mixed_vel_increment += (run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0]) * pow(run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1], 2.0);
							#endif
							#if defined(__MIXED_VORT_STR_FUNC)
							mixed_vort_increment += (run_data->u[SYS_DIM * (((i + r_inc) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0]) * pow(run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j], 2.0);
							#endif	
						}
					}
				}
				#if defined(__VEL_STR_FUNC)
				// Compute velocity str functions - normalize here
				stats_data->vel_str_func[0][p - 1][r_inc - 1]     += vel_long_increment * norm_fac;	
				stats_data->vel_str_func[1][p - 1][r_inc - 1]     += vel_trans_increment * norm_fac;
				stats_data->vel_str_func_abs[0][p - 1][r_inc - 1] += vel_long_increment_abs * norm_fac;	
				stats_data->vel_str_func_abs[1][p - 1][r_inc - 1] += vel_trans_increment_abs * norm_fac;
				#endif
				#if defined(__VORT_STR_FUNC)
				// Compute vorticity str functions - normalize here
				stats_data->vort_str_func[0][p - 1][r_inc - 1]     += vort_long_increment * norm_fac;	
				stats_data->vort_str_func[1][p - 1][r_inc - 1]     += vort_trans_increment * norm_fac;
				stats_data->vort_str_func_abs[0][p - 1][r_inc - 1] += vort_long_increment_abs * norm_fac;	
				stats_data->vort_str_func_abs[1][p - 1][r_inc - 1] += vort_trans_increment_abs * norm_fac;
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
	#if defined(__VEL_STR_FUNC) || defined(__VORT_STR_FUNC) || defined(__MIXED_VEL_STR_FUNC) || defined(__MIXED_VORT_STR_FUNC)
	int N_max_incr = (int) GSL_MIN(Nx, Ny) / 2;
	#endif

	// Initialize stats counter and stats flag
	stats_data->num_stats_steps = 0;
	stats_data->set_stats_flag  = 1;


	// --------------------------------	
	//  Initialize Increment Stats
	// --------------------------------
	// Initialize GSL objects for the increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int r = 0; r < NUM_INCR; ++r) {
			#if defined(__VEL_INC) 
			// Initialize histogram and stats objects for the velocity increments
			stats_data->vel_inc_hist[i][r]     = gsl_histogram_alloc(VEL_NUM_BINS);
			stats_data->vel_inc_stats[i][r]    = gsl_rstat_alloc();
			stats_data->vel_inc_sqr[i][r]      = 0.0;
			stats_data->vel_inc_num_data[i][r] = 0.0;
			#endif

			#if defined(__VORT_INC) 			
			// Initialize histogram and stats objects for the vorticity increments
			stats_data->vort_inc_hist[i][r]    = gsl_histogram_alloc(VORT_NUM_BINS);
			stats_data->vort_inc_stats[i][r]   = gsl_rstat_alloc();
			stats_data->vel_inc_sqr[i][r]      = 0.0;
			stats_data->vel_inc_num_data[i][r] = 0.0;
			#endif
		}
	}

	// --------------------------------	
	//  Initialize Str Func Stats
	// --------------------------------
	// Allocate memory for each structure function for each of the increment directions
	#if defined(__VEL_STR_FUNC) || defined(__VORT_STR_FUNC)
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int p = 1; p < NUM_POW; ++p) {
			#if defined(__VEL_STR_FUNC)
			stats_data->vel_str_func[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->vel_str_func[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Structure Functions");
				exit(1);
			}
			stats_data->vel_str_func_abs[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->vel_str_func_abs[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Absolute Velocity Structure Functions");
				exit(1);
			}

			// Initialize arrays
			for (int r = 0; r < N_max_incr; ++r) {
				stats_data->vel_str_func[i][p - 1][r]     = 0.0;
				stats_data->vel_str_func_abs[i][p - 1][r] = 0.0;
			}
			#endif

			#if defined(__VORT_STR_FUNC)
			stats_data->vort_str_func[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->vort_str_func[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Structure Functions");
				exit(1);
			}
			stats_data->vort_str_func_abs[i][p - 1] = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
			if (stats_data->vort_str_func_abs[i][p - 1] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Absolute Vorticity Structure Functions");
				exit(1);
			}

			// Initialize arrays
			for (int r = 0; r < N_max_incr; ++r) {
				stats_data->vort_str_func[i][p - 1][r]     = 0.0;
				stats_data->vort_str_func_abs[i][p - 1][r] = 0.0;
			}
			#endif
		}
	}
	#endif
	#if defined(__MIXED_VEL_STR_FUNC)
	stats_data->vel_mixed_str_func = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
	if (stats_data->vel_mixed_str_func == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Mixed Velocity Structure Functions");
		exit(1);
	}
	for (int r = 0; r < N_max_incr; ++r) {
		stats_data->vort_mixed_str_func[r] = 0.0;
	}
	#endif
	#if defined(__MIXED_VORT_STR_FUNC)
	stats_data->vort_mixed_str_func = (double* )fftw_malloc(sizeof(double) * (N_max_incr));
	if (stats_data->vort_mixed_str_func == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Mixed Vorticity Structure Functions");
		exit(1);
	}
	for (int r = 0; r < N_max_incr; ++r) {
		stats_data->vort_mixed_str_func[r] = 0.0;
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
	for (int p = 1; p < NUM_POW; ++p) {
		#if defined(__VEL_STR_FUNC)
		fftw_free(stats_data->vel_str_func[0][p - 1]);
		fftw_free(stats_data->vel_str_func[1][p - 1]);
		fftw_free(stats_data->vel_str_func_abs[0][p - 1]);
		fftw_free(stats_data->vel_str_func_abs[1][p - 1]);
		#endif
		#if defined(__VORT_STR_FUNC)
		fftw_free(stats_data->vort_str_func[0][p - 1]);
		fftw_free(stats_data->vort_str_func[1][p - 1]);
		fftw_free(stats_data->vort_str_func_abs[0][p - 1]);
		fftw_free(stats_data->vort_str_func_abs[1][p - 1]);
		#endif
	}
	#if defined(__MIXED_VEL_STR_FUNC)
	fftw_free(stats_data->vel_mixed_str_func);
	#endif
	#if defined(__MIXED_VORT_STR_FUNC)
	fftw_free(stats_data->vort_mixed_str_func);
	#endif
	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int r = 0; r < NUM_INCR; ++r) {
			#if defined(__VEL_INC_STATS)
			gsl_histogram_free(stats_data->vel_inc_hist[i][r]);
			gsl_rstat_free(stats_data->vel_inc_stats[i][r]);
			#endif
			#if defined(__VORT_INC_STATS)
			gsl_histogram_free(stats_data->vort_inc_hist[i][r]);
			gsl_rstat_free(stats_data->vort_inc_stats[i][r]);
			#endif
		}	
	}
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------