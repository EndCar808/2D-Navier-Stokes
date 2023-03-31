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
	double w_max = 0.0;
	double w_min = 1e8;
	gsl_stats_minmax(&w_min, &w_max, run_data->w, 1, Nx * Ny);

	double u_max = 0.0; 
	double u_min = 1e8;
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

	// Perform inverse transform and normalize to get the gradients in real space - no need to presave grad_w_hat & grad_u_hat, wont be used again
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_w_hat, proc_data->grad_w);
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_u_hat, proc_data->grad_u);
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Normalize the gradients
			proc_data->grad_w[SYS_DIM * indx + 0] *= 1.0; //norm_fac;
			proc_data->grad_w[SYS_DIM * indx + 1] *= 1.0; //norm_fac;
			proc_data->grad_u[SYS_DIM * indx + 0] *= 1.0; //norm_fac;
			proc_data->grad_u[SYS_DIM * indx + 1] *= 1.0; //norm_fac;
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

			// Update stats accumulators
			gsl_rstat_add(run_data->w[indx], stats_data->r_stat_w);			
			for (int i = 0; i < SYS_DIM + 1; ++i) {
				if (i < SYS_DIM) {
					// Add to the individual directions accumulators
					gsl_rstat_add(run_data->u[SYS_DIM * indx + i], stats_data->r_stat_u[i]);
				}
				else {
					// Add to the combined accumulator
					for (int j = 0; j < SYS_DIM; ++j) {
						gsl_rstat_add(run_data->u[SYS_DIM * indx + j], stats_data->r_stat_u[SYS_DIM]);	
					}
				}
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
 * Allocates memory and initializes GSL stats objects for the stats computations
 * @param N Array containing the size of each dimension
 */
void AllocateStatsMemory(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;


	// --------------------------------	
	//  Initialize Real Space Stats
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

	// Initialize the running stats for the velocity and vorticity fields
	stats_data->r_stat_w = gsl_rstat_alloc();
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		stats_data->r_stat_u[i] = gsl_rstat_alloc();
	}
	#endif
	
	// --------------------------------
	//  Initialize Graient Stats
	// --------------------------------
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
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Gradient");
		exit(1);
	}
	proc_data->grad_w = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (proc_data->grad_w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Gradient");
		exit(1);
	}

	for (int i = 0; i < SYS_DIM + 1; ++i) {
		// Initialize the gradient histograms
		stats_data->vel_grad[i]  = gsl_histogram_alloc(N_BINS);
		stats_data->vort_grad[i] = gsl_histogram_alloc(N_BINS);

		// Initialize the running stats for the gradients
		stats_data->r_stat_grad_u[i] = gsl_rstat_alloc();
		stats_data->r_stat_grad_w[i] = gsl_rstat_alloc();
	}
	#endif


	// --------------------------------	
	//  Initialize Increment Stats
	// --------------------------------
	#if defined(__VEL_INC_STATS)
	// Initialize GSL objects for the increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			// Initialize a histogram objects for each increments for each direction	
			stats_data->w_incr[i][j]   = gsl_histogram_alloc(N_BINS);
			stats_data->vel_incr[i][j] = gsl_histogram_alloc(N_BINS);
			
			// Initialize running stats for the increments
			stats_data->r_stat_vel_incr[i][j]  = gsl_rstat_alloc();
			stats_data->r_stat_vort_incr[i][j] = gsl_rstat_alloc();
		}
	}
	#endif


	// --------------------------------	
	//  Initialize Str Func Stats
	// --------------------------------
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
}
/**
 * Frees memory and GSL objects allocated to perform the stats computations.
 */
void FreeStatsObjects(void) {

	// --------------------------------
	//  Free memory
	// --------------------------------
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

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	#if defined(__REAL_STATS)
	gsl_histogram_free(stats_data->w_pdf);
	gsl_histogram_free(stats_data->u_pdf);
	gsl_rstat_free(stats_data->r_stat_w);
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		gsl_rstat_free(stats_data->r_stat_u[i]);
	}
	#endif
	#if defined(__VEL_INC_STATS)
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			gsl_histogram_free(stats_data->vel_incr[j][i]);
			gsl_histogram_free(stats_data->w_incr[j][i]);
			gsl_rstat_free(stats_data->r_stat_vel_incr[j][i]);
			gsl_rstat_free(stats_data->r_stat_vort_incr[j][i]);
		}	
	}
	#endif
	#if defined(__GRAD_STATS)
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		gsl_histogram_free(stats_data->vel_grad[i]);
		gsl_histogram_free(stats_data->vort_grad[i]);
		gsl_rstat_free(stats_data->r_stat_grad_u[i]);
		gsl_rstat_free(stats_data->r_stat_grad_w[i]);
	}
	#endif
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
