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
#include <time.h>
#include <sys/time.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "utils.h"
#include "stats.h"
#include "post_proc.h"
#include "data_types.h"
#include "hdf5_funcs.h"
#include "phase_sync.h"
#include "full_field.h"
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/** 
* Performs post processing of the solver data
*/
void PostProcessing(void) {

	// --------------------------------
	//  Open Input File and Get Data
	// --------------------------------
	OpenInputAndInitialize(); 

	// --------------------------------
	//  Open Output File
	// --------------------------------
	OpenOutputFile();
	int chk_pt_indx = 0;

	// --------------------------------
	//  Allocate Processing Memmory
	// --------------------------------
	AllocateMemory(sys_vars->N);

	InitializeFFTWPlans(sys_vars->N);
	// --------------------------------
	//  Perform Precomputations
	// --------------------------------
	#if defined(__GRAD_STATS) || defined(__VEL_INC_STATS)
	Precompute();
	#endif

	//////////////////////////////
	// Begin Snapshot Processing
	//////////////////////////////
	printf("\n\nStarting Snapshot Processing:\n");
	for (int s = 0; s <  sys_vars->num_snaps; ++s) {		
		
		// Start timer
		double loop_begin = omp_get_wtime();

		// --------------------------------
		//  Read in Data
		// --------------------------------
		ReadInData(s);

		// --------------------------------
		//  Real Space Stats
		// --------------------------------
		#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
		RealSpaceStats(s);
		#endif

		// --------------------------------
		//  Spectra Data
		// --------------------------------
		#if defined(__SPECTRA)
		// Compute the enstrophy spectrum
		EnstrophySpectrum();
        EnergySpectrum();
		#endif
		#if defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC)
        FluxSpectra(s);
		#endif

		// --------------------------------
		//  Full Field Data
		// --------------------------------
		#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
		FullFieldData();
		#endif

		// --------------------------------
		//  Phase Sync
		// --------------------------------
		#if defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
		PhaseSync(s); 
		PhaseSyncSector(s);
		#endif

		// --------------------------------
		//  Write Data to File
		// --------------------------------
		WriteDataToFile(run_data->time[s], s);
		if ((s + 1) % sys_vars->chk_pt_every == 0) { // shifted by 1 to avoid writing first iteration
			// Write post computation to checkpoint file
			CreateCheckPointCopy(s, chk_pt_indx);
			chk_pt_indx++;
		}
		
		// End timer for current loop
		double loop_end = omp_get_wtime();

		// Print update to screen
		printf("Snapshot: %d/%ld\tSaving Index: %d \t Time: %g(s)\n", s + 1, sys_vars->num_snaps, chk_pt_indx, (loop_end - loop_begin));
	}
	///////////////////////////////
	// End Snapshot Processing
	///////////////////////////////

	// ---------------------------------
	//  Final Write of Data and Close 
	// ---------------------------------
	// Write any remaining datasets to output file
	FinalWriteAndClose(file_info->output_file_name);
	
	// --------------------------------
	//  Clean Up
	// --------------------------------
	// Free allocated memory
	FreeMemoryAndCleanUp();
}
/**
 * Performs a run over the data to precompute and quantities needed before performing
 * the proper run over the data
 */
void Precompute(void) {

	// Initialize variables
	int gsl_status;
	int tmp, indx;
	const long int Ny         = sys_vars->N[0];
	const long int Nx         = sys_vars->N[1];
	const long int Nx_Fourier = Nx / 2 + 1;
	int r;
	int N_max_incr = (int) (GSL_MIN(Ny, Nx) / 2);
	double vel_long_increment_x, vel_trans_increment_x;
	double vel_long_increment_y, vel_trans_increment_y;
	int x_incr, y_incr;
	double vort_long_increment, vort_trans_increment;
	double norm_fac = 1.0 / (Ny * Nx);
	double std_u, std_w;
	double std_u_incr, std_w_incr;
	double delta_x = 2.0 * M_PI / Nx;
	double delta_y = 2.0 * M_PI / Ny;

	// Print to screen that a pre computation step is need and begin timeing it
	printf("\n["YELLOW"NOTE"RESET"] --- Performing a precomputation step...\n\n");
	struct timeval begin, end;
	gettimeofday(&begin, NULL);

	// --------------------------------
	// Loop Through Snapshots
	// --------------------------------
	for (int s = 0; s < sys_vars->num_snaps; ++s) {

		printf("Precomputation Step: %d/%ld\n", s + 1, sys_vars->num_snaps);
		
		// Read in snaps
		ReadInData(s);

		// --------------------------------
		// Precompute Stats
		// --------------------------------
		// Compute the graident in Fourier space
		#if defined(__VEL_GRAD_STATS) || defined(__VORT_GRAD_STATS)
		for (int i = 0; i < Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Compute the gradients of the vorticity
				#if defined(__VORT_GRAD_STATS)
				proc_data->grad_w_hat[SYS_DIM * indx + 0] = I * run_data->k[1][j] * run_data->w_hat[indx];
				proc_data->grad_w_hat[SYS_DIM * indx + 1] = I * run_data->k[0][i] * run_data->w_hat[indx];
				#endif

				// Compute the gradients of the vorticity
				#if defined(__VEL_GRAD_STATS)
				proc_data->grad_u_hat[SYS_DIM * indx + 0] = I * run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 0];
				proc_data->grad_u_hat[SYS_DIM * indx + 1] = I * run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 1];
				#endif
			}
		}
		// Perform inverse transform to get the gradients in real space - no need to presave grad_w_hat & grad_u_hat, wont be used again
		#if defined(__VORT_GRAD_STATS)
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_w_hat, proc_data->grad_w);
		#endif
		#if defined(__VEL_GRAD_STATS)
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_u_hat, proc_data->grad_u);
		#endif
		#endif

		// Loop over real space
		for (int i = 0; i < Ny; ++i) {
			tmp = i * Nx;
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize the gradients 
				#if defined(__VORT_GRAD_STATS)
				proc_data->grad_w[SYS_DIM * indx + 0] *= 1.0; //norm_fac;
				proc_data->grad_w[SYS_DIM * indx + 1] *= 1.0; //norm_fac;
				#endif
				#if defined(__VEL_GRAD_STATS)
				proc_data->grad_u[SYS_DIM * indx + 0] *= 1.0; //norm_fac;
				proc_data->grad_u[SYS_DIM * indx + 1] *= 1.0; //norm_fac;
				#endif

				// Add gradients to stats accumulators
				for (int i = 0; i < SYS_DIM + 1; ++i) {
					if (i < SYS_DIM) {
						// Add to the individual directions accumulators
						#if defined(__VORT_GRAD_STATS)
						gsl_rstat_add(proc_data->grad_w[SYS_DIM * indx + i] * delta_x, stats_data->w_grad_stats[i]);
						#endif
						#if defined(__VEL_GRAD_STATS)
						gsl_rstat_add(proc_data->grad_u[SYS_DIM * indx + i] * delta_x, stats_data->u_grad_stats[i]);
						#endif
					}
					else {
						// Add to the combined accumulator
						for (int j = 0; j < SYS_DIM; ++j) {
							#if defined(__VORT_GRAD_STATS)
							gsl_rstat_add(proc_data->grad_w[SYS_DIM * indx + j] * delta_x * delta_y, stats_data->w_grad_stats[SYS_DIM]);
							#endif
							#if defined(__VEL_GRAD_STATS)
							gsl_rstat_add(proc_data->grad_u[SYS_DIM * indx + j] * delta_x * delta_y, stats_data->u_grad_stats[SYS_DIM]);
							#endif
						}
					}
				}


				// Compute velocity increments and update histograms
				for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
					// Get the current increment
					r = stats_data->increments[r_indx];

					//------------- Get the longitudinal and transverse Velocity increments
					#if defined(__VEL_INC_STATS)
					// Increments in the x direction
					x_incr = j + r;
					if (x_incr < Nx) {
						// Longitudinal increment in the x direction
						vel_long_increment_x  = run_data->u[SYS_DIM * (i * Nx + x_incr) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0];
						gsl_rstat_add(vel_long_increment_x, stats_data->u_incr_stats[0][r_indx]);
						
						// Transverse increment in the x direction
						vel_trans_increment_x = run_data->u[SYS_DIM * (i * Nx + x_incr) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1];
						gsl_rstat_add(vel_trans_increment_x, stats_data->u_incr_stats[1][r_indx]);
					}

					// Increments in the y direction
					y_incr = i + r;
					if (y_incr < Ny) {
						// Longitudinal increment in the y direction
						vel_long_increment_y  = run_data->u[SYS_DIM * (y_incr * Nx + j) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1];
						gsl_rstat_add(vel_long_increment_y, stats_data->u_incr_stats[0][r_indx]);
						
						// Transverse increment in the y direction
						vel_trans_increment_y = run_data->u[SYS_DIM * (y_incr * Nx + j) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0];
						gsl_rstat_add(vel_trans_increment_y, stats_data->u_incr_stats[1][r_indx]);
					}
					#endif

					//------------- Get the longitudinal and transverse Vorticity increments
					#if defined(__VORT_INC_STATS)
					// Increment in the x direction 
					x_incr = j + r;
					if (x_incr < Nx) {
						vort_long_increment  = run_data->w[i * Nx + x_incr] - run_data->w[i * Nx + j];
						gsl_rstat_add(vort_long_increment, stats_data->w_incr_stats[0][r_indx]);

					}

					// Increment in the y direction
					y_incr = i + r; 
					if (y_incr < Ny) {
						vort_trans_increment = run_data->w[y_incr * Nx + j] - run_data->w[i * Nx + j];
						gsl_rstat_add(vort_trans_increment, stats_data->w_incr_stats[1][r_indx]);
					}
					#endif

					// //------------- Get the longitudinal and transverse Velocity increments
					// #if defined(__VEL_INC_STATS)
					// vel_long_increment  = run_data->u[SYS_DIM * (i * Nx + (j + r) % Nx) + 0] - run_data->u[SYS_DIM * (i * Nx + j) + 0];
					// vel_trans_increment = run_data->u[SYS_DIM * (i * Nx + (j + r) % Nx) + 1] - run_data->u[SYS_DIM * (i * Nx + j) + 1];

					// // Update the stats accumulators
					// gsl_rstat_add(vel_long_increment, stats_data->u_incr_stats[0][r_indx]);
					// gsl_rstat_add(vel_trans_increment, stats_data->u_incr_stats[1][r_indx]);
					// #endif

					// //------------- Get the longitudinal and transverse Vorticity increments
					// #if defined(__VORT_INC_STATS)
					// vort_long_increment  = run_data->w[i * Nx + (j + r) % Nx] - run_data->w[i * Nx + j];
					// vort_trans_increment = run_data->w[((i + r) % Ny) * Nx + j] - run_data->w[i * Nx + j];

					// // Update the stats accumulators
					// gsl_rstat_add(vort_long_increment, stats_data->w_incr_stats[0][r_indx]);
					// gsl_rstat_add(vort_trans_increment, stats_data->w_incr_stats[1][r_indx]);			
					// #endif
				}
			}
		}
	}

	// --------------------------------
	// Initialize Gradient Histograms
	// --------------------------------
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		#if defined(__VEL_GRAD_STATS)
		// Get the std of the gradients
		std_u = gsl_rstat_sd(stats_data->u_grad_stats[i]);

		// Velocity gradients
		gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_grad_hist[i], -BIN_LIM * std_u, BIN_LIM * std_u);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Gradient Increments");
			exit(1);
		}
		#endif
		
		#if defined(__VORT_GRAD_STATS)
		// Get the std of the gradients
		std_w = gsl_rstat_sd(stats_data->w_grad_stats[i]);

		// Vorticity gradients
		gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_grad_hist[i], -BIN_LIM * std_w, BIN_LIM * std_w);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Gradient Increments");
			exit(1);
		}
		#endif
	}

	// --------------------------------
	// Initialize Increment Histograms
	// --------------------------------
	// Set the bin limits for the velocity increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			#if defined(__VEL_INC_STATS)
			// Get the std of the incrments
			std_u_incr = gsl_rstat_sd(stats_data->u_incr_stats[i][j]);

			// Velocity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_incr_hist[i][j], -BIN_LIM * std_u_incr, BIN_LIM * std_u_incr);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Increments");
				exit(1);
			}
			#endif

			#if defined(__VORT_INC_STATS)						
			// Get the std of the incrments
			std_w_incr = gsl_rstat_sd(stats_data->w_incr_stats[i][j]);

			// Vorticity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_incr_hist[i][j], -BIN_LIM *std_w_incr, BIN_LIM * std_w_incr);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Increments");
				exit(1);
			}
			#endif
		}
	}

	printf("\n["YELLOW"NOTE"RESET"] --- Precomputation step complete!\n");

	// Finish timing pre compute step and print to screen
	gettimeofday(&end, NULL);
	// PrintTime(begin.tv_sec, end.tv_sec);	
}
/**
 * Wrapper function used to allocate the nescessary data objects
 * @param N Array containing the dimensions of the system
 */
void AllocateMemory(const long int* N) {

	// Initialize variables
	int tmp1, tmp2, tmp3;
	const long int Ny = N[0];
	const long int Nx = N[1];
	const long int Nx_Fourier = Nx / 2 + 1;


	// --------------------------------
	//  Allocate Field Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}
	run_data->tmp_w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->tmp_w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}
	// Allocate the Fourier phases and amplitudes
	run_data->phi_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->phi_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Phases");
		exit(1);
	}
	run_data->a_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->a_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Amplitudes");
		exit(1);
	}

	// Allocate the Fourier stream funciton
	run_data->psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->psi_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Stream Function");
		exit(1);
	}

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

	// Initialize arrays
	for (int i = 0; i < Ny; ++i) {
		tmp2 = i * (Nx_Fourier);
		for (int j = 0; j < Nx_Fourier; ++j) {
			run_data->w_hat[tmp2 + j]     = 0.0 + 0.0 * I;
			run_data->tmp_w_hat[tmp2 + j] = 0.0 + 0.0 * I;
			run_data->psi_hat[tmp2 + j]   = 0.0 + 0.0 * I;
			proc_data->dw_hat_dt[SYS_DIM * (tmp2 + j) + 0] = 0.0 + 0.0 * I; 
			proc_data->dw_hat_dt[SYS_DIM * (tmp2 + j) + 1] = 0.0 + 0.0 * I; 
		}
	}

	for (int i = 0; i < Ny; ++i) {
		tmp2 = i * (Nx);
		for (int j = 0; j < Nx; ++j) {
			proc_data->nabla_psi[SYS_DIM * (tmp2 + j) + 0] = 0.0; 
			proc_data->nabla_psi[SYS_DIM * (tmp2 + j) + 1] = 0.0; 
		}
	}

	//  Allocate Stats Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w = (double* )fftw_malloc(sizeof(double) * Ny * Nx);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity");
		exit(1);
	}

	// Allocate current Fourier vorticity
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier * SYS_DIM);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	run_data->tmp_u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier * SYS_DIM);
	if (run_data->tmp_u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	run_data->tmp_u_hat_x = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->tmp_u_hat_x == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	run_data->tmp_u_hat_y = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
	if (run_data->tmp_u_hat_y == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}

	// Allocate current Fourier vorticity
	run_data->u = (double* )fftw_malloc(sizeof(double) * Ny * Nx * SYS_DIM);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}
	run_data->tmp_u_x = (double* )fftw_malloc(sizeof(double) * Ny * Nx);
	if (run_data->tmp_u_x == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}
	run_data->tmp_u_y = (double* )fftw_malloc(sizeof(double) * Ny * Nx);
	if (run_data->tmp_u_y == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}
	#if defined(__VEL_GRAD_STATS)
	run_data->tmp_u = (double* )fftw_malloc(sizeof(double) * Ny * Nx * SYS_DIM);
	if (run_data->tmp_u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Temporary Velocity");
		exit(1);
	}
	#endif
	
	//--------------------- Allocate memory for the stats objects
	#if defined(__REAL_STATS) || defined(__GRAD_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS)
	AllocateStatsMemory(N);
	#endif


	// Initialize the arrays
	for (int i = 0; i < Ny; ++i) {
		tmp1 = i * Nx;
		tmp2 = i * Nx_Fourier;
		for (int j = 0; j < Nx; ++j) {
			if (j < Nx_Fourier) {
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 0]     = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 1]     = 0.0 + 0.0 * I;
				run_data->tmp_u_hat[SYS_DIM * (tmp2 + j) + 0] = 0.0 + 0.0 * I;
				run_data->tmp_u_hat[SYS_DIM * (tmp2 + j) + 1] = 0.0 + 0.0 * I;
				run_data->tmp_u_hat_x[tmp2 + j] = 0.0 + 0.0 * I;
				run_data->tmp_u_hat_y[tmp2 + j] = 0.0 + 0.0 * I;
			}
			run_data->w[tmp1 + j] = 0.0;
			run_data->tmp_u_x[tmp1 + j]               = 0.0;
			run_data->tmp_u_y[tmp1 + j]               = 0.0;
			run_data->u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
			run_data->u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
			#if defined(__VEL_GRAD_STATS)
			run_data->tmp_u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
			run_data->tmp_u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
			#endif
		}
	}

	// -------------------------------------
	//  Allocate Full Field & Spectra Data
	// -------------------------------------
	#if defined(__FULL_FIELD) || defined(__SPECTRA) || defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	AllocateFullFieldMemory(N);
	#endif

	// --------------------------------	
	//  Allocate Phase Sync Data
	// --------------------------------
	#if defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	AllocatePhaseSyncMemory(N);
	#endif
}
/**
 * Wrapper function for initializing FFTW plans
 * @param N Array containing the size of the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N) {

	// Initialize variables
	const long int Ny = N[0];
	const long int Nx = N[1];
	const int N_batch[SYS_DIM] = {Ny, Nx};

	// Initialize Fourier Transforms -> only on a single thread as plan creation and destruction are not thread safe -> plan execution is thread safe
	sys_vars->fftw_2d_dft_c2r = fftw_plan_dft_c2r_2d(Ny, Nx, run_data->w_hat, run_data->w, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic C2R FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
	}

	sys_vars->fftw_2d_dft_r2c = fftw_plan_dft_r2c_2d(Ny, Nx, run_data->w, run_data->w_hat, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_r2c == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic R2C FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
	}

	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_c2r = fftw_plan_many_dft_c2r(SYS_DIM, N_batch, SYS_DIM, run_data->u_hat, NULL, SYS_DIM, 1, run_data->u, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch C2R FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
	}

	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_r2c = fftw_plan_many_dft_r2c(SYS_DIM, N_batch, SYS_DIM, run_data->u, NULL, SYS_DIM, 1, run_data->u_hat, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_batch_r2c == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch R2C FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
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
	fftw_free(run_data->phi_k);
	fftw_free(run_data->a_k);
	fftw_free(run_data->tmp_w_hat);
	fftw_free(run_data->time);
	fftw_free(run_data->psi_hat);
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}
	fftw_free(proc_data->nabla_psi);
	fftw_free(proc_data->dw_hat_dt);
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	fftw_free(run_data->tmp_u_hat);
	fftw_free(run_data->tmp_u_hat_x);
	fftw_free(run_data->tmp_u_hat_y);
	fftw_free(run_data->tmp_u_x);
	fftw_free(run_data->tmp_u_y);
	#if defined(__GRAD_STATS)
	fftw_free(run_data->tmp_u);
	#endif

	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
	FreeStatsObjects();
	#endif

	#if defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS)
	FreePhaseSyncObjects();
	#endif
	
	#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC) || defined(__SEC_PHASE_SYNC_STATS) || defined(__SPECTRA) || defined(__ENST_FLUX) || defined(__ENRG_FLUX)
	FreeFullFieldObjects();
	#endif
	
	// --------------------------------
	//  Free FFTW Plans
	// --------------------------------
	// Destroy FFTW plans -> only on a single thread as plan creation and destruction are not thread safe -> plan execution is thread safe
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_r2c);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_r2c);


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
