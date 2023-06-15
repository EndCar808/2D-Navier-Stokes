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

	// --------------------------------
	//  Allocate Processing Memmory
	// --------------------------------
	AllocateMemory(sys_vars->N);

	InitializeFFTWPlans(sys_vars->N);
	// --------------------------------
	//  Perform Precomputations
	// --------------------------------
	#if defined(__GRAD_STATS) || defined(__VEL_INC_STATS) || (defined(__SEC_PHASE_SYNC) && defined(__SEC_PHASE_SYNC_FLUX_STATS))
	Precompute();
	#endif

	//////////////////////////////
	// Begin Snapshot Processing
	//////////////////////////////
	printf("\n\nStarting Snapshot Processing:\n");
	for (int s = 0; s < sys_vars->num_snaps; ++s) { 
		
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
		#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC)
		FullFieldData();
		#endif

		// --------------------------------
		//  Phase Sync
		// --------------------------------
		#if defined(__PHASE_SYNC)
		PhaseSync(s);
		#endif
		#if defined(__SEC_PHASE_SYNC) 
		PhaseSyncSector(s, 0);
		#endif

		// --------------------------------
		//  Write Data to File
		// --------------------------------
		WriteDataToFile(run_data->time[s], s);


		// End timer for current loop
		double loop_end = omp_get_wtime();

		// Print update to screen
		printf("Snapshot: %d/%ld\tTime: %g(s)\n", s + 1, sys_vars->num_snaps, (loop_end - loop_begin));
	}
	///////////////////////////////
	// End Snapshot Processing
	///////////////////////////////

	// ---------------------------------
	//  Phase Sync Conditional Stats
	// ---------------------------------
	ComputePhaseSyncConditionalStats();


	// ---------------------------------
	//  Final Write of Data and Close 
	// ---------------------------------
	// Write any remaining datasets to output file
	FinalWriteAndClose();
	
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
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	int r;
	int N_max_incr = (int) (GSL_MIN(Nx, Ny) / 2);
	int increment[NUM_INCR] = {1, N_max_incr};
	double long_increment, trans_increment;
	double norm_fac = 1.0 / (Nx * Ny);
	double std_u, std_w;

	// Print to screen that a pre computation step is need and begin timeing it
	printf("\n["YELLOW"NOTE"RESET"] --- Performing a precomputation step...\n");
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
		#if defined(__SEC_PHASE_SYNC) && defined(__SEC_PHASE_SYNC_FLUX_STATS)
		// Compute full field needed for the sector phase sync data
		FullFieldData();

		// Call sector sync in pre compute mode
		PhaseSyncSector(s, PRE_COMPUTE);
		#endif

		// --------------------------------
		// Precompute Stats
		// --------------------------------
		// Compute the graident in Fourier space
		#if defined(__GRAD_STATS)
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
				// printf("grad_u_x: %1.16lf + %1.16lf I\tgrad_u_y:  %1.16lf + %1.16lf I\n", creal(proc_data->grad_u_hat[SYS_DIM * indx + 0]), cimag(proc_data->grad_u_hat[SYS_DIM * indx + 0]), creal(proc_data->grad_u_hat[SYS_DIM * indx + 1]), cimag(proc_data->grad_u_hat[SYS_DIM * indx + 1]));
			}
		}
		// Perform inverse transform to get the gradients in real space - no need to presave grad_w_hat & grad_u_hat, wont be used again
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_w_hat, proc_data->grad_w);
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, proc_data->grad_u_hat, proc_data->grad_u);
		#endif

		#if defined(__GRAD_STATS) || defined(__VEL_INC_STATS)
		// Loop over real space
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;


				#if defined(__GRAD_STATS)
				// Normalize the gradients 
				proc_data->grad_w[SYS_DIM * indx + 0] *= norm_fac;
				proc_data->grad_w[SYS_DIM * indx + 1] *= norm_fac;
				proc_data->grad_u[SYS_DIM * indx + 0] *= norm_fac;
				proc_data->grad_u[SYS_DIM * indx + 1] *= norm_fac;


				// Add gradients to stats accumulators
				for (int i = 0; i < SYS_DIM + 1; ++i) {
					if (i < SYS_DIM) {
						// Add to the individual directions accumulators
						gsl_rstat_add(proc_data->grad_w[SYS_DIM * indx + i], stats_data->r_stat_grad_w[i]);
						gsl_rstat_add(proc_data->grad_u[SYS_DIM * indx + i], stats_data->r_stat_grad_u[i]);
					}
					else {
						// Add to the combined accumulator
						for (int j = 0; j < SYS_DIM; ++j) {
							gsl_rstat_add(proc_data->grad_w[SYS_DIM * indx + j], stats_data->r_stat_grad_w[SYS_DIM]);
							gsl_rstat_add(proc_data->grad_u[SYS_DIM * indx + j], stats_data->r_stat_grad_u[SYS_DIM]);	
						}
					}
				}
				#endif

				#if defined(__VEL_INC_STATS)
				// Compute velocity increments and update histograms
				for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
					// Get the current increment
					r = increment[r_indx];

					//------------- Get the longitudinal and transverse Velocity increments
					long_increment  = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 0] - run_data->u[SYS_DIM * (i * Ny + j) + 0];
					trans_increment = run_data->u[SYS_DIM * (((i + r) % Nx) * Ny + j) + 1] - run_data->u[SYS_DIM * (i * Ny + j) + 1];

					// Update the stats accumulators
					gsl_rstat_add(long_increment, stats_data->r_stat_vel_incr[0][r_indx]);
					gsl_rstat_add(trans_increment, stats_data->r_stat_vel_incr[1][r_indx]);


					//------------- Get the longitudinal and transverse Vorticity increments
					long_increment  = run_data->w[((i + r) % Nx) * Ny + j] - run_data->w[i * Ny + j];
					trans_increment = run_data->w[i * Ny + ((j + r) % Ny)] - run_data->w[i * Ny + j];

					// Update the stats accumulators
					gsl_rstat_add(long_increment, stats_data->r_stat_vort_incr[0][r_indx]);
					gsl_rstat_add(trans_increment, stats_data->r_stat_vort_incr[1][r_indx]);			
				}
				#endif
			}
		}
		#endif
	}

	// --------------------------------
	// Initialize Gradient Histograms
	// --------------------------------
	#if defined(__GRAD_STATS)
	for (int i = 0; i < SYS_DIM + 1; ++i) {
		// Get the std of the gradients
		std_u = gsl_rstat_sd(stats_data->r_stat_grad_u[i]);
		std_w = gsl_rstat_sd(stats_data->r_stat_grad_w[i]);


		// Velocity gradients
		gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vel_grad[i], -BIN_LIM * std_u, BIN_LIM * std_u);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Gradient Increments");
			exit(1);
		}
		// Vorticity gradients
		gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vort_grad[i], -BIN_LIM * std_w, BIN_LIM * std_w);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Gradient Increments");
			exit(1);
		}
	}
	#endif

	// --------------------------------
	// Initialize Increment Histograms
	// --------------------------------
	#if defined(__VEL_INC_STATS)
	// Set the bin limits for the velocity increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			// Get the std of the incrments
			std_u = gsl_rstat_sd(stats_data->r_stat_vel_incr[i][j]);
			std_w = gsl_rstat_sd(stats_data->r_stat_vort_incr[i][j]);


			// Velocity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->vel_incr[i][j], -BIN_LIM * std_u, BIN_LIM * std_u);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Increments");
				exit(1);
			}
			// Vorticity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_incr[i][j], -BIN_LIM *std_w, BIN_LIM * std_w);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Increments");
				exit(1);
			}
		}
	}
	#endif

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
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;

	// Compute maximum wavenumber
	sys_vars->kmax = (int) (Nx / 3.0);
	
	// Get the various kmax variables
	sys_vars->kmax_sqr   = pow(sys_vars->kmax, 2.0);
	sys_vars->kmax_C   	 = (int) ceil(sys_vars->kmax_frac * sys_vars->kmax);
	sys_vars->kmax_C_sqr = pow(sys_vars->kmax_C, 2.0);

	// --------------------------------
	//  Allocate Field Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}
	run_data->tmp_w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->tmp_w_hat == NULL) {
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
			run_data->w_hat[tmp2 + j]     = 0.0 + 0.0 * I;
			run_data->tmp_w_hat[tmp2 + j] = 0.0 + 0.0 * I;
			run_data->psi_hat[tmp2 + j]   = 0.0 + 0.0 * I;
		}
	}

	//  Allocate Stats Data
	// --------------------------------
	#if defined(__REAL_STATS) || defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS) || defined(__VORT_REAL) || defined(__MODES) || defined(__REALSPACE)
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
	run_data->tmp_u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	if (run_data->tmp_u_hat == NULL) {
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
	run_data->tmp_u = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
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
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		tmp2 = i * Ny_Fourier;
		for (int j = 0; j < Ny; ++j) {
			if (j < Ny_Fourier) {
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 0]     = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 1]     = 0.0 + 0.0 * I;
				run_data->tmp_u_hat[SYS_DIM * (tmp2 + j) + 0] = 0.0 + 0.0 * I;
				run_data->tmp_u_hat[SYS_DIM * (tmp2 + j) + 1] = 0.0 + 0.0 * I;
			}
			run_data->w[tmp1 + j] = 0.0;
			run_data->u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
			run_data->u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
			#if defined(__GRAD_STATS)
			run_data->tmp_u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
			run_data->tmp_u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
			#endif
		}
	}
	#endif

	// -------------------------------------
	//  Allocate Full Field & Spectra Data
	// -------------------------------------
	#if defined(__FULL_FIELD) || defined(__SPECTRA) || defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC)
	AllocateFullFieldMemory(N);
	#endif

	// --------------------------------	
	//  Allocate Phase Sync Data
	// --------------------------------
	#if defined(__SEC_PHASE_SYNC)
	AllocatePhaseSyncMemory(N);
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

	// Initialize Fourier Transforms
	sys_vars->fftw_2d_dft_c2r = fftw_plan_dft_c2r_2d(Nx, Ny, run_data->w_hat, run_data->w, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic C2R FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
	}
	sys_vars->fftw_2d_dft_r2c = fftw_plan_dft_r2c_2d(Nx, Ny, run_data->w, run_data->w_hat, FFTW_MEASURE);
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
	fftw_free(run_data->tmp_w_hat);
	fftw_free(run_data->time);
	fftw_free(run_data->psi_hat);
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}
	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC) || defined(__GRAD_STATS) || defined(__VORT_REAL) || defined(__MODES) || defined(__REALSPACE)
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	fftw_free(run_data->tmp_u_hat);
	#if defined(__GRAD_STATS)
	fftw_free(run_data->tmp_u);
	#endif
	#endif

	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__STR_FUNC_STATS) || defined(__GRAD_STATS)
	FreeStatsObjects();
	#endif

	#if defined(__SEC_PHASE_SYNC)
	FreePhaseSyncObjects();
	#endif
	
	#if defined(__FULL_FIELD) || defined(__SEC_PHASE_SYNC) || defined(__SPECTRA) || defined(__ENST_FLUX) || defined(__ENRG_FLUX)
	FreeFullFieldObjects();
	#endif
	
	// --------------------------------
	//  Free FFTW Plans
	// --------------------------------
	// Destroy FFTW plans
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
