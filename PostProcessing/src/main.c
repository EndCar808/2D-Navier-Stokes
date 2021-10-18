/**
* @file main.c 
* @author Enda Carroll
* @date Sept 2021
* @brief Main file for post processing solver data from the 2D Navier stokes psuedospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <time.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"
#include "post_proc.h"
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// Define the global points that will be pointed to the global structs
runtime_data_struct*      run_data;
system_vars_struct*       sys_vars;
HDF_file_info_struct*    file_info;
postprocess_data_struct* proc_data;
stats_data_struct*      stats_data;
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Initialize variables
	int tmp, tmp1, tmp2;
	int indx;
	double k_sqr, k_sqr_fac, phase, amp;
	double w_max, w_min;
	double u_max, u_min;

	
	// --------------------------------
	//  Create Global Stucts
	// --------------------------------
	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	HDF_file_info_struct   HDF_file_info;
	postprocess_data_struct postproc_data;
	stats_data_struct statistics_data;

	// Point the global pointers to these structs
	run_data   = &runtime_data;
	sys_vars   = &system_vars;
	file_info  = &HDF_file_info;
	proc_data  = &postproc_data;
	stats_data = &statistics_data;

	// --------------------------------
	//  Get Command Line Arguements
	// --------------------------------
	if ((GetCMLArgs(argc, argv)) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line aguments, check utils.c file for details\n");
		exit(1);
	}

	// --------------------------------
	//  Open Input File and Get Data
	// --------------------------------
	OpenInputAndInitialize(); 
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// --------------------------------
	//  Open Output File
	// --------------------------------
	OpenOutputFile();

	// --------------------------------
	//  Allocate Processing Memmory
	// --------------------------------
	AllocateMemory(sys_vars->N);

	InitializeFFTWPlans(sys_vars->N);

	//////////////////////////////
	// Begin Snapshot Processing
	//////////////////////////////
	printf("\nStarting Snapshot Processing:\n");
	for (int s = 0; s < sys_vars->num_snaps; ++s) {
		
		// Print update to screen
		printf("Snapshot: %d\n", s);
	
		// --------------------------------
		//  Read in Data
		// --------------------------------
		ReadInData(s);

		// --------------------------------
		//  Real Space Stats
		// --------------------------------
		// Get min and max data for histogram limits
		w_max = 0.0;
		w_min = 1e8;
		gsl_stats_minmax(&w_min, &w_max, run_data->w, 1, Nx * Ny);
		u_max = 0.0; 
		u_min = 1e8;
		gsl_stats_minmax(&u_min, &u_max, run_data->u, 1, Nx * (Ny + 2) * SYS_DIM);
			
		// Set histogram ranges
		if (gsl_histogram_set_ranges_uniform(stats_data->w_pdf, w_min, w_max) != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
			exit(1);
		}
		if (gsl_histogram_set_ranges_uniform(stats_data->u_pdf, u_min, u_max) != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
			exit(1);
		}

		// Update histograms
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Add current value to appropriate bin
				if (gsl_histogram_increment(stats_data->w_pdf, run_data->w[indx]) != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
					exit(1);
				}
				if (gsl_histogram_increment(stats_data->u_pdf, run_data->u[SYS_DIM * indx + 0]) != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
					exit(1);
				}
				if (gsl_histogram_increment(stats_data->u_pdf, run_data->u[SYS_DIM * indx + 1]) != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
					exit(1);
				}
			}
		}

		// --------------------------------
		//  Full Field Data
		// --------------------------------
		for (int i = 0; i < Nx; ++i) {
			if (abs(run_data->k[0][i]) < sys_vars->kmax) {
				tmp  = i * Ny_Fourier;	
				tmp1 = (sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
				tmp2 = (sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;
					if (abs(run_data->k[1][j] < sys_vars->kmax)) {

						// Compute |k|^2
						k_sqr     = (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]); 
						if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
							k_sqr_fac = 1.0 / k_sqr;
						}
						else {
							k_sqr_fac = 0.0;	
						}

						// Compute data
						phase = fmod(carg(run_data->w_hat[indx]) + 2.0 * M_PI, 2.0 * M_PI);
						amp   = cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));

						// fill the full field phases and spectra
		 				if (sqrt(k_sqr) < sys_vars->kmax) {
		 					// No conjugate for ky = 0
		 					if (run_data->k[1][j] == 0) {
		 						// proc_data->k_full[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = phase;
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp * k_sqr_fac;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
		 					}
		 					else {
		 						// Fill data and its conjugate
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = phase;
		 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = fmod(-phase + 2.0 * M_PI, 2.0 * M_PI);
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
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 					}
		 					else {	
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = -50.0;
		 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = -50.0;
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
		// --------------------------------
		//  Write Data to File
		// --------------------------------
		WriteDataToFile(run_data->time[s], s);
	}
	//////////////////////////////
	// End Snapshot Processing
	//////////////////////////////

	
	// --------------------------------
	//  Clean Up
	// --------------------------------
	// Free allocated memory
	FreeMemoryAndCleanUp();


	return 0;
}
/**
 * Wrapper function used to allocate the nescessary data objects
 * @param N Array containing the dimensions of the system
 */
void AllocateMemory(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Compute maximum wavenumber
	sys_vars->kmax    = (int) (Nx / 3);	


	// --------------------------------
	//  Allocate Field Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}

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
	run_data->u = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2) * SYS_DIM);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}

	// Allocate memory for the full field phases
	proc_data->phases = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->phases == NULL) {
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

	// // Allocate memory for the full field wavenumbers
	// proc_data->k_full = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	// if (proc_data->k_full == NULL) {
	// 	fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Energy");
	// 	exit(1);
	// }	

	// --------------------------------
	//  Allocate Stats Data
	// --------------------------------
	// Allocate vorticity histograms
	stats_data->w_pdf = gsl_histogram_alloc(N_BINS);
	if (stats_data->w_pdf == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Histogram");
		exit(1);
	}	

	// Allocate velocity histograms
	stats_data->u_pdf = gsl_histogram_alloc(N_BINS);
	if (stats_data->u_pdf == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Histogram");
		exit(1);
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

	// Initialize Fourier Transforms
	sys_vars->fftw_2d_dft_c2r = fftw_plan_dft_c2r_2d(Nx, Ny, run_data->w_hat, run_data->w, FFTW_ESTIMATE);
	
	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_c2r = fftw_plan_many_dft_c2r(SYS_DIM, N_batch, SYS_DIM, run_data->u_hat, NULL, SYS_DIM, 1, run_data->u, NULL, SYS_DIM, 1, FFTW_MEASURE);
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
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(proc_data->phases);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	gsl_histogram_free(stats_data->w_pdf);
	gsl_histogram_free(stats_data->u_pdf);

	// --------------------------------
	//  Free FFTW Plans
	// --------------------------------
	// Destroy FFTW plans
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);

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
//  End of File
// ---------------------------------------------------------------------