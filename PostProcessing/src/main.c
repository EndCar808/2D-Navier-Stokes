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
	int gsl_status;
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
		#ifdef __REAL_STATS
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

		// Set histogram ranges
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

		// Update histograms
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
		#endif

		// --------------------------------
		//  Spectra Data
		// --------------------------------
		#ifdef __SPECTRA
		// Compute the enstrophy spectrum
		EnstrophySpectrum();
		#endif

		// --------------------------------
		//  Full Field Data
		// --------------------------------
		#ifdef __FULL_FIELD
		for (int i = 0; i < Nx; ++i) {
			if (abs(run_data->k[0][i]) < sys_vars->kmax) {
				tmp  = i * Ny_Fourier;	
				tmp1 = (sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
				tmp2 = (sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;
					if (abs(run_data->k[1][j] < sys_vars->kmax)) {

						// Compute |k|^2
						k_sqr = (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]); 
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
		#endif
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
	int tmp1, tmp2, tmp3;
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

	// Initialize arrays
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		proc_data->enst_spec[i] = 0.0;
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
	//  Compute spectrum
	// --------------------------------	
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute spectrum index/bin
			spec_indx = (int )round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

			if ((j == 0) || (j == Ny_Fourier - 1)) {
				proc_data->enst_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}
			else {
				proc_data->enst_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
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
	#ifdef __REAL_STATS
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	#endif
	#ifdef __FULL_FIELD
	fftw_free(proc_data->phases);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	#endif
	#ifdef __SPECTRA
	fftw_free(proc_data->enst_spec);
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
//  End of File
// ---------------------------------------------------------------------