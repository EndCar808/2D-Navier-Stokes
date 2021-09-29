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
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// Define the global points that will be pointed to the global structs
runtime_data_struct*      run_data;
system_vars_struct*       sys_vars;
HDF_file_info_struct*    file_info;
postprocess_data_struct *proc_data;
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Initialize variables
	int tmp, tmp1, tmp2;
	int indx, indx1, indx2;
	double k_sqr, phase, amp;
	
	// --------------------------------
	//  Create Global Stucts
	// --------------------------------
	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	HDF_file_info_struct   HDF_file_info;
	postprocess_data_struct postproc_data;

	// Point the global pointers to these structs
	run_data  = &runtime_data;
	sys_vars  = &system_vars;
	file_info = &HDF_file_info;
	proc_data = &postproc_data;

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
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}

	// Allocate memory for the full field phases
	sys_vars->kmax    = (int) (Nx / 3);
	proc_data->phases = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1));
	if (proc_data->phases == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field enstrophy spectrum
	proc_data->enst = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1));
	if (proc_data->enst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Enstrophy");
		exit(1);
	}	

	// Allocate memory for the full field enrgy spectrum
	proc_data->enrg = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1));
	if (proc_data->enrg == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Energy");
		exit(1);
	}	


	//////////////////////////////
	// End Integration
	//////////////////////////////
	for (int s = 0; s < sys_vars->num_snaps; ++s) {
		
		// Print update to screen
		printf("Snapshot: %d\n", s);
	
		// --------------------------------
		//  Read in Data
		// --------------------------------
		ReadInData(s);

		// --------------------------------
		//  Process Data
		// --------------------------------
		for (int i = 0; i < Nx; ++i) {
			if (abs(run_data->k[0][i]) < sys_vars->kmax) {
				tmp  = i * Ny_Fourier;
				tmp1 = (sys_vars->kmax - 1 + run_data->k[0][i]) * Ny_Fourier;
				tmp2 = (sys_vars->kmax - 1 - run_data->k[0][i]) * Ny_Fourier;
				for (int j = 0; j < Ny_Fourier; ++j) {
					if (abs(run_data->k[1][j] < sys_vars->kmax)) {
						indx = tmp + j;

						// Compute |k|^2
						k_sqr = (double)run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]; 

						// Compute data
						phase = carg(run_data->w_hat[indx]);
						amp   = cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));

						// fill the full field phases and spectra
		 				if (sqrt(k_sqr) < sys_vars->kmax) {
		 					// No conjugate for ky = 0
		 					if (run_data->k[1][j] == 0) {
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = fmod(phase, 2.0 * M_PI);
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp / k_sqr + 1e-50;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
		 					}
		 					else {
		 						// Fill data and its conjugate
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = fmod(phase, 2.0 * M_PI);
		 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = fmod(-phase, 2.0 * M_PI);
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp / k_sqr + 1e-50;
		 						proc_data->enrg[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = amp / k_sqr + 1e-50;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
		 						proc_data->enst[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = amp;
		 					}
		 				}
		 				else {
		 					// All dealiased modes set to zero
							if (run_data->k[1][j] == 0) {
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 0.0;
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 					}
		 					else {	
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 0.0;
		 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = 0.0;
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
	// End Integration
	//////////////////////////////


	
	// --------------------------------
	//  Clean Up
	// --------------------------------
	fftw_free(run_data->w_hat);
	fftw_free(proc_data->phases);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}




	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------