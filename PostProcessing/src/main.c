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
	//  Begin Timing
	// --------------------------------
	// Initialize timing counter
	clock_t begin = omp_get_wtime();

	// --------------------------------
	//  Get Command Line Arguements
	// --------------------------------
	if ((GetCMLArgs(argc, argv)) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line aguments, check utils.c file for details\n");
		exit(1);
	}

	// Set the number of threads and get thread IDs
	omp_set_num_threads(sys_vars->num_threads);
	#pragma omp parallel
	{
		if (!(sys_vars->thread_id)) {
			printf("\nThreads Active: "CYAN"%d"RESET"\n", sys_vars->num_threads);
		}
	}

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

	//////////////////////////////
	// Begin Snapshot Processing
	//////////////////////////////
	printf("\nStarting Snapshot Processing:\n");
	for (int s = 0; s < 1; ++s) { // sys_vars->num_snaps
		
		// Print update to screen
		printf("Snapshot: %d\n", s);
	
		// --------------------------------
		//  Read in Data
		// --------------------------------
		ReadInData(s);

		// --------------------------------
		//  Real Space Stats
		// --------------------------------
		#if defined(__REAL_STATS) || defined(__VEL_INCR_STATS) || defined(__STR_FUNC_STATS)
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
		#if defined(__ENST_FLUX) || defined(__SEC_PHASE_SYNC)
        EnstrophyFluxSpectrum(s);
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
		#if defined(__SEC_PHASE_SYNC) 
		SectorPhaseOrder(s);
		// SectorPhaseOrderPar(s);		
		#endif

		// --------------------------------
		//  Write Data to File
		// --------------------------------
		WriteDataToFile(run_data->time[s], s);
	}
	///////////////////////////////
	// End Snapshot Processing
	///////////////////////////////

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

	// --------------------------------
	//  End Timing
	// --------------------------------
	// Finish timing
	clock_t end = omp_get_wtime();

	// calculate execution time
	double time_spent = (double)(end - begin);
	int hh = (int) time_spent / 3600;
	int mm = ((int )time_spent - hh * 3600) / 60;
	int ss = time_spent - hh * 3600 - mm * 60;
	printf("\n\nTotal Execution Time: ["CYAN"%5.10lf"RESET"] --> "CYAN"%d"RESET" hrs : "CYAN"%d"RESET" mins : "CYAN"%d"RESET" secs\n\n", time_spent, hh, mm, ss);


	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------