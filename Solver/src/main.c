/**
* @file main.c 
* @author Enda Carroll
* @date Jun 2021
* @brief Main file for calling the pseudospectral solver on the 2D Ekman Navier Stokes in vorticity form
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "solver.h"
#include "utils.h"

// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// Define the global points that will be pointed to the global structs
runtime_data_struct*      run_data;
system_vars_struct*       sys_vars;
HDF_file_info_struct*    file_info;

// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	HDF_file_info_struct   HDF_file_info;
	
	// Point the global pointers to these structs
	run_data  = &runtime_data;
	sys_vars  = &system_vars;
	file_info = &HDF_file_info;


	//////////////////////////////////
	// Initialize MPI section
	MPI_Init(&argc, &argv);
	//////////////////////////////////
	
	// Get the number of active processes and their rank and print to screen
	MPI_Comm_size(MPI_COMM_WORLD, &(sys_vars->num_procs));      
	MPI_Comm_rank(MPI_COMM_WORLD, &(sys_vars->rank));  
	if ( !(sys_vars->rank) ) {
		printf("\nTotal number of MPI tasks running: %d\n\n", sys_vars->num_procs);
	}

	// Initialize FFTW MPI interface - must be called after MPI_Init but before anything else in FFTW
	fftw_mpi_init();

	

	//////////////////////////////////
	// Call Solver
	//////////////////////////////////
	SpectralSolve();
	//////////////////////////////////
	// Call Solver
	//////////////////////////////////
	


	// Cleanup FFTW MPI interface - Calls the serial fftw_cleanup function also
	fftw_mpi_cleanup();    

	//////////////////////////////////
	// Exit MPI scetion
	MPI_Finalize();
	//////////////////////////////////
	



	// Return statement
	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
