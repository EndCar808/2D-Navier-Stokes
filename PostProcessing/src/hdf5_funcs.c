/**
* @file hdf5_funcs.c  
* @author Enda Carroll
* @date Sept 2021
* @brief File containing HDF5 function wrappers for creating, opening, wrtining to and closing input/output HDF5 file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to open the input file, read in simulation data and initialize some of the system parameters
 */
void OpenInputAndInitialize(void) {

	// Initialize variables
	hid_t dset;
	hid_t dspace;
	herr_t status;
	hsize_t Dims[SYS_DIM];
	int snaps = 0;
	char group_string[64];
	double tmp_time;

	// --------------------------------
	//  Create Complex Datatype
	// --------------------------------
	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();

	// --------------------------------
	//  Get Input File Path
	// --------------------------------
	if (strcmp(file_info->input_dir, "NONE") != 0) {
		// Check input file mode
		if (!(file_info->input_file_only)) {
			// If input folder construct input file path
			strcpy(file_info->input_file_name, file_info->input_dir);
			strcat(file_info->input_file_name, "Main_HDF_Data.h5");
		}
		else {
			// If file only mode construct input file path
			strcpy(file_info->input_file_name, file_info->input_dir);
		}

		// Print input file path to screen
		printf("\nInput File: "CYAN"%s"RESET"\n\n", file_info->input_file_name);
	}
	else {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- No input directory/file provided. Please provide input directory/file - see utils.c\n-->> Exiting...\n");
		exit(1);
	}
	
	

	// --------------------------------
	//  Open File
	// --------------------------------
	// Check if file exists
	if (access(file_info->input_file_name, F_OK) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Input file ["CYAN"%s"RESET"] does not exist\n-->> Exiting...\n", file_info->input_file_name);
		exit(1);
	}
	else {
		// Open file
		file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->input_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name);
			exit(1);
		}
	}

	// --------------------------------
	//  Get Number of Snaps
	// --------------------------------
	// Count the number of snapshot groups in file
	printf("Checking Number of Snapshots ");
	for(int i = 0; i < (int) 1e6; ++i) {
		// Check for snap
		sprintf(group_string, "/Iter_%05d", i);	
		if(H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
			snaps++;
		}
	}
	// Print total number of snaps to screen
	printf("\nTotal Snapshots: ["CYAN"%d"RESET"]\n\n", snaps);

	// Save the total number of snaps
	sys_vars->num_snaps = snaps;

	// --------------------------------
	//  Get Time series
	// --------------------------------
	// Allocate memory
	run_data->time = (double* )fftw_malloc(sizeof(double) * sys_vars->num_snaps);
	if(run_data->time == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "time");
		exit(1);
	}

	// Loop through group snapshots and read in time value
	for (int i = 0; i < sys_vars->num_snaps; ++i) {
		sprintf(group_string, "/Iter_%05d", i);	
		if(H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
			// Read in the time attribute
			H5LTget_attribute_double(file_info->input_file_handle, group_string, "TimeValue", &tmp_time);	
			run_data->time[i] = tmp_time;
		}
	}

	// --------------------------------
	//  Get System Dimensions
	// --------------------------------
	// Open dataset
	sprintf(group_string, "/Iter_%05d/w_hat", 0);	
	if((dset = H5Dopen(file_info->input_file_handle, group_string , H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset ["CYAN"%s"RESET"]\n-->> Exiting...\n", group_string);
		exit(1);
	}

	// Get dataspace handle
	dspace = H5Dget_space(dset); 

	// Get dims from dataspace
	if(H5Sget_simple_extent_ndims(dspace) != SYS_DIM) {
 	  fprintf(stderr, "\n["RED"ERROR"RESET"] --- Number of dimensions in HDF5 Datasets ["CYAN"%s"RESET"] is not 2!\n-->> Exiting...\n", group_string);
		exit(1);		 
	}
	if((H5Sget_simple_extent_dims(dspace, Dims, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- unable to get data extents(dimensions) from HDF5 dataset ["CYAN"%s"RESET"]\n-->> Exiting...\n", group_string);
		exit(1);		
	}	

	// Record system dims
	sys_vars->N[0] = (long int)Dims[0];
	sys_vars->N[1] = ((long int)Dims[1] - 1) * 2;

	// Close identifiers
	status = H5Dclose(dset);
	status = H5Sclose(dspace);
	
	// --------------------------------
	//  Read In/Initialize Space Arrays
	// --------------------------------
	// Allocate memory for real space
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->N[0]);
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * sys_vars->N[1]);
	if(run_data->x[0] == NULL || run_data->x[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "collocation points");
		exit(1);
	}

	// Allocate memory for fourier space
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->N[0]);
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * (sys_vars->N[1] / 2 + 1));
	if(run_data->k[0] == NULL || run_data->k[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "wavenumber list");
		exit(1);
	}

	// Read in space arrays if they exist in input file if not intialize them
	if((H5Lexists(file_info->input_file_handle, "x", H5P_DEFAULT) > 0) && (H5Lexists(file_info->input_file_handle, "y", H5P_DEFAULT) > 0) && (H5Lexists(file_info->input_file_handle, "kx", H5P_DEFAULT) > 0) && (H5Lexists(file_info->input_file_handle, "ky", H5P_DEFAULT) > 0)) {
		// Read in real space arrays
		if(H5LTread_dataset(file_info->input_file_handle, "x", H5T_NATIVE_DOUBLE, run_data->x[0]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "x");
			exit(1);	
		}
		if(H5LTread_dataset(file_info->input_file_handle, "y", H5T_NATIVE_DOUBLE, run_data->x[1]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "y");
			exit(1);
		}

		// Read in Fourier space arrays
		if(H5LTread_dataset(file_info->input_file_handle, "kx", H5T_NATIVE_INT, run_data->k[0]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "kx");
			exit(1);	
		}
		if(H5LTread_dataset(file_info->input_file_handle, "ky", H5T_NATIVE_INT, run_data->k[1]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ky");
			exit(1);
		}
	}
	else {
		InitializeSpaceVariables(run_data->x, run_data->k, sys_vars->N);
	}

	// --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	status = H5Fclose(file_info->input_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name, "initial");
		exit(1);		
	}
}
/**
 * Function to read in the solver data for the current snapshot. If certain data does not exist but is needed it is computed here e.g. real space vorticity and velocity
 * @param snap_indx The index of the currrent snapshot
 */
void ReadInData(int snap_indx) {

	// Initialize variables
	int indx, tmp;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	char group_string[64];
	hid_t dset;
	herr_t status;

	// --------------------------------
	//  Open File
	// --------------------------------
	// Check if file exists
	if (access(file_info->input_file_name, F_OK) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Input file ["CYAN"%s"RESET"] does not exist\n-->> Exiting...\n", file_info->input_file_name);
		exit(1);
	}
	else {
		// Open file
		file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->input_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name);
			exit(1);
		}
	}

	// --------------------------------
	//  Read in Fourier Vorticity
	// --------------------------------
	// Open Fourier space vorticity
	sprintf(group_string, "/Iter_%05d/w_hat", snap_indx);	
	dset = H5Dopen (file_info->input_file_handle, group_string, H5P_DEFAULT);
	if (dset < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", snap_indx);
		exit(1);		
	} 
	// Read in Fourier space vorticity
	if(H5LTread_dataset(file_info->input_file_handle, group_string, file_info->COMPLEX_DTYPE, run_data->w_hat) < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", snap_indx);
		exit(1);	
	}

	// --------------------------------
	//  Read in Real Vorticity
	// --------------------------------
	#ifdef __REAL_STATS
	// If Real Space vorticity exists read it in
	sprintf(group_string, "/Iter_%05d/w", snap_indx);	
	if (H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
		dset = H5Dopen (file_info->input_file_handle, group_string, H5P_DEFAULT);
		if (dset < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w", snap_indx);
			exit(1);		
		} 
		// Read in Real space vorticity
		if(H5LTread_dataset(file_info->input_file_handle, group_string, H5T_NATIVE_DOUBLE, run_data->w) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w", snap_indx);
			exit(1);	
		}
	
		// Real space vorticity exists
		sys_vars->REAL_VORT_FLAG = 1; 
	}
	else {
		// Real space vorticity exists
		sys_vars->REAL_VORT_FLAG = 0; 

		// Get the real space vorticity from the Fourier space
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->w_hat, run_data->w);
		for (int i = 0; i < Nx; ++i) {	
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize the vorticity
				run_data->w[indx] /= (Nx * Ny);
			}
		}
	}
	#endif

	// --------------------------------
	//  Read in Real Velocity
	// --------------------------------
	#ifdef __REAL_STATS
	// If Real Space Velocity exists read it in
	sprintf(group_string, "/Iter_%05d/u", snap_indx);	
	if (H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
		dset = H5Dopen (file_info->input_file_handle, group_string, H5P_DEFAULT);
		if (dset < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "u", snap_indx);
			exit(1);		
		} 
		// Read in Real space vorticity
		if(H5LTread_dataset(file_info->input_file_handle, group_string, H5T_NATIVE_DOUBLE, run_data->u) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "u", snap_indx);
			exit(1);	
		}

		// Real space vorticity exists
		sys_vars->REAL_VEL_FLAG = 1; 
	}
	else {
		fftw_complex k_sqr;
		
		// Real space vorticity exists
		sys_vars->REAL_VEL_FLAG = 0; 

		// Compute the Fourier velocity
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
					// Compute the prefactor
					k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
					
					// Compute the Fourier velocity
					run_data->u_hat[SYS_DIM * indx + 0] = k_sqr * (double)run_data->k[1][j] * run_data->w_hat[indx];
					run_data->u_hat[SYS_DIM * indx + 1] = -k_sqr * (double)run_data->k[0][i] * run_data->w_hat[indx];
				}
				else {
					run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
				}
			}
		}

		// Transform back to Real space and Normalize
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, run_data->u_hat, run_data->u);
		for (int i = 0; i < Nx; ++i) {	
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize the velocity
				run_data->u[SYS_DIM * indx + 0] /= (Nx * Ny);
				run_data->u[SYS_DIM * indx + 1] /= (Nx * Ny);
			}
		}
	}
	#endif

	// --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	status = H5Dclose(dset);
	status = H5Fclose(file_info->input_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->input_file_name, snap_indx);
		exit(1);		
	}
}
/**
 * Function to create and open the output file
 */
void OpenOutputFile(void) {

	// Initialize variables
	herr_t status;
	struct stat st = {0};	// this is used to check whether the output directories exist or not.

	// --------------------------------
	//  Generate Output File Path
	// --------------------------------
	if (strcmp(file_info->output_dir, "NONE") == -1) {
		// Construct pathh
		strcpy(file_info->output_file_name, file_info->output_dir);
		strcat(file_info->output_file_name, "PostProcessing_HDF_Data.h5");

		// Print output file path to screen
		printf("Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);	
	}
	else if ((strcmp(file_info->output_dir, "NONE") == 0) && (stat(file_info->input_dir, &st) == 0)) {
		printf("\n["YELLOW"NOTE"RESET"] --- No Output directory provided. Using input directory instead \n");

		// Construct pathh
		strcpy(file_info->output_file_name, file_info->input_dir);
		strcat(file_info->output_file_name, "PostProcessing_HDF_Data.h5");

		// Print output file path to screen
		printf("Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);	
	}
	else if ((stat(file_info->input_dir, &st) == -1) && (stat(file_info->output_dir, &st) == -1)) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Output folder not provided or doesn't exist. Please provide output folder - see utils.c: \n-->>Exiting....\n");
		exit(1);
	}

	
	// --------------------------------
	//  Create Output File
	// --------------------------------
	file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file_info->output_file_handle < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->output_file_name);
		exit(1);
	}	

	// --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_file_name, "initial");
		exit(1);		
	}
}
/**
 * Function to write the data for the current snapshot to the file
 * @param t     The current time in the simulaiton 
 * @param snap The current snapshot
 */
void WriteDataToFile(double t, long int snap) {

	// Initialize variables
	char group_name[128];
	herr_t status;
	hid_t group_id;
	static const hsize_t Dims1D = 1;
	hsize_t dset_dims_1d[Dims1D];        // array to hold dims of the dataset to be created
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims_2d[Dims2D];        
	

	// -------------------------------
	// Check for Output File
	// -------------------------------
	// Check if output file exist if not create it
	if (access(file_info->output_file_name, F_OK) != 0) {
		file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%ld"RESET"]\n-->> Exiting...\n", file_info->output_file_name, snap);
			exit(1);
		}
	}
	else {
		// Open file with default I/O access properties
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%ld"RESET"]\n-->> Exiting...\n", file_info->output_file_name, snap);
			exit(1);
		}
	}

	// -------------------------------
	// Create Group for Current Snap
	// -------------------------------
	// Initialize Group Name
	sprintf(group_name, "/Snap_%05d", (int)snap);

	// Create group for the current snapshot
	group_id = CreateGroup(file_info->output_file_handle, file_info->output_file_name, group_name, t, snap);
	if (group_id < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", file_info->output_file_name, t, snap);
		exit(1);
	}

	// -------------------------------
	// Write Datasets
	// -------------------------------
	#ifdef __FULL_FIELD
	// The full field phases
	dset_dims_2d[0] = 2 * sys_vars->kmax - 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax - 1;
	H5LTmake_dataset(group_id, "FullFieldPhases", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->phases);	

	// The full field energy spectrum
	dset_dims_2d[0] = 2 * sys_vars->kmax - 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax - 1;
	H5LTmake_dataset(group_id, "FullFieldEnergySpectrum", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->enrg);		

	// The full field enstrophy spectrum
	dset_dims_2d[0] = 2 * sys_vars->kmax - 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax - 1;
	H5LTmake_dataset(group_id, "FullFieldEnstrophySpectrum", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->enst);		
	#endif
	#ifdef __REAL_STATS
	// The vorticity histogram bin ranges and bin counts
	dset_dims_1d[0] = stats_data->w_pdf->n + 1;
	H5LTmake_dataset(group_id, "VorticityPDFRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_pdf->range);		
	dset_dims_1d[0] = stats_data->w_pdf->n;
	H5LTmake_dataset(group_id, "VorticityPDFCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_pdf->bin);	

	// The velocity histogram bin ranges and bin counts
	dset_dims_1d[0] = stats_data->u_pdf->n + 1;
	H5LTmake_dataset(group_id, "VelocityPDFRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_pdf->range);		
	dset_dims_1d[0] = stats_data->u_pdf->n;
	H5LTmake_dataset(group_id, "VelocityPDFCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_pdf->bin);		
	#endif
	
	// -------------------------------
	// Close HDF5 Identifiers
	// -------------------------------
	status = H5Gclose(group_id);
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%ld"RESET"]\n-->> Exiting...\n", file_info->output_file_name, snap);
		exit(1);
	}
}
/**
 * Wrapper function used to create a Group for the current iteration in the output HDF5 file 
 * @param  group_name The name of the group - will be the snapshot counter
 * @param  t          The current time in the simulation
 * @param  snap      The current snapshot counter
 * @return            Returns a hid_t identifier for the created group
 */
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, long int snap) {

	// Initialize variables
	herr_t status;
	hid_t attr_id;
	hid_t group_id;
	hid_t attr_space;
	static const hsize_t attrank = 1;
	hsize_t attr_dims[attrank];

	// -------------------------------
	// Create the group
	// -------------------------------
	// Check if group exists
	if(H5Lexists(file_handle, group_name, H5P_DEFAULT)) {		
		// Open group if it already exists
		group_id = H5Gopen(file_handle, group_name, H5P_DEFAULT);
		if (group_id < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, snap);
			exit(1);
		}

	}
	else {
		// If not create new group and add time data as attribute to Group
		group_id = H5Gcreate(file_handle, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);	
		if (group_id < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, snap);
			exit(1);
		}


		// -------------------------------
		// Write Timedata as Attribute
		// -------------------------------
		// Create attribute datatspace
		attr_dims[0] = 1;
		attr_space   = H5Screate_simple(attrank, attr_dims, NULL); 	

		// Create attribute for current time in the integration
		attr_id = H5Acreate(group_id, "TimeValue", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
		if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &t)) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current time as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, snap);
			exit(1);
		}
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, snap);
			exit(1);
		}


		// -------------------------------
		// Close the attribute identifiers
		// -------------------------------
		status = H5Sclose(attr_space);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, snap);
			exit(1);
		}
	}

	// return group identifier
	return group_id;
}
/**
 * Function to create a HDF5 datatype for complex data
 */
hid_t CreateComplexDatatype(void) {

	// Declare HDF5 datatype variable
	hid_t dtype;

	// error handling var
	herr_t status;
	
	// Create complex struct
	struct complex_type_tmp cmplex;
	cmplex.re = 0.0;
	cmplex.im = 0.0;

	// create complex compound datatype
	dtype  = H5Tcreate(H5T_COMPOUND, sizeof(cmplex));

	// Insert the real part of the datatype
  	status = H5Tinsert(dtype, "r", offsetof(complex_type_tmp,re), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert real part for the Complex Compound Datatype!!\nExiting...\n");
  		exit(1);
  	}

  	// Insert the imaginary part of the datatype
  	status = H5Tinsert(dtype, "i", offsetof(complex_type_tmp,im), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert imaginary part for the Complex Compound Datatype! \n-->>Exiting...\n");
  		exit(1);
  	}

  	return dtype;
}
/**
 * Function to close output and input files
 */
void CloseFiles(void) {

	// --------------------------------
	//  Close Input File
	// --------------------------------
	H5Fclose(file_info->input_file_handle);

}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------