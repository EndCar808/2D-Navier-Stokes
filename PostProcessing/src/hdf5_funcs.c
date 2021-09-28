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
void OpenInputAndInitialize(void) {

	// Initialize variables
	hid_t dset;
	hid_t dspace;
	herr_t status;
	hsize_t Dims[SYS_DIM];
	int snaps = 0;
	char group_string[64];

	// --------------------------------
	//  Create Complex Datatype
	// --------------------------------
	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();


	// --------------------------------
	//  Get Input File Path
	// --------------------------------
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
	for(int i = 0; i < 1e6; ++i) {
		sprintf(group_string, "/Iter_%05d", i);	
		if(H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
			snaps++;
		}
	}

	// Save the total number of snaps
	sys_vars->num_snaps = snaps;

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
		if(H5LTread_dataset(file_info->input_file_handle, "kx", H5T_NATIVE_INT, run_data->x[0]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "kx");
			exit(1);	
		}
		if(H5LTread_dataset(file_info->input_file_handle, "ky", H5T_NATIVE_INT, run_data->x[1]) < 0) {
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
	status = H5Dclose(dset);
	status = H5Sclose(dspace);
	status = H5Fclose(file_info->input_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->input_file_name, 0);
		exit(1);		
	}
}
/**
 * Function to read in the solver data for the current snapshot
 * @param snap_indx The index of the currrent snapshot
 */
void ReadInData(int snap_indx) {

	// Initialize variables
	char group_string[64];

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
	//  Read in Vorticity
	// --------------------------------
	sprintf(group_string, "/Iter_%05d/w_hat", snap_indx);	
	if(H5LTread_dataset(file_info->input_file_handle, group_string, file_info->COMPLEX_DTYPE, run_data->w_hat) < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "w_hat", snap_indx);
		exit(1);	
	}

	// --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	H5Fclose(file_info->input_file_handle);
}
/**
 * Function to create and open the output file
 */
void OpenOutputFile(void) {

	// Initialize variables
	herr_t status;

	// --------------------------------
	//  Generate Output File Path
	// --------------------------------
	// Construct pathh
	strcpy(file_info->output_file_name, file_info->output_dir);
	strcat(file_info->output_file_name, "PostProcessing_HDF_Data.h5");

	// Print output file path to screen
	printf("Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);

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
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, 0);
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
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims[Dims2D];        // array to hold dims of the dataset to be created
	

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


	// -------------------------------
	// Write Datasets
	// -------------------------------
	// The full field phases
	dset_dims[0] = 2.0 * sys_vars->kmax - 1;
	dset_dims[1] = 2.0 * sys_vars->kmax - 1;
	H5LTmake_dataset(group_id, "FullFieldPhases", Dims2D, dset_dims, H5T_NATIVE_DOUBLE, proc_data->phases);	

	// The full field energy spectrum
	dset_dims[0] = 2.0 * sys_vars->kmax - 1;
	dset_dims[1] = 2.0 * sys_vars->kmax - 1;
	H5LTmake_dataset(group_id, "FullFieldEnergySpectrum", Dims2D, dset_dims, H5T_NATIVE_DOUBLE, proc_data->enrg);		

	// The full field enstrophy spectrum
	dset_dims[0] = 2.0 * sys_vars->kmax - 1;
	dset_dims[1] = 2.0 * sys_vars->kmax - 1;
	H5LTmake_dataset(group_id, "FullFieldEnstrophySpectrum", Dims2D, dset_dims, H5T_NATIVE_DOUBLE, proc_data->enst);		


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
	if (H5Lexists(file_handle, group_name, H5P_DEFAULT)) {		
		// Open group if it already exists
		group_id = H5Gopen(file_handle, group_name, H5P_DEFAULT);
	}
	else {
		// If not create new group and add time data as attribute to Group
		group_id = H5Gcreate(file_handle, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);	

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
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, snap);
			exit(1);
		}
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