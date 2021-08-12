/**
* @file hdf5_funcs.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing HDF5 function wrappers for creating, opening,wrtining to and closing output file
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
 * Wrapper function that creates the ouput directory creates and opens the output file
 */
void CreateOutputFileWriteICs(const long int* N, double dt) {

	// Initialize variabeles
	hid_t group_id;
	char group_name[128];
	herr_t status;
	hid_t plist_id;
	
	///////////////////////////
	/// Create & Open File
	/// ///////////////////////
	// -------------------------------
	// Create Parallel File PList
	// -------------------------------
	// Create proptery list for file access and set to parallel I/O
	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	status   = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
	if (status < 0) {
		printf("\n[ERROR] --- Could not set parallel I/O access for HDF5 output file! \n-->>Exiting....\n");
		exit(1);
	}

	// -----------------------------------
	// Create Output Directory and Path
	// -----------------------------------
	char file_data[512];
	sprintf(file_data, "_N[%ld,%ld]_ITERS[%ld].h5", sys_vars->N[0], sys_vars->N[1], sys_vars->num_t_steps);
	strcpy(file_info->output_file_name,"../Data/Test/Test"); // /work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/2D_NavierStokes
	strcat(file_info->output_file_name, file_data);
	if ( !(sys_vars->rank) ) {
		printf("\nOutput File: %s\n\n", file_info->output_file_name);
	}

	// ---------------------------------
	// Create the output file
	// ---------------------------------
	file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	if (file_info->output_file_handle < 0) {
		fprintf(stderr, "\n[ERROR] --- Could not create HDF5 output file at: %s \n-->>Exiting....\n", file_info->output_file_name);
		exit(1);
	}


	////////////////////////////////
	/// Write Initial Condtions
	/// ////////////////////////////
	// --------------------------------------
	// Create Group for Initial Conditions
	// --------------------------------------
	// Initialize Group Name
	sprintf(group_name, "/Iter_%05d", 0);
	
	// Create group for the current iteration data
	group_id = CreateGroup(group_name, 0.0, dt, 0);


	// --------------------------------------
	// Write Initial Conditions
	// --------------------------------------
	// Create dimension arrays
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims[Dims2D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims[Dims2D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims[Dims2D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding

	#ifdef __VORT_REAL
	// Transform vorticity back to real space
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->w_hat, run_data->w);

	// Specify dataset dimensions
	slab_dims[0]      = sys_vars->local_Nx;
	slab_dims[1]      = sys_vars->N[1];
	mem_space_dims[0] = sys_vars->local_Nx;
	mem_space_dims[1] = sys_vars->N[1] + 2;

	// Write the real space vorticity
	WriteDataReal(0.0, 0, group_id, "w", H5T_NATIVE_DOUBLE, (hsize_t* )sys_vars->N, slab_dims, mem_space_dims, sys_vars->local_Nx_start, run_data->w);
	#endif
	#ifdef __VORT_FOUR
	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();

	// Create dimension arrays
	dset_dims[0] 	  = sys_vars->N[0];
	dset_dims[1] 	  = sys_vars->N[1] / 2 + 1;
	slab_dims[0] 	  = sys_vars->local_Nx;
	slab_dims[1] 	  = sys_vars->N[1] / 2 + 1;
	mem_space_dims[0] = sys_vars->local_Nx;
	mem_space_dims[1] = sys_vars->N[1] / 2 + 1;

	// Write the real space vorticity
	WriteDataFourier(0.0, 0, group_id, "w_hat", file_info->COMPLEX_DTYPE, dset_dims, slab_dims, mem_space_dims, sys_vars->local_Nx_start, run_data->w_hat);
	#endif
	


	// ------------------------------------
	// Close Identifiers - also close file
	// ------------------------------------
	status = H5Pclose(plist_id);
	status = H5Gclose(group_id);
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n[ERROR] --- Unable to close output file [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", file_info->output_file_name, 0, 0.0);
		exit(1);		
	}
}
/**
 * Wrapper function that writes the data to file by openining it, creating a group for the current iteration and writing the data under this group. The file is then closed again 
 * @param t     The current time of the simulation
 * @param dt    The current timestep being used
 * @param iters The current iteration
 */
void WriteDataToFile(double t, double dt, long int iters) {

	// Initialize Variables
	char group_name[128];
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	herr_t status;
	hid_t group_id;
	hid_t plist_id;
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims[Dims2D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims[Dims2D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims[Dims2D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding

	// --------------------------------------
	// Check if file exists and Open/Create
	// --------------------------------------
	// Create property list for setting parallel I/O access properties for file
	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	// Check if file exists - open it if it does if not create it
	if (access(file_info->output_file_name, F_OK) != 0) {
		file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n[ERROR] --- Unable to create output file [%s] at: Iter = [%ld] t = [%lf]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
			exit(1);
		}
	}
	else {
		// Open file with parallel I/O access properties
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR , plist_id);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n[ERROR] --- Unable to open output file [%s] at: Iter = [%ld] t = [%lf]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
			exit(1);
		}
	}
	H5Pclose(plist_id);


	// -------------------------------
	// Create Group 
	// -------------------------------
	// Initialize Group Name
	sprintf(group_name, "/Iter_%05d", (int)iters);
	
	// Create group for the current iteration data
	group_id = CreateGroup(group_name, t, dt, iters);

	
	// -------------------------------
	// Write Data to File
	// -------------------------------
	#ifdef __VORT_REAL
	// Transform Fourier space vorticiy to real space
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->w_hat, run_data->w);

	// Create dimension arrays
	slab_dims[0]      = sys_vars->local_Nx;
	slab_dims[1]      = Ny;
	mem_space_dims[0] = sys_vars->local_Nx;
	mem_space_dims[1] = Ny + 2;

	// Write the real space vorticity
	WriteDataReal(t, (int)iters, group_id, "w", H5T_NATIVE_DOUBLE, (hsize_t* )sys_vars->N, slab_dims, mem_space_dims, sys_vars->local_Nx_start, run_data->w);
	#endif
	#ifdef __VORT_FOUR
	// Create dimension arrays
	dset_dims[0] = sys_vars->N[0];
	dset_dims[1] = sys_vars->N[1] / 2 + 1;
	slab_dims[0] = sys_vars->local_Nx;
	slab_dims[1] = Ny_Fourier;

	// Write the real space vorticity
	WriteDataFourier(t, (int)iters, group_id, "w_hat", file_info->COMPLEX_DTYPE, dset_dims, slab_dims, slab_dims, sys_vars->local_Nx_start, run_data->w_hat);
	#endif

	// -------------------------------
	// Close identifiers and File
	// -------------------------------
	status = H5Gclose(group_id);
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n[ERROR] --- Unable to close output file [%s] at: Iter = [%ld] t = [%lf]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
		exit(1);
	}
}
/**
 * Wrapper function used to create a Group for the current iteration in the HDF5 file 
 * @param  group_name The name of the group - will be the Iteration counter
 * @param  t          The current time in the simulation
 * @param  dt         The current timestep being used
 * @param  iters      The current iteration counter
 * @return            Returns a hid_t identifier for the created group
 */
hid_t CreateGroup(char* group_name, double t, double dt, long int iters) {

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
	if (H5Lexists(file_info->output_file_handle, group_name, H5P_DEFAULT)) {		
		// Open group if it already exists
		group_id = H5Gopen(file_info->output_file_handle, group_name, H5P_DEFAULT);
	}
	else {
		// If not create new group and add time data as attribute to Group
		group_id = H5Gcreate(file_info->output_file_handle, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);	

		// -------------------------------
		// Write Timedata as Attribute
		// -------------------------------
		// Create attribute datatspace
		attr_dims[0] = 1;
		attr_space   = H5Screate_simple(attrank, attr_dims, NULL); 	

		// Create attribute for current time in the integration
		attr_id = H5Acreate(group_id, "TimeValue", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
		if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &t)) < 0) {
			fprintf(stderr, "\n[ERROR] --- Could not write current time as attribute to group in file at: t = [%lf] Iter = [%ld]!!\n-->> Exiting...\n", t, iters);
			exit(1);
		}
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n[ERROR] --- Unable to close attribute Idenfiers: t = [%lf] Iter = [%ld]!!\n-->> Exiting...\n", t, iters);
			exit(1);
		}
		// Create attribute for the current timestep
		attr_id = H5Acreate(group_id, "TimeStep", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
		if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &dt)) < 0) {
			fprintf(stderr, "\n[ERROR] --- Could not write current timestep as attribute to group in file at: t = [%lf] Iter = [%ld]!!\n-->> Exiting...\n", t, iters);
			exit(1);
		}


		// -------------------------------
		// Close the attribute identifiers
		// -------------------------------
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n[ERROR] --- Unable to close attribute Idenfiers: t = [%lf] Iter = [%ld]!!\n-->> Exiting...\n", t, iters);
			exit(1);
		}
		status = H5Sclose(attr_space);
		if (status < 0 ) {
			fprintf(stderr, "\n[ERROR] --- Unable to close attribute Idenfiers: t = [%lf] Iter = [%ld]!!\n-->> Exiting...\n", t, iters);
			exit(1);
		}
	}

	return group_id;
}
/**
 * Function that creates a dataset in a created Group in the output file and writes the data to this dataset for Fourier Space arrays
 * @param group_id       The identifier of the Group for the current iteration to write the data to
 * @param dset_name      The name of the dataset to write
 * @param dtype          The datatype of the data being written
 * @param dset_dims      Array containg the dimensions of the dataset to create
 * @param slab_dims      Array containing the dimensions of the hyperslab to select
 * @param mem_space_dims Array containing the dimensions of the memory space that will be written to file
 * @param offset_Nx      The offset in the dataset that each process will write to
 * @param data           The data being written to file
 */
void WriteDataFourier(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, fftw_complex* data) {

	// Initialize variables
	hid_t plist_id;
	hid_t dset_space;
	hid_t file_space;
	hid_t mem_space;
	static const hsize_t Dims2D = 2;
	hsize_t dims2d[Dims2D];        // array to hold dims of the dataset to be created
	hsize_t mem_dims[Dims2D];	   // Array to hold the dimensions of the memory space - this will be diferent to slab dims for real data due to zero
	hsize_t mem_offset[Dims2D];    // Array to hold the offset in eahc direction for the local hypslabs to write from
	hsize_t slabsize[Dims2D];      // Array holding the size of the hyperslab in each direction
	hsize_t dset_offset[Dims2D];   // Array containig the offset positions in the file for each process to write to
	hsize_t dset_slabsize[Dims2D]; // Array containing the size of the slabbed that is being written to in file	

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	dims2d[0] = dset_dims[0];
	dims2d[1] = dset_dims[1];
	dset_space = H5Screate_simple(Dims2D, dims2d, NULL); 
	if (dset_space < 0) {
		fprintf(stderr, "\n[ERROR] --- Unable to set dataspace for dataset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, dtype, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// -------------------------------
	// Create Appropriate Hyperslabs
	// -------------------------------
	// Setup for hyperslab selection
	slabsize[0]      = slab_dims[0];
	slabsize[1]      = slab_dims[1];
	mem_offset[0]    = 0;
	mem_offset[1]    = 0;
	dset_offset[0]   = offset_Nx;
	dset_offset[1]   = 0;
	dset_slabsize[0] = slab_dims[0];
	dset_slabsize[1] = slab_dims[1];
	
	// Create the memory space for the hyperslabs for each process - reset second dimension for hyperslab selection to ignore padding
	mem_dims[0] = mem_space_dims[0];
	mem_dims[1] = mem_space_dims[1];
	mem_space = H5Screate_simple(Dims2D, mem_dims, NULL);

	// Select local hyperslab from the memoryspace (slab size adjusted to ignore 0 padding) - local to each process
	if ((H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_offset, NULL, slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n[ERROR] --- unable to select local hyperslab for datset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// --------------------------------------
	// Set Up Collective Write & Write Data
	// --------------------------------------
	// Set up Collective write property
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	// Select the hyperslab in the dataset on file to write to
	if ((H5Sselect_hyperslab(dset_space, H5S_SELECT_SET, dset_offset, NULL, dset_slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n[ERROR] --- Unable to select hyperslab in file for datset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// Write data to file
	if ((H5Dwrite(file_space, dtype, mem_space, dset_space, plist_id, data)) < 0 ) {
		fprintf(stderr, "\n[ERROR] --- Unable to write data to datset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Close identifiers
	// -------------------------------
	H5Pclose(plist_id);
	H5Dclose(file_space);
	H5Sclose(dset_space);
	H5Sclose(mem_space);
}
/**
 * Function that creates a dataset in a created Group in the output file and writes the data to this dataset for Real Space arrays
 * @param group_id       The identifier of the Group for the current iteration to write the data to
 * @param dset_name      The name of the dataset to write
 * @param dtype          The datatype of the data being written
 * @param dset_dims      Array containg the dimensions of the dataset to create
 * @param slab_dims      Array containing the dimensions of the hyperslab to select
 * @param mem_space_dims Array containing the dimensions of the memory space that will be written to file
 * @param offset_Nx      The offset in the dataset that each process will write to
 * @param data           The data being written to file
 */
void WriteDataReal(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, double* data) {

	// Initialize variables
	hid_t plist_id;
	hid_t dset_space;
	hid_t file_space;
	hid_t mem_space;
	static const hsize_t Dims2D = 2;
	hsize_t dims2d[Dims2D];        // array to hold dims of the dataset to be created
	hsize_t mem_dims[Dims2D];	   // Array to hold the dimensions of the memory space - this will be diferent to slab dims for real data due to zero
	hsize_t mem_offset[Dims2D];    // Array to hold the offset in eahc direction for the local hypslabs to write from
	hsize_t slabsize[Dims2D];      // Array holding the size of the hyperslab in each direction
	hsize_t dset_offset[Dims2D];   // Array containig the offset positions in the file for each process to write to
	hsize_t dset_slabsize[Dims2D]; // Array containing the size of the slabbed that is being written to in file	

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	dims2d[0] = dset_dims[0];
	dims2d[1] = dset_dims[1];
	dset_space = H5Screate_simple(Dims2D, dims2d, NULL); 
	if (dset_space < 0) {
		fprintf(stderr, "\n[ERROR] --- Unable to set dataspace for dataset: [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, H5T_NATIVE_DOUBLE, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// -------------------------------
	// Create Appropriate Hyperslabs
	// -------------------------------
	// Setup for hyperslab selection
	slabsize[0]      = slab_dims[0];
	slabsize[1]      = slab_dims[1];
	mem_offset[0]    = 0;
	mem_offset[1]    = 0;
	dset_offset[0]   = offset_Nx;
	dset_offset[1]   = 0;
	dset_slabsize[0] = slab_dims[0];
	dset_slabsize[1] = slab_dims[1];
	
	// Create the memory space for the hyperslabs for each process - reset second dimension for hyperslab selection to ignore padding
	mem_dims[0] = mem_space_dims[0];
	mem_dims[1] = mem_space_dims[1];
	mem_space = H5Screate_simple(Dims2D, mem_dims, NULL);

	// Select local hyperslab from the memoryspace (slab size adjusted to ignore 0 padding) - local to each process
	if ((H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_offset, NULL, slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n[ERROR] --- unable to select local hyperslab for datset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// --------------------------------------
	// Set Up Collective Write & Write Data
	// --------------------------------------
	// Set up Collective write property
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	// Select the hyperslab in the dataset on file to write to
	if ((H5Sselect_hyperslab(dset_space, H5S_SELECT_SET, dset_offset, NULL, dset_slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n[ERROR] --- Unable to select hyperslab in file for datset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// Write data to file
	if ((H5Dwrite(file_space, dtype, mem_space, dset_space, plist_id, data)) < 0 ) {
		fprintf(stderr, "\n[ERROR] --- Unable to write data to datset [%s] at: Iter = [%d] t = [%lf]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Close identifiers
	// -------------------------------
	H5Pclose(plist_id);
	H5Dclose(file_space);
	H5Sclose(dset_space);
	H5Sclose(mem_space);
}
/**
 * Wrapper function that writes all the non-slabbed/chunk datasets to file after integeration has finished - to do so the file must be reponed 
 * with the right read/write permissions and normal I/0 access properties -> otherwise writing to file in a non MPI way would not work
 * @param N Array containing the dimensions of the system
 */
void FinalWriteAndCloseOutputFile(const long int* N) {

	// Initialize Variables
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	herr_t status;
	const hsize_t D1 = 1;
	hsize_t dims1D[D1];


	////////////////////////////////
	/// Repon and Write Datasets
	////////////////////////////////
	// Repon Output file with read/write permissions
	if (!(sys_vars->rank)) {
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR , H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n[ERROR] --- Unable to reopen output file for writing non chunked/slabbed datasets! \n-->>Exiting....\n");
			exit(1);
		}
	}

	// -------------------------------
	// Write Wavenumbers
	// -------------------------------
	#ifdef __WAVELIST
	// Allocate array to gather the wavenumbers from each of the local arrays - in the x direction
	int* k0 = (int* )fftw_malloc(sizeof(int) * Nx);
	MPI_Gather(run_data->k[0], sys_vars->local_Nx, MPI_INT, k0, sys_vars->local_Nx, MPI_INT, 0, MPI_COMM_WORLD); 

	// Write to file
	if (!(sys_vars->rank)) {
		dims1D[0] = Nx;
		if ( (H5LTmake_dataset(file_info->output_file_handle, "kx", D1, dims1D, H5T_NATIVE_INT, k0)) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "kx");
		}
		dims1D[0] = Ny_Fourier;
		if ( (H5LTmake_dataset(file_info->output_file_handle, "ky", D1, dims1D, H5T_NATIVE_INT, run_data->k[1])) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "ky");
		}
	}
	fftw_free(k0);
	#endif

	// -------------------------------
	// Write Collocation Points
	// -------------------------------
	#ifdef __COLLOC_PTS
	// Allocate array to gather the collocation points from each of the local arrays
	double* x0 = (double* )fftw_malloc(sizeof(double) * Nx);
	MPI_Gather(run_data->x[0], sys_vars->local_Nx, MPI_DOUBLE, x0, sys_vars->local_Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

	// Write to file
	if (!(sys_vars->rank)) {
		dims1D[0] = Nx;
		if ( (H5LTmake_dataset(file_info->output_file_handle, "x", D1, dims1D, H5T_NATIVE_DOUBLE, x0)) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "x");
		}
		dims1D[0] = Ny;
		if ( (H5LTmake_dataset(file_info->output_file_handle, "y", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->x[1]))< 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "y");
		}
	}
	fftw_free(x0);
	#endif

	// -------------------------------
	// Write System Measures
	// -------------------------------
	// Time
	#ifdef __TIME
	// Time array only on rank 0
	if (!(sys_vars->rank)) {
		dims1D[0] = sys_vars->num_print_steps;
		if ( (H5LTmake_dataset(file_info->output_file_handle, "Time", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->time)) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "Time");
		}
	}
	#endif

	
	// Total Energy, Enstrophy and Palinstrophy -> need to reduce (in place on rank 0) all arrays across the processess
	if (!(sys_vars->rank)) {
		// Reduce on to rank 0
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_energy, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_enstr, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_palin, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// Dataset dims
		dims1D[0] = sys_vars->num_print_steps;

		// Energy
		if ( (H5LTmake_dataset(file_info->output_file_handle, "TotalEnergy", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_energy)) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "TotalEnergy");
		}
		// Enstrophy
		if ( (H5LTmake_dataset(file_info->output_file_handle, "TotalEnstrophy", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_enstr)) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "TotalEnstrophy");
		}
		// Palinstrophy
		if ( (H5LTmake_dataset(file_info->output_file_handle, "TotalPalinstrophy", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_palin)) < 0) {
			printf("\n[WARNING] --- Failed to make dataset [%s]\n", "TotalPalinstrophy");
		}
	}
	else {
		// Reduce all other process to rank 0
		MPI_Reduce(run_data->tot_energy, NULL,  sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_enstr, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_palin, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	// -------------------------------
	// Close File for the final time
	// -------------------------------
	if (!(sys_vars->rank)) {
		status = H5Fclose(file_info->output_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n[ERROR] --- Unable to close output file! \n-->>Exiting....\n");
			exit(1);
		}
	}
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
  		fprintf(stderr, "\n[ERROR] --- Could not insert real part for the Complex Compound Datatype!!\nExiting...\n");
  		exit(1);
  	}

  	// Insert the imaginary part of the datatype
  	status = H5Tinsert(dtype, "i", offsetof(complex_type_tmp,im), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n[ERROR] --- Could not insert imaginary part for the Complex Compound Datatype! \n-->>Exiting...\n");
  		exit(1);
  	}

  	return dtype;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------