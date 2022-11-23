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
#include "stats.h"


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
			strcpy(file_info->sys_msr_file_name, file_info->input_dir);
			strcat(file_info->sys_msr_file_name, "SystemMeasures_HDF_Data.h5");			
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
	if (access(file_info->sys_msr_file_name, F_OK) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- System Measures Input file ["CYAN"%s"RESET"] does not exist\n-->> Exiting...\n", file_info->sys_msr_file_name);
		exit(1);
	}
	else {
		// Open file
		file_info->sys_msr_file_handle = H5Fopen(file_info->sys_msr_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->sys_msr_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open system measures file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->sys_msr_file_name);
			exit(1);
		}
	}

	// --------------------------------
	//  Get Number of Snaps
	// --------------------------------
	// Count the number of snapshot groups in file
	printf("Checking Number of Snapshots ");
	for(int i = 0; i < (int) 1e7; ++i) {
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
	if((H5Lexists(file_info->sys_msr_file_handle, "x", H5P_DEFAULT) > 0) && (H5Lexists(file_info->sys_msr_file_handle, "y", H5P_DEFAULT) > 0) && (H5Lexists(file_info->sys_msr_file_handle, "kx", H5P_DEFAULT) > 0) && (H5Lexists(file_info->sys_msr_file_handle, "ky", H5P_DEFAULT) > 0)) {
		// Read in real space arrays
		if(H5LTread_dataset(file_info->sys_msr_file_handle, "x", H5T_NATIVE_DOUBLE, run_data->x[0]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "x");
			exit(1);	
		}
		if(H5LTread_dataset(file_info->sys_msr_file_handle, "y", H5T_NATIVE_DOUBLE, run_data->x[1]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "y");
			exit(1);
		}

		// Read in Fourier space arrays
		if(H5LTread_dataset(file_info->sys_msr_file_handle, "ky", H5T_NATIVE_INT, run_data->k[0]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ky");
			exit(1);
		}
		if(H5LTread_dataset(file_info->sys_msr_file_handle, "kx", H5T_NATIVE_INT, run_data->k[1]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "kx");
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
	status = H5Fclose(file_info->sys_msr_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close  system measures file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->sys_msr_file_name, "initial");
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
	const long int Ny 		  = sys_vars->N[0];
	const long int Nx 		  = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	char group_string[64];
	char phase_group_string[64];
	char amp_group_string[64];
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
	sprintf(phase_group_string, "/Iter_%05d/phi_k", snap_indx);	
	sprintf(amp_group_string, "/Iter_%05d/a_k", snap_indx);	
	if (H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
		// Read in the Fourier space vorticity
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
	}
	else if ((H5Lexists(file_info->input_file_handle, phase_group_string, H5P_DEFAULT) > 0) && (H5Lexists(file_info->input_file_handle, amp_group_string, H5P_DEFAULT) > 0)){
		// Read in the Fourier phases
		dset = H5Dopen (file_info->input_file_handle, phase_group_string, H5P_DEFAULT);
		if (dset < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "phi_k", snap_indx);
			exit(1);
		} 
		if(H5LTread_dataset(file_info->input_file_handle, phase_group_string, H5T_NATIVE_DOUBLE, run_data->phi_k) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "phi_k", snap_indx);
			exit(1);	
		}

		// Read in the Fourier amplitudes
		dset = H5Dopen (file_info->input_file_handle, amp_group_string, H5P_DEFAULT);
		if (dset < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "a_k", snap_indx);
			exit(1);		
		} 
		if(H5LTread_dataset(file_info->input_file_handle, amp_group_string, H5T_NATIVE_DOUBLE, run_data->a_k) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "a_k", snap_indx);
			exit(1);	
		}

		// Create the Fourier space vorticity from the amplitudes and phases
		for (int i = 0; i < Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;
				run_data->w_hat[indx] = run_data->a_k[indx] * cexp(I * run_data->phi_k[indx]);
			}
		}
	}
	else {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to find ["CYAN"%s"RESET"] or ["CYAN"%s"RESET"] and ["CYAN"%s"RESET"] in file ["CYAN"%s"RESET"]. Please check input file\n-->> Exiting...\n", "w_hat", "a_k", "phi_k", file_info->input_file_name);
		exit(1);
	}

	// --------------------------------
	//  Read in Real Vorticity
	// --------------------------------
	#if defined(__REAL_STATS) || defined(__VORT_INC_STATS) || defined(__VORT_STR_FUNC_STATS) || defined(__VORT_RADIAL_STR_FUNC_STATS) || defined(__VORT_GRAD_STATS)
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

		// Get the real space vorticity from the Fourier space
		for (int i = 0; i < Ny; ++i) {	
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Save the fourier vorticity before transform as it will be overwrittend
				run_data->tmp_w_hat[indx] = run_data->w_hat[indx];
			}
		}
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->tmp_w_hat, run_data->w);
		for (int i = 0; i < Ny; ++i) {	
			tmp = i * Nx;
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize the vorticity
				run_data->w[indx] *= 1.0; // /(Ny * Nx);
			}
		}
		
		// Real space vorticity does not exists
		sys_vars->REAL_VORT_FLAG = 1; 
	}
	#endif

	// --------------------------------
	//  Read in Real Velocity
	// --------------------------------
	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__VEL_STR_FUNC_STATS) || defined(__VEL_GRAD_STATS)
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

		// Real space velocity exists
		sys_vars->REAL_VEL_FLAG = 1; 

		#if defined(__VEL_GRAD_STATS)
		// Transform from Real space To Fourier Space
		for (int i = 0; i < Ny; ++i) {	
			tmp = i * Nx;
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize the velocity
				run_data->tmp_u[SYS_DIM * indx + 0] = run_data->u[SYS_DIM * indx + 0];
				run_data->tmp_u[SYS_DIM * indx + 1] = run_data->u[SYS_DIM * indx + 1];
			}
		}
		fftw_execute_dft_r2c(sys_vars->fftw_2d_dft_batch_r2c, run_data->tmp_u, run_data->u_hat);

		// Real space velocity exists
		sys_vars->FOUR_VEL_FLAG = 1; 
		#endif
	}
	else {
		fftw_complex k_sqr;
		
		// Compute the Fourier velocity from the Fourier vorticity
		for (int i = 0; i < Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
					// Compute the prefactor
					k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
					
					// Compute the Fourier velocity
					run_data->u_hat[SYS_DIM * indx + 0] = k_sqr * (double)run_data->k[0][i] * run_data->w_hat[indx];
					run_data->u_hat[SYS_DIM * indx + 1] = -k_sqr * (double)run_data->k[1][j] * run_data->w_hat[indx];
				}
				else {
					run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
				}

				// Save the Fourier velocities in temp array before transform
				run_data->tmp_u_hat[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0];
				run_data->tmp_u_hat[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1];	
			}
		}

		// Real space velocity exists
		sys_vars->FOUR_VEL_FLAG = 1; 

		// Transform back to Real space and Normalize
		fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, run_data->tmp_u_hat, run_data->u);
		for (int i = 0; i < Ny; ++i) {	
			tmp = i * Nx;
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize the velocity
				run_data->u[SYS_DIM * indx + 0] *= 1.0; // /(Ny * Nx);
				run_data->u[SYS_DIM * indx + 1] *= 1.0; // /(Ny * Nx);
			}
		}

		// Real space velocity exists now
		sys_vars->REAL_VEL_FLAG = 1; 
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
	char file_name[1024];
	herr_t status;
	struct stat st = {0};	// this is used to check whether the output directories exist or not.

	// --------------------------------
	//  Generate Output File Path
	// --------------------------------
	if (strcmp(file_info->output_dir, "NONE") != 0) {
		// Construct pathh
		strcpy(file_info->output_file_name, file_info->output_dir);
		sprintf(file_name, "PostProcessing_HDF_Data_SECTORS[%d,%d]_KFRAC[%1.2lf]_TAG[%s].h5", sys_vars->num_k3_sectors, sys_vars->num_k1_sectors, sys_vars->kmax_frac, file_info->output_tag);
		strcat(file_info->output_file_name, file_name);

		// Print output file path to screen
		printf("Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);	
	}
	else if ((strcmp(file_info->output_dir, "NONE") == 0) && (stat(file_info->input_dir, &st) == 0)) {
		printf("\n["YELLOW"NOTE"RESET"] --- No Output directory provided. Using input directory instead \n");

		// Construct pathh
		strcpy(file_info->output_file_name, file_info->input_dir);
		sprintf(file_name, "PostProcessing_HDF_Data_SECTORS[%d,%d]_KFRAC[%1.2lf]_TAG[%s].h5", sys_vars->num_k3_sectors, sys_vars->num_k1_sectors, sys_vars->kmax_frac, file_info->output_tag);
		strcat(file_info->output_file_name, file_name);

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
		fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 output file at: "CYAN"%s"RESET" \n-->> Exiting....\n", file_info->output_file_name);
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
	int indx, temp;
	const long int Ny 		  = sys_vars->N[0];
	const long int Nx 		  = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	static const hsize_t Dims1D = 1;
	hsize_t dset_dims_1d[Dims1D];       
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims_2d[Dims2D];   
	static const hsize_t Dims3D = 3;
	hsize_t dset_dims_3d[Dims3D];     
	
	// printf("file_name: %s", file_info->output_file_name);
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
	// Write Vorticity
	// -------------------------------
	#if defined(__VORT_FOUR)
	// The full field phases
	dset_dims_2d[0] = sys_vars->N[0];
	dset_dims_2d[1] = sys_vars->N[1] / 2 + 1;
	status = H5LTmake_dataset(group_id, "w_hat", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, run_data->w_hat);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "w_hat", t, snap);
		exit(1);
	}
	#endif

	#if defined(__MODES) || defined(__REALSPACE)
	// Get the Fourier space velocities
	fftw_complex k_sqr_inv;
	for (int i = 0; i < Ny; ++i) {
		temp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = temp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Get I / |k|^2
				k_sqr_inv = I / (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Fill the fourier space velocities
				run_data->u_hat[SYS_DIM * indx + 0] = k_sqr_inv * run_data->k[0][i] * run_data->w_hat[indx];
				run_data->u_hat[SYS_DIM * indx + 1] = -k_sqr_inv * run_data->k[1][j] * run_data->w_hat[indx];
			}
			else {
				run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
			}
		}
	}
	#endif


	// -------------------------------
	// Write Real Space Vorticity
	// -------------------------------
	#if defined(__VORT_REAL)
	// Get the real space vorticity from the Fourier space
	for (int i = 0; i < Ny; ++i) {	
		temp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = temp + j;

			// Save the fourier vorticity before transform as it will be overwritten
			run_data->tmp_w_hat[indx] = run_data->w_hat[indx];
		}
	}
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->tmp_w_hat, run_data->w);
	for (int i = 0; i < Ny; ++i) {	
		temp = i * Nx;
		for (int j = 0; j < Nx; ++j) {
			indx = temp + j;

			// Normalize the vorticity
			run_data->w[indx] *= 1.0; // /(Ny * Nx);
		}
	}

	// Write the real space vorticity
	dset_dims_2d[0] = sys_vars->N[0];
	dset_dims_2d[1] = sys_vars->N[1];
	status = H5LTmake_dataset(group_id, "w", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, run_data->w);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "w", t, snap);
		exit(1);
	}
	#endif

	// -------------------------------
	// Write Fourier Space Velocity
	// -------------------------------
	#if defined(__MODES)
	// Write the Fourier modes
	dset_dims_3d[0] = sys_vars->N[0];
	dset_dims_3d[1] = sys_vars->N[1] / 2 + 1;
	dset_dims_3d[2] = SYS_DIM;
	status = H5LTmake_dataset(group_id, "u_hat", Dims3D, dset_dims_3d, file_info->COMPLEX_DTYPE, run_data->u_hat);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "u_hat", t, snap);
		exit(1);
	}
	#endif

	// -------------------------------
	// Write Real Space Velocity
	// -------------------------------
	#if defined(__REALSPACE)
	// Get the real space vorticity from the Fourier space
	for (int i = 0; i < Ny; ++i) {
		temp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = temp + j;

			// Save the Fourier velocities in temp array before transform
			run_data->tmp_u_hat[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0];
			run_data->tmp_u_hat[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1];	
		}
	}
	fftw_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, run_data->tmp_u_hat, run_data->u);
	for (int i = 0; i < Ny; ++i) {	
		temp = i * Nx;
		for (int j = 0; j < Nx; ++j) {
			indx = temp + j;

			// Normalize the velocity
			run_data->u[SYS_DIM * indx + 0] *= 1.0; // /(Ny * Nx);
			run_data->u[SYS_DIM * indx + 1] *= 1.0; // /(Ny * Nx);
		}
	}

	// Write the Fourier modes
	dset_dims_3d[0] = sys_vars->N[0];
	dset_dims_3d[1] = sys_vars->N[1];
	dset_dims_3d[2] = SYS_DIM;
	status = H5LTmake_dataset(group_id, "u", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, run_data->u);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "u", t, snap);
		exit(1);
	}
	#endif
	
	// -------------------------------
	// Write Full Field Data
	// -------------------------------
	#if defined(__FULL_FIELD)
	// The full field phases
	dset_dims_2d[0] = 2 * sys_vars->kmax + 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax + 1;
	status = H5LTmake_dataset(group_id, "FullFieldPhases", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->phases);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Full Field Phases", t, snap);
		exit(1);
	}

	// The full field amplitudes
	dset_dims_2d[0] = 2 * sys_vars->kmax + 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax + 1;
	status = H5LTmake_dataset(group_id, "FullFieldAmplitudes", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->amps);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Full Field Amplitudes", t, snap);
		exit(1);
	}

	// The full field energy spectrum
	dset_dims_2d[0] = 2 * sys_vars->kmax + 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax + 1;
	status = H5LTmake_dataset(group_id, "FullFieldEnergySpectrum", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->enrg);		
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Full Field Energy Spectrum", t, snap);
		exit(1);
	}

	// The full field enstrophy spectrum
	dset_dims_2d[0] = 2 * sys_vars->kmax + 1;
	dset_dims_2d[1] = 2 * sys_vars->kmax + 1;
	status = H5LTmake_dataset(group_id, "FullFieldEnstrophySpectrum", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, proc_data->enst);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Full Field Enstrophy Spectrum", t, snap);
		exit(1);
	}
	#endif


	// -------------------------------
	// Write Real Space Stats
	// -------------------------------
	#if defined(__REAL_STATS)
	// The vorticity histogram bin ranges and bin counts
	dset_dims_1d[0] = stats_data->w_hist->n + 1;
	status = H5LTmake_dataset(group_id, "VorticityPDFRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_hist->range);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Vorticity PDF Bin Ranges", t, snap);
		exit(1);
	}		
	dset_dims_1d[0] = stats_data->w_hist->n;
	status = H5LTmake_dataset(group_id, "VorticityPDFCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->w_hist->bin);	
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Vorticity PDF Bin Counts", t, snap);
		exit(1);
	}
	
	// The velocity histogram bin ranges and bin counts
	dset_dims_1d[0] = stats_data->u_hist->n + 1;
	status = H5LTmake_dataset(group_id, "VelocityPDFRanges", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_hist->range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Velocity PDF Bin Ranges", t, snap);
        exit(1);
    }		
	dset_dims_1d[0] = stats_data->u_hist->n;
	status = H5LTmake_dataset(group_id, "VelocityPDFCounts", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, stats_data->u_hist->bin);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Velocity PDF Bin Counts", t, snap);
        exit(1);
    }	
	#endif

    // -------------------------------
    // Write Spectra Data
    // -------------------------------
	#if defined(__SPECTRA)
	dset_dims_1d[0] = sys_vars->n_spec;
	status = H5LTmake_dataset(group_id, "EnstrophySpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "1D Enstrophy Spectrum", t, snap);
        exit(1);
    }
    
    status = H5LTmake_dataset(group_id, "EnergySpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enrg_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "1D Energy Spectrum", t, snap);
        exit(1);
    }
	#endif
	#if defined(__ENST_FLUX)
	// Write the enstrophy flux spectrum
	dset_dims_1d[0] = sys_vars->n_spec;
	status = H5LTmake_dataset(group_id, "EnstrophyFluxSpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_flux_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Flux Spectrum", t, snap);
        exit(1);
    }
    // Write the time derivative of enstrophy spectrum
    status = H5LTmake_dataset(group_id, "EnstrophyTimeDerivativeSpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->d_enst_dt_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Time Derivative of Enstrophy Spectrum", t, snap);
        exit(1);
    }
    // Write the enstrophy dissipation spectrum
    status = H5LTmake_dataset(group_id, "EnstrophyDissSpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_diss_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Dissipation Spectrum", t, snap);
        exit(1);
    }
	#endif
	#if defined(__ENRG_FLUX)
	// Write the energy flux spectrum
	dset_dims_1d[0] = sys_vars->n_spec;
	status = H5LTmake_dataset(group_id, "EnergyFluxSpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enrg_flux_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Energy Flux Spectrum", t, snap);
        exit(1);
    }
    // Write the time derivative of energy spectrum
    status = H5LTmake_dataset(group_id, "EnergyTimeDerivativeSpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->d_enrg_dt_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Time Derivative of Energy Spectrum", t, snap);
        exit(1);
    }
    // Write the energy dissipation spectrum
    status = H5LTmake_dataset(group_id, "EnergyDissSpectrum", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enrg_diss_spec);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Energy Dissipation Spectrum", t, snap);
        exit(1);
    }
	#endif
    #if (defined(__ENST_FLUX) || defined(__ENRG_FLUX) || defined(__SEC_PHASE_SYNC)) && defined(__NONLIN)
    // Write the nonlinear term in Fourier space
    dset_dims_2d[0] = sys_vars->N[0];
    dset_dims_2d[1] = sys_vars->N[1] / 2 + 1;
   //  for (int i = 0; i < Ny; ++i)
   //  {
   //  	for (int j = 0; j < Nx_Fourier; ++j) {
			// printf("NONLIN[%d,%d]: %5.16lf %5.16lf ", run_data->k[0][i], run_data->k[1][j], creal(proc_data->dw_hat_dt[(i * (Nx_Fourier) + j)]), cimag(proc_data->dw_hat_dt[(i * (Nx_Fourier) + j)]));
   //  	}
   //  }
	status = H5LTmake_dataset(group_id, "NonlinearTerm", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, proc_data->dw_hat_dt);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Nonlinear Term", t, snap);
        exit(1);
    }
    #endif


    // -------------------------------
    // Write Phase Sync Data
    // -------------------------------
	#if defined(__SEC_PHASE_SYNC)
	/// ----------------------- Phase Order Data
	// Write the phase order data for the individual phases
    dset_dims_1d[0] = sys_vars->num_k3_sectors;
    status = H5LTmake_dataset(group_id, "PhaseSync", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->phase_R);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Phase Sync Parameter", t, snap);
        exit(1);
    }
    status = H5LTmake_dataset(group_id, "AverageAngle", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->phase_Phi);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Average Angle", t, snap);
        exit(1);
    }    

    dset_dims_1d[0] = NUM_TRIAD_TYPES + 1;
    status = H5LTmake_dataset(group_id, "TriadPhaseSyncTest", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->triad_R_test);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Phase Sync Parameter", t, snap);
        exit(1);
    }
    status = H5LTmake_dataset(group_id, "TriadAverageAngleTest", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->triad_Phi_test);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Average Angle", t, snap);
        exit(1);
    }    

    //------------------------ Write the phase order data for the triad phases
    // Allocate tmporary memory for writing the phase order data
    double* tmp = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * (NUM_TRIAD_TYPES + 1));
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->triad_R[i][a];
    	}
    }
    dset_dims_2d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_2d[1] = sys_vars->num_k3_sectors;
	status = H5LTmake_dataset(group_id, "TriadPhaseSync", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Triad Sync Parameter", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->triad_Phi[i][a];
    	}
    }
    status = H5LTmake_dataset(group_id, "TriadAverageAngle", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Triad Average Angle", t, snap);
        exit(1);
    }

    //---------------------- Write the 1d contributions phase order data
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->triad_R_1d[i][a];
    	}
    }
    dset_dims_2d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_2d[1] = sys_vars->num_k3_sectors;
	status = H5LTmake_dataset(group_id, "TriadPhaseSync_1D", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Triad Sync Parameter 1D", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->triad_Phi_1d[i][a];
    	}
    }
    status = H5LTmake_dataset(group_id, "TriadAverageAngle_1D", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Triad Average Angle 1D", t, snap);
        exit(1);
    }

    //-------------------------- Phase Sync Across sector
    double* tmp1 = (double*) fftw_malloc(sizeof(double) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors);
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
	    		tmp1[sys_vars->num_k1_sectors * (i * sys_vars->num_k3_sectors + a) + l] = proc_data->triad_R_2d[i][a][l];
    		}
    	}
    }
    dset_dims_3d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_3d[1] = sys_vars->num_k3_sectors;
    dset_dims_3d[2] = sys_vars->num_k1_sectors;
	status = H5LTmake_dataset(group_id, "TriadPhaseSync_2D", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, tmp1);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Triad Sync Parameter Across Sector", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
	    		tmp1[sys_vars->num_k1_sectors * (i * sys_vars->num_k3_sectors + a) + l] = proc_data->triad_Phi_2d[i][a][l];
    		}
    	}
    }
    status = H5LTmake_dataset(group_id, "TriadAverageAngle_2D", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, tmp1);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Triad Average Angle Across Sector", t, snap);
        exit(1);
    }

    /// ----------------------- Enstrophy Flux and Dissipation in/out of C_theta
    dset_dims_1d[0] = sys_vars->num_k3_sectors;
    status = H5LTmake_dataset(group_id, "EnstrophyFlux_C_theta", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_flux_C_theta);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Flux C_theta", t, snap);
        exit(1);
    }
    status = H5LTmake_dataset(group_id, "EnstrophyDiss_C_theta", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_diss_C_theta);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Dissipation C_theta", t, snap);
        exit(1);
    }    

    /// ----------------------- Collectvie phase order parameter C_theta
    dset_dims_1d[0] = sys_vars->num_k3_sectors;
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta", Dims1D, dset_dims_1d, file_info->COMPLEX_DTYPE, proc_data->phase_order_C_theta);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Collective Phase Order C_theta", t, snap);
        exit(1);
    }
    fftw_complex* tmp_cmplx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors);
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp_cmplx[i * sys_vars->num_k3_sectors + a] = proc_data->phase_order_C_theta_triads[i][a];
    	}
    }
    dset_dims_2d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_2d[1] = sys_vars->num_k3_sectors;
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta_Triads", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, tmp_cmplx);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Collective Phase Order C_theta Triads", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp_cmplx[i * sys_vars->num_k3_sectors + a] = proc_data->phase_order_C_theta_triads_unidirec[i][a];
    	}
    }
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta_Triads_Unidirectional", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, tmp_cmplx);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Unidirectional Collective Phase Order C_theta Triads", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp_cmplx[i * sys_vars->num_k3_sectors + a] = proc_data->phase_order_C_theta_triads_1d[i][a];
    	}
    }
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta_Triads_1D", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, tmp_cmplx);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Collective Phase Order C_theta Triads 1D", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp_cmplx[i * sys_vars->num_k3_sectors + a] = proc_data->phase_order_C_theta_triads_unidirec_1d[i][a];
    	}
    }
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta_Triads_1D_Unidirectional", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, tmp_cmplx);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Unidirectional Collective Phase Order C_theta Triads 1D", t, snap);
        exit(1);
    }
    fftw_complex* tmp1_cmplx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors);
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
    			tmp1_cmplx[i * sys_vars->num_k3_sectors + a] = proc_data->phase_order_C_theta_triads_2d[i][a][l];
    		}
    	}
    }
    dset_dims_3d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_3d[1] = sys_vars->num_k3_sectors;
    dset_dims_3d[2] = sys_vars->num_k1_sectors;
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta_Triads_2D", Dims3D, dset_dims_3d, file_info->COMPLEX_DTYPE, tmp1_cmplx);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Collective Phase Order C_theta_Triads 2D", t, snap);
        exit(1);
    }
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
    			tmp1_cmplx[i * sys_vars->num_k3_sectors + a] = proc_data->phase_order_C_theta_triads_unidirec_2d[i][a][l];
    		}
    	}
    }
    status = H5LTmake_dataset(group_id, "CollectivePhaseOrder_C_theta_Triads_2D_Unidirectional", Dims3D, dset_dims_3d, file_info->COMPLEX_DTYPE, tmp1_cmplx);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Unidirectional Collective Phase Order C_theta_Triads 2D", t, snap);
        exit(1);
    }

    fftw_free(tmp_cmplx);
    fftw_free(tmp1_cmplx);

    ///------------------------ Enstrophy Flux
    dset_dims_1d[0] = NUM_TRIAD_TYPES + 1;
    status = H5LTmake_dataset(group_id, "EnstrophyFluxTest", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_flux_test);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Flux Per Sector", t, snap);
        exit(1);
    }

    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->enst_flux[i][a];
    	}
    }
    dset_dims_2d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_2d[1] = sys_vars->num_k3_sectors;
    status = H5LTmake_dataset(group_id, "EnstrophyFluxPerSector", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Flux Per Sector", t, snap);
        exit(1);
    }
    // 1D contribution
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->enst_flux_1d[i][a];
    	}
    }
    status = H5LTmake_dataset(group_id, "EnstrophyFluxPerSector_1D", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Flux Per Sector 1D", t, snap);
        exit(1);
    }
    // Across Sector
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
	    		tmp1[sys_vars->num_k1_sectors * (i * sys_vars->num_k3_sectors + a) + l] = proc_data->enst_flux_2d[i][a][l];
    		}
    	}
    }
    dset_dims_3d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_3d[1] = sys_vars->num_k3_sectors;
    dset_dims_3d[2] = sys_vars->num_k1_sectors;
    status = H5LTmake_dataset(group_id, "EnstrophyFluxPerSector_2D", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, tmp1);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Flux Per Sector Across Sector", t, snap);
        exit(1);
    }

    ///------------------------ Enstorphy Dissipation Field
    dset_dims_2d[0] = sys_vars->N[0];
    dset_dims_2d[1] = sys_vars->N[1] / 2 + 1;
    status = H5LTmake_dataset(group_id, "EnstrophyDissipationField", Dims2D, dset_dims_2d, file_info->COMPLEX_DTYPE, proc_data->enst_diss_field);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Enstrophy Dissipation Field", t, snap);
        exit(1);
    }

    // Free tmp memory
    fftw_free(tmp);
    fftw_free(tmp1);



    ///------------------------- Phase Order Stats Data
    #if defined(__SEC_PHASE_SYNC_STATS)
	// Allocate temporary memory for the phases pdfs
	double* ranges = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * (proc_data->phase_sect_pdf_t[0]->n + 1));
	double* counts = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * proc_data->phase_sect_pdf_t[0]->n);

	// Save phase data in temporary memory for writing to file
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
		for (int j = 0; j < (int) proc_data->phase_sect_pdf_t[0]->n + 1; ++j) {
			ranges[i * ((int)proc_data->phase_sect_pdf_t[0]->n + 1) + j] = proc_data->phase_sect_pdf_t[i]->range[j];
			if (j < (int) proc_data->phase_sect_pdf_t[0]->n) {
				counts[i * ((int)proc_data->phase_sect_pdf_t[0]->n) + j] = proc_data->phase_sect_pdf_t[i]->bin[j];
			}
		}
	}

    dset_dims_2d[0] = sys_vars->num_k3_sectors;
    dset_dims_2d[1] = (int) proc_data->phase_sect_pdf_t[0]->n + 1;
	status = H5LTmake_dataset(group_id, "SectorPhasePDF_InTime_Ranges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Sector Phases PDF In Time Ranges", t, snap);
        exit(1);
    }
    dset_dims_2d[1] = proc_data->phase_sect_pdf_t[0]->n;
	status = H5LTmake_dataset(group_id, "SectorPhasePDF_InTime_Counts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Sector Phases PDF In Time Counts", t, snap);
        exit(1);
    }

    // Free temporary memory
    fftw_free(ranges);
    fftw_free(counts);

    // Allocate temporary memory for the triad phases pdfs
    double* triad_ranges = (double*) fftw_malloc(sizeof(double) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors * (proc_data->phase_sect_pdf_t[0]->n + 1));
    double* triad_counts = (double*) fftw_malloc(sizeof(double) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors * proc_data->phase_sect_pdf_t[0]->n);

    // Save triad phase data in temporary memory for writing to file
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int j = 0; j < (int) proc_data->phase_sect_pdf_t[0]->n + 1; ++j) {
    		for (int q = 0; q < NUM_TRIAD_TYPES + 1; ++q) {
	    		triad_ranges[(NUM_TRIAD_TYPES + 1) * (i * (proc_data->triad_sect_pdf_t[q][0]->n + 1) + j) + q] = proc_data->triad_sect_pdf_t[q][i]->range[j];
	    		if (j < (int) proc_data->triad_sect_pdf_t[q][0]->n) {
	    			triad_counts[(NUM_TRIAD_TYPES + 1) * (i * (proc_data->triad_sect_pdf_t[q][0]->n) + j) + q] = proc_data->triad_sect_pdf_t[q][i]->bin[j];
	    		}
	    	}
	    }
    }

    dset_dims_3d[0] = sys_vars->num_k3_sectors;
    dset_dims_3d[1] = proc_data->triad_sect_pdf_t[0][0]->n + 1;
    dset_dims_3d[2] = NUM_TRIAD_TYPES + 1;
	status = H5LTmake_dataset(group_id, "SectorTriadPhasePDF_InTime_Ranges", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, triad_ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Sector Triad Phases PDF In Time Ranges", t, snap);
        exit(1);
    }
    dset_dims_3d[1] = proc_data->triad_sect_pdf_t[0][0]->n;
	status = H5LTmake_dataset(group_id, "SectorTriadPhasePDF_InTime_Counts", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, triad_counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Sector Triad Phases PDF In Time Counts", t, snap);
        exit(1);
    }

    // Save triad phase data in temporary memory for writing to file
	for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int j = 0; j < (int) proc_data->phase_sect_pdf_t[0]->n + 1; ++j) {
    		for (int q = 0; q < NUM_TRIAD_TYPES + 1; ++q) {
	    		triad_ranges[(NUM_TRIAD_TYPES + 1) * (i * (proc_data->triad_sect_pdf_t[q][0]->n + 1) + j) + q] = proc_data->triad_sect_wghtd_pdf_t[q][i]->range[j];
	    		if (j < (int) proc_data->triad_sect_pdf_t[q][0]->n) {
	    			triad_counts[(NUM_TRIAD_TYPES + 1) * (i * (proc_data->triad_sect_pdf_t[q][0]->n) + j) + q] = proc_data->triad_sect_wghtd_pdf_t[q][i]->bin[j];
	    		}
	    	}
	    }
    }
    
    dset_dims_3d[0] = sys_vars->num_k3_sectors;
    dset_dims_3d[1] = proc_data->triad_sect_pdf_t[0][0]->n + 1;
    dset_dims_3d[2] = NUM_TRIAD_TYPES + 1;
	status = H5LTmake_dataset(group_id, "SectorTriadPhaseWeightedPDF_InTime_Ranges", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, triad_ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Sector Triad Phases Weighted PDF In Time Ranges", t, snap);
        exit(1);
    }
    dset_dims_3d[1] = proc_data->triad_sect_pdf_t[0][0]->n;
	status = H5LTmake_dataset(group_id, "SectorTriadPhaseWeightedPDF_InTime_Counts", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, triad_counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file  at: t = ["CYAN"%lf"RESET"] Snap = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", "Sector Triad Phases Weighted PDF In Time Counts", t, snap);
        exit(1);
    }

    // Free temporary memory
    fftw_free(triad_ranges);
    fftw_free(triad_counts);
    #endif
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
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert real part for the Complex Compound Datatype!!\n-->> Exiting...\n");
  		exit(1);
  	}

  	// Insert the imaginary part of the datatype
  	status = H5Tinsert(dtype, "i", offsetof(complex_type_tmp,im), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert imaginary part for the Complex Compound Datatype!!\n-->> Exiting...\n");
  		exit(1);
  	}

  	return dtype;
}
/**
* Wrapper function for writing any remaining data to output file
*/
void FinalWriteAndClose(void) {

	// Initialize variables
	herr_t status;
	static const hsize_t Dims1D = 1;
	hsize_t dset_dims_1d[Dims1D];        // array to hold dims of the dataset to be created
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims_2d[Dims2D];
	static const hsize_t Dims3D = 3;
	hsize_t dset_dims_3d[Dims3D];

	// -------------------------------
	// Check for Output File
	// -------------------------------
	// Check if output file exist if not create it
	if (access(file_info->output_file_name, F_OK) != 0) {
		file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create output file ["CYAN"%s"RESET"] at final write\n-->> Exiting...\n", file_info->output_file_name);
			exit(1);
		}
	}
	else {
		// Open file with default I/O access properties
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at final write\n-->> Exiting...\n", file_info->output_file_name);
			exit(1);
		}
	}

	// -------------------------------
	// Write Space Variables Datasets
	// -------------------------------
	///----------------------------------- Write Wavevector list
	#if defined(__WAVELIST)
	dset_dims_1d[0] = sys_vars->N[0];
	if ( (H5LTmake_dataset(file_info->output_file_handle, "ky", Dims1D, dset_dims_1d, H5T_NATIVE_INT, run_data->k[0])) < 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "ky");
	}
	dset_dims_1d[0] = sys_vars->N[1] / 2 + 1;
	if ( (H5LTmake_dataset(file_info->output_file_handle, "kx", Dims1D, dset_dims_1d, H5T_NATIVE_INT, run_data->k[1])) < 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "kx");
	}
	#endif

	///----------------------------------- Write Collocation Points
	#if defined(__COLLOC_PTS)
	dset_dims_1d[0] = sys_vars->N[0];
	if ( (H5LTmake_dataset(file_info->output_file_handle, "x", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, run_data->x[0])) < 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "x");
	}
	dset_dims_1d[0] = sys_vars->N[1];
	if ( (H5LTmake_dataset(file_info->output_file_handle, "y", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, run_data->x[1]))< 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "y");
	}
	#endif	



	// -------------------------------
	// Write Flux
	// -------------------------------
	///----------------------------------- Write the Enstrophy Flux out of & Dissipation in C
	#if defined(__ENST_FLUX)
	// Write the enstrophy flux out of the set C
	dset_dims_1d[0] = sys_vars->num_snaps;
	status = H5LTmake_dataset(file_info->output_file_handle, "EnstrophyFluxC", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_flux_C);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Enstrophy Flux in C");
        exit(1);
    }
    // Write the enstrophy dissipation in the set C
    status = H5LTmake_dataset(file_info->output_file_handle, "EnstrophyDissC", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enst_diss_C);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Enstrophy Dissipation in C");
        exit(1);
    }
	#endif
	///----------------------------------- Write the Energy Flux out of & Dissipation in C
	#if defined(__ENST_FLUX)
	// Write the energy flux out of the set C
	dset_dims_1d[0] = sys_vars->num_snaps;
	status = H5LTmake_dataset(file_info->output_file_handle, "EnergyFluxC", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enrg_flux_C);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Energy Flux in C");
        exit(1);
    }
    // Write the energy dissipation in the set C
    status = H5LTmake_dataset(file_info->output_file_handle, "EnergyDissC", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->enrg_diss_C);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Energy Dissipation in C");
        exit(1);
    }
	#endif


    // -------------------------------
    // Write Stats
    // -------------------------------
	#if defined(__REAL_STATS) || defined(__VEL_INC_STATS) || defined(__VORT_INC_STATS) || defined(__VEL_STR_FUNC_STATS) || defined(__VORT_STR_FUNC_STATS) || defined(__VEL_GRAD_STATS) || defined(__VORT_GRAD_STATS)
    WriteStatsToFile();
	#endif



	#if defined(__SEC_PHASE_SYNC)
	///-------------------------- Sector Angles
	// k3 sector angles
    dset_dims_1d[0] = sys_vars->num_k3_sectors;
	status = H5LTmake_dataset(file_info->output_file_handle, "SectorAngles_k3", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->theta_k3);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Sector Angles");
        exit(1);
    }	
    // k1 sector angles
    dset_dims_1d[0] = sys_vars->num_k1_sectors;
	status = H5LTmake_dataset(file_info->output_file_handle, "SectorAngles_k1", Dims1D, dset_dims_1d, H5T_NATIVE_DOUBLE, proc_data->theta_k1);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Sector Angles");
        exit(1);
    }	

    dset_dims_1d[0] = NUM_TRIAD_TYPES + 1;
	status = H5LTmake_dataset(file_info->output_file_handle, "NumTriadsTest", Dims1D, dset_dims_1d, H5T_NATIVE_INT, proc_data->num_triads_test);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Number of Triads Per Sector");
        exit(1);
    }
    double* tmp_test = (double*) fftw_malloc(sizeof(double) * proc_data->num_triads_test[0] * (6));
    for (int i = 0; i < 6; ++i) {
    	for (int a = 0; a < proc_data->num_triads_test[0]; ++a) {
    		tmp_test[i * proc_data->num_triads_test[0] + a] = proc_data->phase_sync_wave_vecs_test[i][a];
    	}
    }
    dset_dims_2d[0] = 6;
    dset_dims_2d[1] = proc_data->num_triads_test[0];
	status = H5LTmake_dataset(file_info->output_file_handle, "WavevectorDataTest", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, tmp_test);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Wavevector Data Test");
        exit(1);
    }
    fftw_free(tmp_test);

    ///------------------------- Number of Triads Per Sector
    int* tmp = (int*) fftw_malloc(sizeof(int) * sys_vars->num_k3_sectors * (NUM_TRIAD_TYPES + 1));
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->num_triads[i][a];
    	}
    }
    dset_dims_2d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_2d[1] = sys_vars->num_k3_sectors;
	status = H5LTmake_dataset(file_info->output_file_handle, "NumTriadsPerSector", Dims2D, dset_dims_2d, H5T_NATIVE_INT, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Number of Triads Per Sector");
        exit(1);
    }
    // 1d contribution
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		tmp[i * sys_vars->num_k3_sectors + a] = proc_data->num_triads_1d[i][a];
    	}
    }
    dset_dims_2d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_2d[1] = sys_vars->num_k3_sectors;
	status = H5LTmake_dataset(file_info->output_file_handle, "NumTriadsPerSector_1D", Dims2D, dset_dims_2d, H5T_NATIVE_INT, tmp);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Number of Triads Per Sector 1D");
        exit(1);
    }
    // Across Sector
    int* tmp1 = (int*) fftw_malloc(sizeof(int) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors * (NUM_TRIAD_TYPES + 1));
    for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
    	for (int a = 0; a < sys_vars->num_k3_sectors; ++a) {
    		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
    			tmp1[sys_vars->num_k1_sectors * (i * sys_vars->num_k3_sectors + a) + l] = proc_data->num_triads_2d[i][a][l];
    		}
    	}
    }
    dset_dims_3d[0] = NUM_TRIAD_TYPES + 1;
    dset_dims_3d[1] = sys_vars->num_k3_sectors;
    dset_dims_3d[2] = sys_vars->num_k1_sectors;
	status = H5LTmake_dataset(file_info->output_file_handle, "NumTriadsPerSector_2D", Dims3D, dset_dims_3d, H5T_NATIVE_INT, tmp1);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!!!\n-->> Exiting...\n", "Number of Triads Per Sector Across Sector");
        exit(1);
    }

    // free temporary memory
    fftw_free(tmp);
    fftw_free(tmp1);
    
    #if defined(__SEC_PHASE_SYNC_STATS)
    ///-------------------------- Sector Phase PDFs
    // Allocate temporary memory
	double* ranges = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * (proc_data->phase_sect_pdf[0]->n + 1));
	double* counts = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * proc_data->phase_sect_pdf[0]->n);
    
    // Save phase data in temporary memory for writing to file
    for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int j = 0; j < (int) proc_data->phase_sect_pdf[0]->n + 1; ++j) {
    		ranges[i * (proc_data->phase_sect_pdf[0]->n + 1) + j] = proc_data->phase_sect_pdf[i]->range[j];
    		if (j < (int) proc_data->phase_sect_pdf[0]->n) {
    			counts[i * (proc_data->phase_sect_pdf[0]->n) + j] = proc_data->phase_sect_pdf[i]->bin[j];
    		}
    	}
    }
    dset_dims_2d[0] = sys_vars->num_k3_sectors;
    dset_dims_2d[1] = proc_data->phase_sect_pdf[0]->n + 1;
	status = H5LTmake_dataset(file_info->output_file_handle, "SectorPhasePDFRanges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Sector Phases PDF Ranges");
        exit(1);
    }
    dset_dims_2d[1] = proc_data->phase_sect_pdf[0]->n;
	status = H5LTmake_dataset(file_info->output_file_handle, "SectorPhasePDFCounts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Sector Phases PDF Counts");
        exit(1);
    }

    // Free temporary memory
    fftw_free(ranges);
    fftw_free(counts);

    /// ------------------------- Sector Triad Phase PDFs
    // Allocate temporary memory
	double* triad_ranges = (double*) fftw_malloc(sizeof(double) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors * (proc_data->phase_sect_pdf[0]->n + 1));
	double* triad_counts = (double*) fftw_malloc(sizeof(double) * (NUM_TRIAD_TYPES + 1) * sys_vars->num_k3_sectors * proc_data->phase_sect_pdf[0]->n);
    
    // Save phase data in temporary memory for writing to file
    for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int j = 0; j < (int) proc_data->triad_sect_pdf[0][0]->n + 1; ++j) {
    		for (int q = 0; q < NUM_TRIAD_TYPES + 1; ++q) {
				triad_ranges[(NUM_TRIAD_TYPES + 1) * (i * (proc_data->triad_sect_pdf[q][0]->n + 1) + j) + q] = proc_data->triad_sect_pdf[q][i]->range[j];
	    		if (j < (int) proc_data->triad_sect_pdf[q][0]->n) {
	    			triad_counts[(NUM_TRIAD_TYPES + 1) * (i * proc_data->triad_sect_pdf[q][0]->n + j) + q] = proc_data->triad_sect_pdf[q][i]->bin[j];
	    		}
       		}
       	}
    }
    dset_dims_3d[0] = sys_vars->num_k3_sectors;
    dset_dims_3d[1] = proc_data->triad_sect_pdf[0][0]->n + 1;
    dset_dims_3d[2] = NUM_TRIAD_TYPES + 1;
	status = H5LTmake_dataset(file_info->output_file_handle, "SectorTriadPhasePDFRanges", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, triad_ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Sector Triad Phases PDF Ranges");
        exit(1);
    }
    dset_dims_3d[1] = proc_data->triad_sect_pdf[0][0]->n;
	status = H5LTmake_dataset(file_info->output_file_handle, "SectorTriadPhasePDFCounts", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, triad_counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Sector Triad Phases PDF Counts");
        exit(1);
    }	

    // Free temporary memory
    fftw_free(triad_ranges);
    fftw_free(triad_counts);

    ///-------------------------- 2D Flux Contribution Histograms
    // Allocate temporary memory
	double* ranges = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors * (proc_data->triad_R_2d_pdf[0][0]->n + 1));
	double* counts = (double*) fftw_malloc(sizeof(double) * sys_vars->num_k3_sectors * sys_vars->num_k1_sectors * proc_data->triad_R_2d_pdf[0][0]->n);
    
    // Save phase data in temporary memory for writing to file
    for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
	    	for (int j = 0; j < (int) proc_data->triad_R_2d_pdf[0][0]->n + 1; ++j) {
	    		ranges[(int) (proc_data->triad_R_2d_pdf[0][0]->n + 1) * (i * (sys_vars->num_k1_sectors) + l) + j] = proc_data->triad_R_2d_pdf[i][l]->range[j];
	    		if (j < (int) proc_data->triad_R_2d_pdf[0][0]->n) {
	    			counts[(int) (proc_data->triad_R_2d_pdf->n) * (i * (sys_vars->num_k1_sectors) + l) + j] = proc_data->triad_R_2d_pdf[i][l]->bin[j];
	    		}
	    	}
    	}
    }
    dset_dims_3d[0] = sys_vars->num_k3_sectors;
    dset_dims_3d[1] = sys_vars->num_k1_sectors;
    dset_dims_3d[2] = proc_data->triad_R_2d_pdf[0][0]->n + 1;
	status = H5LTmake_dataset(file_info->output_file_handle, "Phase_Sync_2D_PDFRanges", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Phase Sync PDF Ranges");
        exit(1);
    }
    dset_dims_3d[2] = proc_data->triad_R_2d_pdf[0][l]->n;
	status = H5LTmake_dataset(file_info->output_file_handle, "Phase_Sync_2D_PDFCounts", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Phase Sync PDF Counts");
        exit(1);
    }

    // Save phase data in temporary memory for writing to file
    for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
	    	for (int j = 0; j < (int) proc_data->triad_Phi_2d_pdf[0][0]->n + 1; ++j) {
	    		ranges[(int) (proc_data->triad_Phi_2d_pdf[0][0]->n + 1) * (i * (sys_vars->num_k1_sectors) + l) + j] = proc_data->triad_Phi_2d_pdf[i][l]->range[j];
	    		if (j < (int) proc_data->triad_Phi_2d_pdf[0][0]->n) {
	    			counts[(int) (proc_data->triad_Phi_2d_pdf->n) * (i * (sys_vars->num_k1_sectors) + l) + j] = proc_data->triad_Phi_2d_pdf[i][l]->bin[j];
	    		}
	    	}
    	}
    }
    dset_dims_3d[2] = proc_data->triad_Phi_2d_pdf[0][0]->n + 1;
	status = H5LTmake_dataset(file_info->output_file_handle, "Average_Phase_2D_PDFRanges", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Average Phase PDF Ranges");
        exit(1);
    }
    dset_dims_3d[2] = proc_data->triad_Phi_2d_pdf[0][0]->n;
	status = H5LTmake_dataset(file_info->output_file_handle, "Average_Phase_2D_PDFCounts", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Average Phase PDF Counts");
        exit(1);
    }

    // Save phase data in temporary memory for writing to file
    for (int i = 0; i < sys_vars->num_k3_sectors; ++i) {
    	for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
	    	for (int j = 0; j < (int) proc_data->enst_flux_2d_pdf[0][0]->n + 1; ++j) {
	    		ranges[(int) (proc_data->enst_flux_2d_pdf[0][0]->n + 1) * (i * (sys_vars->num_k1_sectors) + l) + j] = proc_data->enst_flux_2d_pdf[i][l]->range[j];
	    		if (j < (int) proc_data->enst_flux_2d_pdf[0][0]->n) {
	    			counts[(int) (proc_data->enst_flux_2d_pdf->n) * (i * (sys_vars->num_k1_sectors) + l) + j] = proc_data->enst_flux_2d_pdf[i][l]->bin[j];
	    		}
	    	}
    	}
    }
    dset_dims_3d[2] = proc_data->enst_flux_2d_pdf[0][0]->n + 1;
	status = H5LTmake_dataset(file_info->output_file_handle, "Enstrophy_Flux_2D_PDFRanges", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, ranges);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Enstrophy Flux PDF Ranges");
        exit(1);
    }
    dset_dims_3d[2] = proc_data->enst_flux_2d_pdf[0][0]->n;
	status = H5LTmake_dataset(file_info->output_file_handle, "Enstrophy_Flux_2D_PDFCounts", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, counts);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at final write!!\n-->> Exiting...\n", "Enstrophy Flux PDF Counts");
        exit(1);
    }

    // Free temporary memory
    fftw_free(ranges);
    fftw_free(counts);
    #endif
    #endif

    // -------------------------------
    // Close HDF5 Identifiers
    // -------------------------------
    // status = H5Fclose(file_info->output_file_handle);
    // if (status < 0) {
    // 	fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at final write\n-->> Exiting...\n", file_info->output_file_name);
    // 	exit(1);
    // }
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
