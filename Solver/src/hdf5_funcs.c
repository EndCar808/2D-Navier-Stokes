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
#include <sys/stat.h>
#include <sys/types.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "solver.h"
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Wrapper function that creates the ouput directory creates and opens the main output file using parallel access and the spectra file using normal serial access
 */
void CreateOutputFilesWriteICs(const long int* N, double dt) {

	// Initialize variabeles
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	hid_t main_group_id;
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	hid_t spectra_group_id = (hid_t) NULL;
	#endif
	#if defined(__PHASE_SYNC)
	hid_t sync_group_id = (hid_t) NULL;
	#endif
	#if defined(__VORT_REAL) || defined(__MODES) || defined(__REALSPACE) || defined(__PHASE_ONLY)
	int tmp;
	int indx;
	#endif
	char group_name[128];
	herr_t status;
	hid_t plist_id;

    #if (defined(__VORT_FOUR) || defined(__MODES) || defined(__NONLIN)) && !defined(DEBUG)
	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();
	#endif
		
	///////////////////////////
	/// Create & Open Files
	///////////////////////////
	// -----------------------------------
	// Create Output Directory and Path
	// -----------------------------------
	GetOutputDirPath();

	// ------------------------------------------
	// Create Parallel File PList for Main File
	// ------------------------------------------
	// Create proptery list for main file access and set to parallel I/O
	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	status   = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
	if (status < 0) {
		printf("\n["RED"ERROR"RESET"] --- Could not set parallel I/O access for HDF5 output file! \n-->>Exiting....\n");
		exit(1);
	}

	// ---------------------------------
	// Create the output files
	// ---------------------------------
	// Create the main output file
	file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	if (file_info->output_file_handle < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create main HDF5 output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->output_file_name);
		exit(1);
	}

	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank){
		// Create the spectra output file
		file_info->spectra_file_handle = H5Fcreate(file_info->spectra_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->spectra_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 spectra output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->spectra_file_name);
			exit(1);
		}
	}
	#endif

	#if defined(__PHASE_SYNC)
	if (!sys_vars->rank){
		// Create the Phase sync output file
		file_info->sync_file_handle = H5Fcreate(file_info->sync_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->sync_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 spectra output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->sync_file_name);
			exit(1);
		}
	}
	#endif


	////////////////////////////////
	/// Write Initial Condtions
	////////////////////////////////
	if (sys_vars->TRANS_ITERS_FLAG != TRANSIENT_ITERS) {
		// --------------------------------------
		// Create Group for Initial Conditions
		// --------------------------------------
		// Initialize Group Name
		sprintf(group_name, "/Iter_%05d", 0);
		
		// Create group for the current iteration data
		main_group_id = CreateGroup(file_info->output_file_handle, file_info->output_file_name, group_name, 0.0, dt, 0);
		#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
		if (!sys_vars->rank) {
			spectra_group_id = CreateGroup(file_info->spectra_file_handle, file_info->spectra_file_name, group_name, 0.0, dt, 0);
		}
		#endif
		#if defined(__PHASE_SYNC)
		if (!sys_vars->rank) {
			sync_group_id = CreateGroup(file_info->sync_file_handle, file_info->sync_file_name, group_name, 0.0, dt, 0);
		}
		#endif


		// --------------------------------------
		// Write Initial Conditions
		// --------------------------------------
		// Create dimension arrays
		static const int d_set_rank2D = 2;
		hsize_t dset_dims2D[d_set_rank2D];        // array to hold dims of the dataset to be created
		hsize_t slab_dims2D[d_set_rank2D];	      // Array to hold the dimensions of the hyperslab
		hsize_t mem_space_dims2D[d_set_rank2D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
		#if defined(__MODES) || defined(__REALSPACE)
		static const int d_set_rank3D = 3;
		hsize_t dset_dims3D[d_set_rank3D];        // array to hold dims of the dataset to be created
		hsize_t slab_dims3D[d_set_rank3D];	      // Array to hold the dimensions of the hyperslab
		hsize_t mem_space_dims3D[d_set_rank3D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
		#endif


		///-------------------------------------- Real Space Voriticity
		#if defined(__VORT_REAL)
		// Transform vorticity back to real space and normalize
		fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->w_hat, run_data->w);
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double) (Nx * Ny);
			}
		}

		// Specify dataset dimensions
		dset_dims2D[0] 	  = Nx;
		dset_dims2D[1] 	  = Ny;
		slab_dims2D[0]      = sys_vars->local_Nx;
		slab_dims2D[1]      = Ny;
		mem_space_dims2D[0] = sys_vars->local_Nx;
		mem_space_dims2D[1] = Ny + 2;

		// Write the real space vorticity
		WriteDataReal(0.0, 0, main_group_id, "w", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->w);
		#endif

		///-------------------------------------- Fourier Space Voriticity
		#if defined(__VORT_FOUR)
		// Create dimension arrays
		dset_dims2D[0] 	  = Nx;
		dset_dims2D[1] 	  = Ny_Fourier;
		slab_dims2D[0] 	  = sys_vars->local_Nx;
		slab_dims2D[1] 	  = Ny_Fourier;
		mem_space_dims2D[0] = sys_vars->local_Nx;
		mem_space_dims2D[1] = Ny_Fourier;

		// Write the real space vorticity
		WriteDataFourier(0.0, 0, main_group_id, "w_hat", file_info->COMPLEX_DTYPE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->w_hat);
		#endif

		#if defined(__MODES) || defined(__REALSPACE)
		fftw_complex k_sqr;

		// Get the Fourier velocities
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
					// Compute the prefactor
					k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill the Fourier velocities
					run_data->u_hat[SYS_DIM * indx + 0] = k_sqr * run_data->k[1][j] * run_data->w_hat[indx];
					run_data->u_hat[SYS_DIM * indx + 1] = -k_sqr * run_data->k[0][i] * run_data->w_hat[indx];
				}
				else {
					run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
				}
			}
		}
		#endif
		
		///-------------------------------------- Fourier Space Velocity	
		#if defined(__MODES)
		// Create dimension arrays
		dset_dims3D[0] 	    = Nx;
		dset_dims3D[1] 	    = Ny_Fourier;
		dset_dims3D[2]      = SYS_DIM;
		slab_dims3D[0] 	    = sys_vars->local_Nx;
		slab_dims3D[1] 	    = Ny_Fourier;
		slab_dims3D[2]      = SYS_DIM;
		mem_space_dims3D[0] = sys_vars->local_Nx;
		mem_space_dims3D[1] = Ny_Fourier;
		mem_space_dims3D[2] = SYS_DIM;

		// Write the real space vorticity
		WriteDataFourier(0.0, 0, main_group_id, "u_hat", file_info->COMPLEX_DTYPE, d_set_rank3D, dset_dims3D, slab_dims3D, mem_space_dims3D, sys_vars->local_Nx_start, run_data->u_hat);
		#endif

		///-------------------------------------- Real Space Velocity
		#if defined(__REALSPACE)
		// Transform velocities back to real space and normalize
		fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, run_data->u_hat, run_data->u);
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->u[SYS_DIM * indx + 0] *= 1.0 / (double) (Nx * Ny);
				run_data->u[SYS_DIM * indx + 1] *= 1.0 / (double) (Nx * Ny);
			}
		}

		// Specify dataset dimensions
		dset_dims3D[0] 	  = Nx;
		dset_dims3D[1] 	  = Ny;
		dset_dims3D[2]    = SYS_DIM;
		slab_dims3D[0]    = sys_vars->local_Nx;
		slab_dims3D[1]    = Ny;
		slab_dims3D[2]    = SYS_DIM;
		mem_space_dims3D[0] = sys_vars->local_Nx;
		mem_space_dims3D[1] = (Ny + 2);
		mem_space_dims3D[2] = SYS_DIM;

		// Write the real space vorticity
		WriteDataReal(0.0, 0, main_group_id, "u", H5T_NATIVE_DOUBLE, d_set_rank3D, dset_dims3D, slab_dims3D, mem_space_dims3D, sys_vars->local_Nx_start, run_data->u);
		#endif

		///-------------------------------------- Fourier Phases & Amplitudes of the Vorticity
		#if defined(PHASE_ONLY)
		// Record the Fourier amplitudes and phases
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				//record the amplitudes
				run_data->a_k[i] = cabs(run_data->w_hat[indx]);
				// record the phases 
				run_data->phi_k[i] = carg(run_data->w_hat[indx]);
			}
		}
		
		// Create dimension arrays for the Fourier amplitudes and phases
		dset_dims2D[0] 	  = Nx;
		dset_dims2D[1] 	  = Ny_Fourier;
		slab_dims2D[0] 	  = sys_vars->local_Nx;
		slab_dims2D[1] 	  = Ny_Fourier;
		mem_space_dims2D[0] = sys_vars->local_Nx;
		mem_space_dims2D[1] = Ny_Fourier;

		// Write the Fourier amplitudes
		WriteDataReal(0.0, 0, main_group_id, "a_k", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->a_k);

		// Write the Fourier phases
		WriteDataReal(0.0, 0, main_group_id, "phi_k", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->phi_k);
		#endif

		///-------------------------------------- Taylor Green Initial Condition
		#if defined(TESTING)
		if (!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
			// Create dimension arrays
			dset_dims2D[0] 	    = Nx;
			dset_dims2D[1] 	    = Ny;
			slab_dims2D[0]      = sys_vars->local_Nx;
			slab_dims2D[1]      = Ny;
			mem_space_dims2D[0] = sys_vars->local_Nx;
			mem_space_dims2D[1] = Ny + 2;

			// Write the real space vorticity
			WriteDataReal(0.0, 0, main_group_id, "TGSoln", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->tg_soln);	
		}
		#endif

		///-------------------------------------- Spectra & Phase Sync
		#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT) || defined(__PHASE_SYNC)
		// Gather Spectra data and write to file
		if (!sys_vars->rank) {
			// Gather spectra data on master process & write to file
			#if defined(__ENST_SPECT)
			MPI_Reduce(MPI_IN_PLACE, run_data->enst_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "EnstrophySpectrum", run_data->enst_spect);
			#endif 
			#if defined(__ENRG_SPECT)
			MPI_Reduce(MPI_IN_PLACE, run_data->enrg_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "EnergySpectrum", run_data->enrg_spect);
			#endif
			#if defined(__ENST_FLUX_SPECT)
			MPI_Reduce(MPI_IN_PLACE, run_data->d_enst_dt_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(MPI_IN_PLACE, run_data->enst_diss_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(MPI_IN_PLACE, run_data->enst_flux_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "TimeDerivativeEnstrophySpectrum", run_data->d_enst_dt_spect);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "EnstrophyDissipationSpectrum", run_data->enst_diss_spect);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "EnstrophyFluxSpectrum", run_data->enst_flux_spect);
			#endif
			#if defined(__ENRG_FLUX_SPECT)
			MPI_Reduce(MPI_IN_PLACE, run_data->d_enrg_dt_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(MPI_IN_PLACE, run_data->enrg_diss_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(MPI_IN_PLACE, run_data->enrg_flux_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "TimeDerivativeEnergySpectrum", run_data->d_enrg_dt_spect);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "EnergyDissipationSpectrum", run_data->enrg_diss_spect);
			WriteDataSerial(0.0, 0, spectra_group_id, sys_vars->n_spect, "EnergyFluxSpectrum", run_data->enrg_flux_spect);
			#endif
			#if defined(__PHASE_SYNC)
			MPI_Reduce(MPI_IN_PLACE, run_data->phase_order_k, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(MPI_IN_PLACE, run_data->normed_phase_order_k, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

			// Create temp mem
			double amp_k[sys_vars->n_spect];
			double phase_k[sys_vars->n_spect];
			double normed_amp_k[sys_vars->n_spect];
			double normed_phase_k[sys_vars->n_spect];
			for (int i = 0; i < sys_vars->n_spect; ++i) {
				amp_k[i]          = cabs(run_data->phase_order_k[i]);
				phase_k[i]        = carg(run_data->phase_order_k[i]);
				normed_amp_k[i]   = cabs(run_data->normed_phase_order_k[i]);
				normed_phase_k[i] = carg(run_data->normed_phase_order_k[i]);
			}
			WriteDataSerial(0.0, 0, sync_group_id, sys_vars->n_spect, "PhaseOrder_Theta_k", phase_k);
			WriteDataSerial(0.0, 0, sync_group_id, sys_vars->n_spect, "PhaseOrder_R_k", amp_k);
			WriteDataSerial(0.0, 0, sync_group_id, sys_vars->n_spect, "NormedPhaseOrder_Theta_k", normed_phase_k);
			WriteDataSerial(0.0, 0, sync_group_id, sys_vars->n_spect, "NormedPhaseOrder_R_k", normed_amp_k);
			#endif
		}
		else {
			// Reduce all other process to master rank
			#if defined(__ENST_SPECT)
			MPI_Reduce(run_data->enst_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			#endif
			#if defined(__ENRG_SPECT)
			MPI_Reduce(run_data->enrg_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			#endif
			#if defined(__ENST_FLUX_SPECT)
			MPI_Reduce(run_data->d_enst_dt_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(run_data->enst_diss_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(run_data->enst_flux_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			#endif
			#if defined(__ENRG_FLUX_SPECT)
			MPI_Reduce(run_data->d_enrg_dt_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(run_data->enrg_diss_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(run_data->enrg_flux_spect, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			#endif
			#if defined(__PHASE_SYNC)
			MPI_Reduce(run_data->phase_order_k, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(run_data->normed_phase_order_k, NULL,  sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			#endif
		}
		#endif
	}

	
	
	// ------------------------------------
	// Close Identifiers - also close file
	// ------------------------------------
	status = H5Pclose(plist_id);
	if (sys_vars->TRANS_ITERS_FLAG != TRANSIENT_ITERS) {
		status = H5Gclose(main_group_id);
	}
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, 0, 0.0);
		exit(1);		
	}
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (sys_vars->TRANS_ITERS_FLAG != TRANSIENT_ITERS) {
		if (!sys_vars->rank) {
			status = H5Gclose(spectra_group_id);
			status = H5Fclose(file_info->spectra_file_handle);
			if (status < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, 0, 0.0);
				exit(1);		
			}
		}
	}
	#endif
	#if defined(__PHASE_SYNC)
	if (sys_vars->TRANS_ITERS_FLAG != TRANSIENT_ITERS) {
		if (!sys_vars->rank) {
				status = H5Gclose(sync_group_id);
				status = H5Fclose(file_info->sync_file_handle);
				if (status < 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->sync_file_name, 0, 0.0);
					exit(1);		
				}
			}
		}
	}
	#endif
}
/**
 * Function that creates the output file paths and directories
 */
void GetOutputDirPath(void) {

	// Initialize variables
	char sys_type[64];
	char solv_type[64];
	char model_type[64];
	char tmp_path[512];
	char file_data[512];  
	struct stat st = {0};	// this is used to check whether the output directories exist or not.

	// ----------------------------------
	// Check if Provided Directory Exists
	// ----------------------------------
	if (!sys_vars->rank) {
		// Check if output directory exists
		if (stat(file_info->output_dir, &st) == -1) {
			printf("\n["YELLOW"NOTE"RESET"] --- Provided Output directory doesn't exist, now creating it...\n");
			// If not then create it
			if ((mkdir(file_info->output_dir, 0700)) == -1) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create provided output directory ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_dir);
				exit(1);
			}
		}

	}

	////////////////////////////////////////////
	// Check if Output File Only is Requested
	////////////////////////////////////////////
	if (file_info->file_only) {
		// Update to screen that file only output option is selected
		printf("\n["YELLOW"NOTE"RESET"] --- File only output option selected...\n");
		
		// ----------------------------------
		// Get Simulation Details
		// ----------------------------------
		#if defined(__NAVIER)
		sprintf(sys_type, "%s", "NAV");
		#elif defined(__EULER)
		sprintf(sys_type, "%s", "EUL");
		#else
		sprintf(sys_type, "%s", "UKN");
		#endif
		#if defined(__RK4)
		sprintf(solv_type, "%s", "RK4");
		#elif defined(__RK4CN)
		sprintf(solv_type, "%s", "RK4CN");
		#elif defined(__RK5)
		sprintf(solv_type, "%s", "RK5");
		#elif defined(__DPRK5)
		sprintf(solv_type, "%s", "DP5");
		#elif defined(__AB4)
		sprintf(solv_type, "%s", "AB4");
		#else 
		sprintf(solv_type, "%s", "UKN");
		#endif
		#if defined(__PHASE_ONLY)
		sprintf(model_type, "%s", "PO");
		#else
		sprintf(model_type, "%s", "FULL");
		#endif

		// -------------------------------------
		// Get File Label from Simulation Data
		// -------------------------------------
		// Construct file label from simulation data
		sprintf(file_data, "_SIM[%s-%s-%s]_N[%ld,%ld]_T[%1.1lf,%g,%1.3lf]_NU[%g,%d,%1.1lf]_DRAG[%g,%d,%1.1lf]_CFL[%1.2lf]_FORC[%s,%d,%g]_u0[%s].h5", 
											sys_type, solv_type, model_type, 
											sys_vars->N[0], sys_vars->N[1], 
											sys_vars->t0, sys_vars->dt, sys_vars->T, 
											sys_vars->NU, sys_vars->HYPER_VISC_FLAG, sys_vars->HYPER_VISC_POW,
											sys_vars->EKMN_ALPHA, sys_vars->EKMN_DRAG_FLAG, sys_vars->EKMN_DRAG_POW,
											sys_vars->CFL_CONST,
											sys_vars->forcing, sys_vars->force_k, sys_vars->force_scale_var,
											sys_vars->u0);

		// ----------------------------------
		// Construct File Paths
		// ---------------------------------- 
		// Construct main file path
		strcpy(tmp_path, file_info->output_dir);
		strcat(tmp_path, "Main_HDF_Data"); 
		strcpy(file_info->output_file_name, tmp_path); 
		strcat(file_info->output_file_name, file_data);
		if ( !(sys_vars->rank) ) {
			printf("\nMain Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);
		}

		#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
		if ( !(sys_vars->rank) ) {
			// Construct Spectra file path
			strcpy(tmp_path, file_info->output_dir);
			strcat(tmp_path, "Spectra_HDF_Data"); 
			strcpy(file_info->spectra_file_name, tmp_path); 
			strcat(file_info->spectra_file_name, file_data);
			printf("Spectra Output File: "CYAN"%s"RESET"\n\n", file_info->spectra_file_name);
		}	
		#endif

		#if defined(__PHASE_SYNC)
		if ( !(sys_vars->rank) ) {
			// Construct Phase sync file path
			strcpy(tmp_path, file_info->output_dir);
			strcat(tmp_path, "PhaseSync_HDF_Data"); 
			strcpy(file_info->sync_file_name, tmp_path); 
			strcat(file_info->sync_file_name, file_data);
			printf("Phase Sync Output File: "CYAN"%s"RESET"\n\n", file_info->sync_file_name);
		}	
		#endif
	}
	else {
		// ----------------------------------
		// Get Simulation Details
		// ----------------------------------
		#if defined(__NAVIER)
		sprintf(sys_type, "%s", "NAVIER");
		#elif defined(__EULER)
		sprintf(sys_type, "%s", "EULER");
		#else
		sprintf(sys_type, "%s", "SYS_UNKN");
		#endif
		#if defined(__RK4)
		sprintf(solv_type, "%s", "RK4");
		#elif defined(__RK4CN)
		sprintf(solv_type, "%s", "RK4CN");
		#elif defined(__RK5)
		sprintf(solv_type, "%s", "RK5");
		#elif defined(__AB4)
		sprintf(solv_type, "%s", "AB4");
		#elif defined(__DPRK5)
		sprintf(solv_type, "%s", "DP5");
		#else 
		sprintf(solv_type, "%s", "SOLV_UKN");
		#endif
		#if defined(__PHASE_ONLY)
		sprintf(model_type, "%s", "PHAEONLY");
		#else
		sprintf(model_type, "%s", "FULL");
		#endif

		// ----------------------------------
		// Construct Output folder
		// ----------------------------------
		// Construct file label from simulation data
		sprintf(file_data, "SIM_DATA_%s_%s_%s_N[%ld,%ld]_T[%1.1lf,%g,%1.3lf]_NU[%g,%d,%1.1lf]_DRAG[%g,%d,%1.1lf]_CFL[%1.2lf]_FORC[%s,%d,%g]_u0[%s]_TAG[%s]/", 
												sys_type, solv_type, model_type, 
												sys_vars->N[0], sys_vars->N[1], 
												sys_vars->t0, sys_vars->dt, sys_vars->T, 
												sys_vars->NU, sys_vars->HYPER_VISC_FLAG, sys_vars->HYPER_VISC_POW,
												sys_vars->EKMN_ALPHA, sys_vars->EKMN_DRAG_FLAG, sys_vars->EKMN_DRAG_POW,
												sys_vars->CFL_CONST, 
												sys_vars->forcing, sys_vars->force_k, sys_vars->force_scale_var,
												sys_vars->u0, 
												file_info->output_tag);
		
		// ----------------------------------
		// Check Existence of Output Folder
		// ----------------------------------
		strcat(file_info->output_dir, file_data);
		if (!sys_vars->rank) {
			// Check if folder exists
			if (stat(file_info->output_dir, &st) == -1) {
				// If not create it
				if ((mkdir(file_info->output_dir, 0700)) == -1) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create folder for output files ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_dir);
					exit(1);
				}
			}
		}

		// ----------------------------------
		// Construct File Paths
		// ---------------------------------- 
		// Construct main file path
		strcpy(file_info->output_file_name, file_info->output_dir); 
		strcat(file_info->output_file_name, "Main_HDF_Data.h5");
		if ( !(sys_vars->rank) ) {
			printf("\nMain Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);
		}
		// Construct system measures file
		strcpy(file_info->sys_msr_file_name, file_info->output_dir); 
		strcat(file_info->sys_msr_file_name, "SystemMeasures_HDF_Data.h5");
		if ( !(sys_vars->rank) ) {
			printf("\nSystem Measures File: "CYAN"%s"RESET"\n\n", file_info->sys_msr_file_name);
		}

		#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
		if ( !(sys_vars->rank) ) {
			// Construct spectra file path
			strcpy(file_info->spectra_file_name, file_info->output_dir); 
			strcat(file_info->spectra_file_name, "Spectra_HDF_Data.h5");
			printf("Spectra Output File: "CYAN"%s"RESET"\n\n", file_info->spectra_file_name);
		}	
		#endif

		#if defined(__PHASE_SYNC)
		if ( !(sys_vars->rank) ) {
			// Construct phase sync file path
			strcpy(file_info->sync_file_name, file_info->output_dir); 
			strcat(file_info->sync_file_name, "PhaseSync_HDF_Data.h5");
			printf("Phase Sync Output File: "CYAN"%s"RESET"\n\n", file_info->sync_file_name);
		}	
		#endif
		#if defined(__STATS)
		if ( !(sys_vars->rank) ) {
			// Construct phase sync file path
			strcpy(file_info->stats_file_name, file_info->output_dir); 
			strcat(file_info->stats_file_name, "Stats_HDF_Data.h5");
			printf("Stats Output File: "CYAN"%s"RESET"\n\n", file_info->stats_file_name);
		}	
		#endif
	}

	// Make All process wait before opening output files later
	MPI_Barrier(MPI_COMM_WORLD);
}
/**
 * Wrapper function that writes the data to file by openining it, creating a group for the current iteration and writing the data under this group. The file is then closed again 
 * @param t     The current time of the simulation
 * @param dt    The current timestep being used
 * @param iters The current iteration
 */
void WriteDataToFile(double t, double dt, long int iters) {

	// Initialize Variables
	#if defined(__VORT_REAL) || defined(__MODES) || defined(__REALSPACE) || defined(__PHASE_ONLY)
	int tmp;
	int indx;
	#endif
	char group_name[128];
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	herr_t status;
	hid_t main_group_id;
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	hid_t spectra_group_id = (hid_t) NULL;
	#endif
	#if defined(__PHASE_SYNC)
	hid_t sync_group_id = (hid_t) NULL;
	#endif
	hid_t plist_id;
	#if defined(__VORT_REAL) || defined(__NONLIN) || defined(__PHASE_ONLY) || defined(TESTINGT) || defined(__VORT_FOUR) || defined(TESTING)
	static const int d_set_rank2D = 2;
	hsize_t dset_dims2D[d_set_rank2D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims2D[d_set_rank2D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims2D[d_set_rank2D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
	#endif
	#if defined(__MODES) || defined(__REALSPACE)
	static const int d_set_rank3D = 3;
	hsize_t dset_dims3D[d_set_rank3D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims3D[d_set_rank3D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims3D[d_set_rank3D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
	#endif
	

	// --------------------------------------
	// Check if files exist and Open/Create
	// --------------------------------------
	// Create property list for setting parallel I/O access properties for file
	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	// Check if main file exists - open it if it does if not create it
	if (access(file_info->output_file_name, F_OK) != 0) {
		file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
			exit(1);
		}
	}
	else {
		// Open file with parallel I/O access properties
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, plist_id);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
			exit(1);
		}
	}
	H5Pclose(plist_id);

	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		// Check if spectra file exists - open it if it does if not create it
		if (access(file_info->spectra_file_name, F_OK) != 0) {
			file_info->spectra_file_handle = H5Fcreate(file_info->spectra_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			if (file_info->spectra_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create spectra file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, iters, t);
				exit(1);
			}
		}
		else {
			// Open file with serial access properties
			file_info->spectra_file_handle = H5Fopen(file_info->spectra_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
			if (file_info->spectra_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open spectra file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, iters, t);
				exit(1);
			}
		}
	}
	#endif
	#if defined(__PHASE_SYNC)
	if (!sys_vars->rank) {
		// Check if sync file exists - open it if it does if not create it
		if (access(file_info->sync_file_name, F_OK) != 0) {
			file_info->sync_file_handle = H5Fcreate(file_info->sync_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			if (file_info->sync_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create sync file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->sync_file_name, iters, t);
				exit(1);
			}
		}
		else {
			// Open file with serial access properties
			file_info->sync_file_handle = H5Fopen(file_info->sync_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
			if (file_info->sync_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open sync file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->sync_file_name, iters, t);
				exit(1);
			}
		}
	}
	#endif


	// -------------------------------
	// Create Group 
	// -------------------------------
	// Initialize Group Name
	sprintf(group_name, "/Iter_%05d", (int)iters);
	
	// Create group for the current iteration data
	main_group_id = CreateGroup(file_info->output_file_handle, file_info->output_file_name, group_name, t, dt, iters);
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		spectra_group_id = CreateGroup(file_info->spectra_file_handle, file_info->spectra_file_name, group_name, t, dt, iters);
	}
	#endif
	#if defined(__PHASE_SYNC)
	if (!sys_vars->rank) {
		sync_group_id = CreateGroup(file_info->sync_file_handle, file_info->sync_file_name, group_name, t, dt, iters);
	}
	#endif
	
	// -------------------------------
	// Write Data to File
	// -------------------------------
	///--------------------------------------- Real Space Vorticity
	#if defined(__VORT_REAL)
	// Transform Fourier space vorticiy to real space and normalize
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->w_hat, run_data->w);
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Normalize
			run_data->w[indx] *= 1.0 / (double) (Nx * Ny);
		}
	}

	// Create dimension arrays
	dset_dims2D[0] 	    = Nx;
	dset_dims2D[1] 	    = Ny;
	slab_dims2D[0]      = sys_vars->local_Nx;
	slab_dims2D[1]      = Ny;
	mem_space_dims2D[0] = sys_vars->local_Nx;
	mem_space_dims2D[1] = Ny + 2;

	// Write the real space vorticity
	WriteDataReal(t, (int)iters, main_group_id, "w", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->w);
	#endif

	///--------------------------------------- Fourier Space Vorticity
	#if defined(__VORT_FOUR)
	// Create dimension arrays
	dset_dims2D[0]      = Nx;
	dset_dims2D[1]      = Ny_Fourier;
	slab_dims2D[0]      = sys_vars->local_Nx;
	slab_dims2D[1]      = Ny_Fourier;
	mem_space_dims2D[0] = sys_vars->local_Nx;
	mem_space_dims2D[1] = Ny_Fourier;

	// Write the real space vorticity
	WriteDataFourier(t, (int)iters, main_group_id, "w_hat", file_info->COMPLEX_DTYPE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->w_hat);
	#endif
	#if defined(__MODES) || defined(__REALSPACE)
	fftw_complex k_sqr;

	// Get the Fourier velocities
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute the prefactor
				k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Fill the Fourier velocities
				run_data->u_hat[SYS_DIM * indx + 0] = k_sqr * run_data->k[1][j] * run_data->w_hat[indx];
				run_data->u_hat[SYS_DIM * indx + 1] = -k_sqr * run_data->k[0][i] * run_data->w_hat[indx];
			}
			else {
				run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
			}
		}
	}
	#endif

	///--------------------------------------- Fourier Space Velocities
	#if defined(__MODES)
	// Create dimension arrays
	dset_dims3D[0] 	    = Nx;
	dset_dims3D[1] 	    = Ny_Fourier;
	dset_dims3D[2]      = SYS_DIM;
	slab_dims3D[0] 	    = sys_vars->local_Nx;
	slab_dims3D[1] 	    = Ny_Fourier;
	slab_dims3D[2]      = SYS_DIM;
	mem_space_dims3D[0] = sys_vars->local_Nx;
	mem_space_dims3D[1] = Ny_Fourier;
	mem_space_dims3D[2] = SYS_DIM;

	// Write the real space vorticity
	WriteDataFourier(t, (int)iters, main_group_id, "u_hat", file_info->COMPLEX_DTYPE, d_set_rank3D, dset_dims3D, slab_dims3D, mem_space_dims3D, sys_vars->local_Nx_start, run_data->u_hat);
	#endif

	///--------------------------------------- Real Space Velocities
	#if defined(__REALSPACE)
	// Transform velocities back to real space and normalize
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_batch_c2r, run_data->u_hat, run_data->u);
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Normalize
			run_data->u[SYS_DIM * indx + 0] *= 1.0 / (double) (Nx * Ny);
			run_data->u[SYS_DIM * indx + 1] *= 1.0 / (double) (Nx * Ny);
		}
	}

	// Specify dataset dimensions
	dset_dims3D[0] 	    = Nx;
	dset_dims3D[1] 	    = Ny;
	dset_dims3D[2]      = SYS_DIM;
	slab_dims3D[0]      = sys_vars->local_Nx;
	slab_dims3D[1]      = Ny;
	slab_dims3D[2]      = SYS_DIM;
	mem_space_dims3D[0] = sys_vars->local_Nx;
	mem_space_dims3D[1] = (Ny + 2);
	mem_space_dims3D[2] = SYS_DIM;

	// Write the real space vorticity
	WriteDataReal(t, (int)iters, main_group_id, "u", H5T_NATIVE_DOUBLE, d_set_rank3D, dset_dims3D, slab_dims3D, mem_space_dims3D, sys_vars->local_Nx_start, run_data->u);
	#endif

	///--------------------------------------- Fourier Space Nonlinear Term
	#if defined(__NONLIN)
	// Create dimension arrays for the Fourier nonlinear term
	dset_dims2D[0] 	  = Nx;
	dset_dims2D[1] 	  = Ny_Fourier;
	slab_dims2D[0] 	  = sys_vars->local_Nx;
	slab_dims2D[1] 	  = Ny_Fourier;
	mem_space_dims2D[0] = sys_vars->local_Nx;
	mem_space_dims2D[1] = Ny_Fourier;

	// Write the Fourier nonlinear term
	WriteDataFourier(t, (int)iters, main_group_id, "NonlinearTerm", file_info->COMPLEX_DTYPE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->nonlinterm);
	#endif

	///--------------------------------------- Foureir Phases & Amplitudes of the Vorticity
	#if defined(PHASE_ONLY)
	// Record the Fourier amplitudes and phases
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Record the amplitudes
			run_data->a_k[indx] = cabs(run_data->w_hat[indx]);
			// Record the phases 
			run_data->phi_k[indx] = carg(run_data->w_hat[indx]);
		}
	}

	// Create dimension arrays for the Fourier amplitudes and phases
	dset_dims2D[0] 	  = Nx;
	dset_dims2D[1] 	  = Ny_Fourier;
	slab_dims2D[0] 	  = sys_vars->local_Nx;
	slab_dims2D[1] 	  = Ny_Fourier;
	mem_space_dims2D[0] = sys_vars->local_Nx;
	mem_space_dims2D[1] = Ny_Fourier;

	// Write the Fourier amplitudes
	WriteDataReal(t, (int)iters, main_group_id, "a_k", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->a_k);

	// Write the Fourier phases
	WriteDataReal(t, (int)iters, main_group_id, "phi_k", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->phi_k);
	#endif

	///--------------------------------------- Taylor Green Exact Solution
	#if defined(TESTING)
	if (!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
		// Create dimension arrays
		dset_dims2D[0] 	    = Nx;
		dset_dims2D[1] 	    = Ny;
		slab_dims2D[0]      = sys_vars->local_Nx;
		slab_dims2D[1]      = Ny;
		mem_space_dims2D[0] = sys_vars->local_Nx;
		mem_space_dims2D[1] = Ny + 2;

		// Write the real space vorticity
		WriteDataReal(t, (int)iters, main_group_id, "TGSoln", H5T_NATIVE_DOUBLE, d_set_rank2D, dset_dims2D, slab_dims2D, mem_space_dims2D, sys_vars->local_Nx_start, run_data->tg_soln);	
	}
	#endif

	///--------------------------------------- Spectra and Phase Sync
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT) || defined(__PHASE_SYNC)
	// Gather Spectra data and write to file
	if (!sys_vars->rank) {
		// Gather spectra data on master process & write to file
		#if defined(__ENST_SPECT)
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "EnstrophySpectrum", run_data->enst_spect);
		#endif 
		#if defined(__ENRG_SPECT)
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "EnergySpectrum", run_data->enrg_spect);
		#endif
		#if defined(__ENST_FLUX_SPECT)
		MPI_Reduce(MPI_IN_PLACE, run_data->d_enst_dt_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_diss_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_flux_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "TimeDerivativeEnstrophySpectrum", run_data->d_enst_dt_spect);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "EnstrophyDissipationSpectrum", run_data->enst_diss_spect);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "EnstrophyFluxSpectrum", run_data->enst_flux_spect);
		#endif
		#if defined(__ENRG_FLUX_SPECT)
		MPI_Reduce(MPI_IN_PLACE, run_data->d_enrg_dt_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_diss_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_flux_spect, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "TimeDerivativeEnergySpectrum", run_data->d_enrg_dt_spect);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "EnergyDissipationSpectrum", run_data->enrg_diss_spect);
		WriteDataSerial(t, iters, spectra_group_id, sys_vars->n_spect, "EnergyFluxSpectrum", run_data->enrg_flux_spect);
		#endif
		#if defined(__PHASE_SYNC)
		MPI_Reduce(MPI_IN_PLACE, run_data->phase_order_k, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->normed_phase_order_k, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// Create temp mem
		double amp_k[sys_vars->n_spect];
		double phase_k[sys_vars->n_spect];
		double normed_amp_k[sys_vars->n_spect];
		double normed_phase_k[sys_vars->n_spect];
		for (int i = 0; i < sys_vars->n_spect; ++i) {
			amp_k[i]          = cabs(run_data->phase_order_k[i]);
			phase_k[i]        = carg(run_data->phase_order_k[i]);
			normed_amp_k[i]   = cabs(run_data->normed_phase_order_k[i]);
			normed_phase_k[i] = carg(run_data->normed_phase_order_k[i]);
		}
		WriteDataSerial(t, iters, sync_group_id, sys_vars->n_spect, "PhaseOrder_Theta_k", phase_k);
		WriteDataSerial(t, iters, sync_group_id, sys_vars->n_spect, "PhaseOrder_R_k", amp_k);
		WriteDataSerial(t, iters, sync_group_id, sys_vars->n_spect, "NormedPhaseOrder_Theta_k", normed_phase_k);
		WriteDataSerial(t, iters, sync_group_id, sys_vars->n_spect, "NormedPhaseOrder_R_k", normed_amp_k);
		#endif
	}
	else {
		// Reduce all other process to master rank
		#if defined(__ENST_SPECT)
		MPI_Reduce(run_data->enst_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENRG_SPECT)
		MPI_Reduce(run_data->enrg_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENST_FLUX_SPECT)
		MPI_Reduce(run_data->d_enst_dt_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enst_diss_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enst_flux_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENRG_FLUX_SPECT)
		MPI_Reduce(run_data->d_enrg_dt_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enrg_diss_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enrg_flux_spect, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__PHASE_SYNC)
		MPI_Reduce(run_data->phase_order_k, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->normed_phase_order_k, NULL, sys_vars->n_spect, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
	}
	#endif 


	// -------------------------------
	// Close identifiers and File
	// -------------------------------
	status = H5Gclose(main_group_id);
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
		exit(1);
	}
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		status = H5Gclose(spectra_group_id);
		status = H5Fclose(file_info->spectra_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, iters, t);
			exit(1);		
		}
	}
	#endif
	#if defined(__PHASE_SYNC)
	if (!sys_vars->rank) {
		status = H5Gclose(sync_group_id);
		status = H5Fclose(file_info->sync_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->sync_file_name, iters, t);
			exit(1);		
		}
	}
	#endif
}
/**
 * Wrapper function used to create a Group for the current iteration in the HDF5 file 
 * @param  group_name The name of the group - will be the Iteration counter
 * @param  t          The current time in the simulation
 * @param  dt         The current timestep being used
 * @param  iters      The current iteration counter
 * @return            Returns a hid_t identifier for the created group
 */
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters) {

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
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current time as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
		// Create attribute for the current timestep
		attr_id = H5Acreate(group_id, "TimeStep", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
		if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &dt)) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current timestep as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}


		// -------------------------------
		// Close the attribute identifiers
		// -------------------------------
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
		status = H5Sclose(attr_space);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
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
void WriteDataFourier(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, fftw_complex* data) {

	// Initialize variables
	hid_t plist_id;
	hid_t dset_space;
	hid_t file_space;
	hid_t mem_space;
	const int Dims = dset_rank;
	hsize_t dims[Dims];          // array to hold dims of the dataset to be created
	hsize_t mem_dims[Dims];	     // Array to hold the dimensions of the memory space - this will be diferent to slab dims for real data due to zero
	hsize_t mem_offset[Dims];    // Array to hold the offset in eahc direction for the local hypslabs to write from
	hsize_t slabsize[Dims];      // Array holding the size of the hyperslab in each direction
	hsize_t dset_offset[Dims];   // Array containig the offset positions in the file for each process to write to
	hsize_t dset_slabsize[Dims]; // Array containing the size of the slabbed that is being written to in file	

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	for (int i = 0; i < dset_rank; ++i) {
		dims[i] = dset_dims[i];
	}
	dset_space = H5Screate_simple(Dims, dims, NULL); 
	if (dset_space < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set dataspace for dataset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, dtype, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// -------------------------------
	// Select Hyperslab in Memory
	// -------------------------------
	// Setup for memory hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		slabsize[i]   = slab_dims[i];
		mem_offset[i] = 0;
		mem_dims[i]   = mem_space_dims[i];
	}
	
	// Create the memory space for the hyperslabs for each process
	mem_space = H5Screate_simple(Dims, mem_dims, NULL);

	// Select local hyperslab from the memoryspace (slab size adjusted to ignore 0 padding) - local to each process
	if ((H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_offset, NULL, slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- unable to select local hyperslab for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Select Hyperslab in File
	// -------------------------------
	// Setup for file hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		dset_offset[i]   = 0;
		dset_slabsize[i] = slab_dims[i];
	}
	dset_offset[0]   = offset_Nx;

	// Select the hyperslab in the dataset on file to write to
	if ((H5Sselect_hyperslab(dset_space, H5S_SELECT_SET, dset_offset, NULL, dset_slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to select hyperslab in file for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// --------------------------------------
	// Set Up Collective Write & Write Data
	// --------------------------------------
	// Set up Collective write property
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	// Write data to file
	if ((H5Dwrite(file_space, dtype, mem_space, dset_space, plist_id, data)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write data to datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
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
void WriteDataReal(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, double* data) {

	// Initialize variables
	hid_t plist_id;
	hid_t dset_space;
	hid_t file_space;
	hid_t mem_space;
	const int Dims = dset_rank;
	hsize_t dims[Dims];          // array to hold dims of the dataset to be created
	hsize_t mem_dims[Dims];	     // Array to hold the dimensions of the memory space - this will be diferent to slab dims for real data due to zero
	hsize_t mem_offset[Dims];    // Array to hold the offset in each direction for the local hypslabs to write from
	hsize_t slabsize[Dims];      // Array holding the size of the hyperslab in each direction
	hsize_t dset_offset[Dims];   // Array containig the offset positions in the file for each process to write to
	hsize_t dset_slabsize[Dims]; // Array containing the size of the slabs that is being written to in file	

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	for (int i = 0; i < dset_rank; ++i) {
		dims[i] = dset_dims[i];
	}
	dset_space = H5Screate_simple(Dims, dims, NULL); 
	if (dset_space < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set dataspace for dataset: ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}	

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, H5T_NATIVE_DOUBLE, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// -------------------------------
	// Select Hyperslab in Memory
	// -------------------------------
	// Setup for memory hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		slabsize[i]   = slab_dims[i];
		mem_offset[i] = 0;
		mem_dims[i]   = mem_space_dims[i];
	}
	
	// Create the memory space for the hyperslabs for each process - reset second dimension for hyperslab selection to ignore padding
	mem_space = H5Screate_simple(Dims, mem_dims, NULL);

	// Select local hyperslab from the memoryspace (slab size adjusted to ignore 0 padding) - local to each process
	if ((H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_offset, NULL, slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- unable to select local hyperslab for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Select Hyperslab in File
	// -------------------------------
	// Set up file hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		dset_offset[i]   = 0;
		dset_slabsize[i] = slab_dims[i];
	}
	dset_offset[0]   = offset_Nx;

	// Select the hyperslab in the dataset on file to write to
	if ((H5Sselect_hyperslab(dset_space, H5S_SELECT_SET, dset_offset, NULL, dset_slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to select hyperslab in file for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// --------------------------------------
	// Set Up Collective Write & Write Data
	// --------------------------------------
	// Set up Collective write property
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	// Write data to file
	if ((H5Dwrite(file_space, dtype, mem_space, dset_space, plist_id, data)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write data to datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
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
 * Function to write the spectra & sync data for the current iteration to file
 * @param t         The current time in the system
 * @param iters     The current iteration in the system
 * @param group_id  The current group in the spectra file to write to
 * @param dims      The size of the dataset that is being written
 * @param dset_name The name of the dataset that is being written
 * @param data      The data that is to be written to file
 */
void WriteDataSerial(double t, int iters, hid_t group_id, int dims, char* dset_name, double* data) {

	// Initialize Variables
	hid_t dset_space;
	hid_t file_space;
	static const hsize_t Dims1D = 1;
	hsize_t dims1d[Dims1D];        // array to hold dims of the dataset to be created

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	dims1d[0] = dims;
	dset_space = H5Screate_simple(Dims1D, dims1d, NULL);
	if (dset_space < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set dataspace for dataset: ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, H5T_NATIVE_DOUBLE, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// --------------------------------------
	// Write Data to File Space
	// --------------------------------------
	if ((H5Dwrite(file_space, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write data to datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Close identifiers
	// -------------------------------
	H5Dclose(file_space);
	H5Sclose(dset_space);
}
/**
 * Wrapper function that writes all the non-slabbed/chunk datasets to file after integeration has finished - to do so the file must be reponed 
 * with the right read/write permissions and normal I/0 access properties -> otherwise writing to file in a non MPI way would not work
 * @param N 			 Array containing the dimensions of the system
 * @param iters 	     The number of iterations performed by the simulation 
 * @param save_data_indx The number of saving steps performed by the simulation
 */
void FinalWriteAndCloseOutputFiles(const long int* N, int iters, int save_data_indx) {

	// Initialize Variables
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	herr_t status;
	static const hsize_t D1 = 1;
	hsize_t dims1D[D1];
	#if defined(__STATS) && (defined(__VEL_STR_FUNC) || defined(__VORT_STR_FUNC))
	static const hsize_t D3 = 3;
	hsize_t dims3D[D3];
	#endif

	// Record total iterations
	sys_vars->tot_iters      = (long int)iters - 1;
	sys_vars->tot_save_steps = (long int)save_data_indx - 1;

	////////////////////////////////
	/// Create Output files 
	////////////////////////////////
	// Repon Output file with read/write permissions
	if (!(sys_vars->rank)) {
		file_info->sys_msr_file_handle = H5Fcreate(file_info->sys_msr_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->sys_msr_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create system measures file for writing non chunked/slabbed datasets! \n-->>Exiting....\n");
			exit(1);
		}
	}

	// Create stats file
	#if defined(__STATS)
	if (!sys_vars->rank){
		// Create the Stats sync output file
		file_info->stats_file_handle = H5Fcreate(file_info->stats_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->stats_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 spectra output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->stats_file_name);
			exit(1);
		}
	}
	#endif

	////////////////////////////////
	/// Write Datasets 
	////////////////////////////////
	// -------------------------------
	// Write Wavenumbers
	// -------------------------------
	#if defined(__WAVELIST)
	// Allocate array to gather the wavenumbers from each of the local arrays - in the x direction
	int* k0 = (int* )fftw_malloc(sizeof(int) * Nx);
	MPI_Gather(run_data->k[0], sys_vars->local_Nx, MPI_INT, k0, sys_vars->local_Nx, MPI_INT, 0, MPI_COMM_WORLD); 

	// Write to file
	if (!(sys_vars->rank)) {
		dims1D[0] = Nx;
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "kx", D1, dims1D, H5T_NATIVE_INT, k0)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "kx");
		}
		dims1D[0] = Ny_Fourier;
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "ky", D1, dims1D, H5T_NATIVE_INT, run_data->k[1])) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "ky");
		}
	}
	fftw_free(k0);
	#endif

	// -------------------------------
	// Write Collocation Points
	// -------------------------------
	#if defined(__COLLOC_PTS)
	// Allocate array to gather the collocation points from each of the local arrays
	double* x0 = (double* )fftw_malloc(sizeof(double) * Nx);
	MPI_Gather(run_data->x[0], sys_vars->local_Nx, MPI_DOUBLE, x0, sys_vars->local_Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

	// Write to file
	if (!(sys_vars->rank)) {
		dims1D[0] = Nx;
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "x", D1, dims1D, H5T_NATIVE_DOUBLE, x0)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "x");
		}
		dims1D[0] = Ny;
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "y", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->x[1]))< 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "y");
		}
	}
	fftw_free(x0);
	#endif

	// -------------------------------
	// Write System Measures
	// -------------------------------
	//------------- Time
	#if defined(__TIME)
	// Time array only on rank 0
	if (!(sys_vars->rank)) {
		dims1D[0] = sys_vars->num_print_steps;
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "Time", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->time)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "Time");
		}
	}
	#endif

	#if defined(__SYS_MEASURES)
	//-------------- Write mean flows
	// Mean flow in the y direction -> \bar{v}(x)
	double* tmp = (double* )fftw_malloc(sizeof(double) * Nx);
	MPI_Gather(run_data->mean_flow_y, sys_vars->local_Nx, MPI_DOUBLE, tmp, sys_vars->local_Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (int i = 0; i < Nx; ++i) {
		tmp[i] /= sys_vars->num_sys_msr_counts;
	}
	dims1D[0] = Nx;
	if (!sys_vars->rank) {
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "MeanFlow_y", D1, dims1D, H5T_NATIVE_DOUBLE, tmp)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "MeanFlow_y");
		}
	}
	fftw_free(tmp);

	// Mean flow in the x direction -> \bar{u}(y)	
	for (int i = 0; i < Ny_Fourier; ++i) {
		run_data->mean_flow_x[i] /= sys_vars->num_sys_msr_counts;
	}
	dims1D[0] = Ny_Fourier;
	if (!sys_vars->rank) {
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "MeanFlow_x", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->mean_flow_x)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "MeanFlow_x");
		}
	}
	#endif

	// -------------------------------
	// Prepare Stats for Writing
	// -------------------------------
	#if defined(__STATS)
	// Get max scales
	int num_r_scales = (int) GSL_MIN(Nx, Ny) / 2;

	///------------------- Structure Functions
	#if defined(__VEL_STR_FUNC)
	// Allocate temporary memory for structure funcitons
	double* tmp_vel_str_functions = (double*)fftw_malloc(sizeof(double ) * INCR_TYPES * NUM_POW * num_r_scales);

	// Write record local (to each process) strucure functions to tmp memory
	for (int p = 0; p < NUM_POW; ++p) {
		for (int r_scale = 0; r_scale < num_r_scales; ++r_scale) {
			for (int type = 0; type < INCR_TYPES; ++type) {
				tmp_vel_str_functions[INCR_TYPES * (p * num_r_scales + r_scale) + type] = stats_data->vel_str_func[type][p][r_scale] / stats_data->num_stats_steps;
			}
		}
	}
	#endif
	#if defined(__VORT_STR_FUNC)
	// Allocate temporary memory for structure funcitons
	double* tmp_vort_str_functions = (double*)fftw_malloc(sizeof(double ) * INCR_TYPES * NUM_POW * num_r_scales);

	// Write record local (to each process) strucure functions to tmp memory
	for (int p = 0; p < NUM_POW; ++p) {
		for (int r_scale = 0; r_scale < num_r_scales; ++r_scale) {
			for (int type = 0; type < INCR_TYPES; ++type) {
				tmp_vort_str_functions[INCR_TYPES * (p * num_r_scales + r_scale) + type] = stats_data->vort_str_func[type][p][r_scale] / stats_data->num_stats_steps;
			}
		}
	}
	#endif

	///------------------- Increment Histograms
	#if defined(__VEL_INC)
	double* vel_inc_counts = (double* )fftw_malloc(sizeof(double) * INCR_TYPES * NUM_INCR * VEL_NUM_BINS);
	double* vel_inc_ranges = (double* )fftw_malloc(sizeof(double) * INCR_TYPES * NUM_INCR * (VEL_NUM_BINS + 1));
	for (int num_r = 0; num_r < NUM_INCR; ++num_r) {
		for (int n = 0; n < VEL_NUM_BINS + 1; ++n) {
			for (int type = 0; type < INCR_TYPES; ++type) {
				if (n < VEL_NUM_BINS) {
					vel_inc_counts[INCR_TYPES * (num_r * VEL_NUM_BINS + n) + type] += stats_data->vel_inc_hist[type][num_r]->bin[n];
				}
				vel_inc_ranges[INCR_TYPES * (num_r * (VEL_NUM_BINS + 1) + n) + type] += stats_data->vel_inc_hist[type][num_r]->range[n];
			}
		}
	}
	#endif
	#if defined(__VORT_INC)
	double* vort_inc_counts = (double* )fftw_malloc(sizeof(double) * INCR_TYPES * NUM_INCR * VORT_NUM_BINS);
	double* vort_inc_ranges = (double* )fftw_malloc(sizeof(double) * INCR_TYPES * NUM_INCR * (VORT_NUM_BINS + 1));
	for (int num_r = 0; num_r < NUM_INCR; ++num_r) {
		for (int n = 0; n < VORT_NUM_BINS + 1; ++n) {
			for (int type = 0; type < INCR_TYPES; ++type) {
				if (n < VORT_NUM_BINS) {
					vort_inc_counts[INCR_TYPES * (num_r * VORT_NUM_BINS + n) + type] += stats_data->vort_inc_hist[type][num_r]->bin[n];
				}
				vort_inc_ranges[INCR_TYPES * (num_r * (VORT_NUM_BINS + 1) + n) + type] += stats_data->vort_inc_hist[type][num_r]->range[n];
			}
		}
	}
	#endif
	#endif
	// ----------------------------------------
	// Reduce & Write System Measures & Stats
	// ----------------------------------------
	//--------------- System measures & Stats --> need to reduce (in place on rank 0) all arrays across the processess
	if (!(sys_vars->rank)) {
		// Reduce on to rank 0
		#if defined(__SYS_MEASURES)
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_energy, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_enstr, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_palin, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_forc, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_div, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_diss, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_diss, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENST_FLUX)
		MPI_Reduce(MPI_IN_PLACE, run_data->d_enst_dt_sbst, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_flux_sbst, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_flux_sbst, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENRG_FLUX)
		MPI_Reduce(MPI_IN_PLACE, run_data->d_enrg_dt_sbst, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_flux_sbst, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_flux_sbst, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__STATS)
		///------------------ Structure functions
		#if defined(__VEL_STR_FUNC)
		MPI_Reduce(MPI_IN_PLACE, tmp_vel_str_functions, (int)(INCR_TYPES * NUM_POW * num_r_scales), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__VORT_STR_FUNC)
		MPI_Reduce(MPI_IN_PLACE, tmp_vort_str_functions, (int)(INCR_TYPES * NUM_POW * num_r_scales), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__MIXED_VORT_STR_FUNC)
		MPI_Reduce(MPI_IN_PLACE, stats_data->vort_mixed_str_func, (int)(num_r_scales), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		for (int i = 0; i < num_r_scales; ++i) {
			stats_data->vort_mixed_str_func[i] /= stats_data->num_stats_steps;
		}
		#endif
		#if defined(__MIXED_VEL_STR_FUNC)
		MPI_Reduce(MPI_IN_PLACE, stats_data->vel_mixed_str_func, (int)(num_r_scales), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		for (int i = 0; i < num_r_scales; ++i) {
			stats_data->vort_mixed_str_func[i] /= stats_data->num_stats_steps;
		}
		#endif
		///---------------- Increment Histograms
		#if defined(__VEL_INC)
		MPI_Reduce(MPI_IN_PLACE, vel_inc_counts, (int)(INCR_TYPES * NUM_INCR * VEL_NUM_BINS), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__VORT_INC)
		MPI_Reduce(MPI_IN_PLACE, vort_inc_counts, (int)(INCR_TYPES * NUM_INCR * VORT_NUM_BINS), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#endif

		// Dataset dims
		dims1D[0] = sys_vars->num_print_steps;

		#if defined(__SYS_MEASURES)
		// Energy
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TotalEnergy", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_energy)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TotalEnergy");
		}
		// Enstrophy
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TotalEnstrophy", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_enstr)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TotalEnstrophy");
		}
		// Palinstrophy
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TotalPalinstrophy", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_palin)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TotalPalinstrophy");
		}
		// Forcing Input
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TotalForcing", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_forc)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TotalForcing");
		}
		// Divergence
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TotalDivergence", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->tot_div)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TotalDivergence");
		}
		// Energy dissipation rate
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "EnergyDissipation", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->enrg_diss)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "EnergyDissipation");
		}
		// Enstrophy dissipation rate
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "EnstrophyDissipation", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->enst_diss)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "EnstrophyDissipation");
		}
		#endif
		#if defined(__ENST_FLUX)
		// Time deriviative of enstorphy of a subset of modes
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TimeDerivativeEnstrophySubset", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->d_enst_dt_sbst)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TimeDerivativeEnstrophySubset");
		}
		// Enstrophy flux in/out of a subset of modes
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "EnstrophyFluxSubset", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->enst_flux_sbst)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "EnstrophyFluxSubset");
		}
		// Enstrophy dissipation of a subset of modes
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "EnstrophyDissSubset", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->enst_diss_sbst)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "EnstrophyDissSubset");
		}
		#endif
		#if defined(__ENRG_FLUX)
		// Time derivative of energy of a subset of modes
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "TimeDerivativeEnergySubset", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->d_enrg_dt_sbst)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "TimeDerivativeEnergySubset");
		}
		// Energy flux in/out of a subset of modes
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "EnergyFluxSubset", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->enrg_flux_sbst)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "EnergyFluxSubset");
		}
		// Energy dissipation of a subset of modes
		if ( (H5LTmake_dataset(file_info->sys_msr_file_handle, "EnergyDissSubset", D1, dims1D, H5T_NATIVE_DOUBLE, run_data->enrg_diss_sbst)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "EnergyDissSubset");
		}
		#endif
		#if defined(__STATS)
		#if defined(__MIXED_VEL_STR_FUNC) 
		// Define the dimensions of the mixed velocity structure funciton array
		dims1D[0] = num_r_scales;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "MixedVelocityStructureFunctions", D1, dims1D, H5T_NATIVE_DOUBLE, stats_data->vel_mixed_str_func)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "MixedVelocityStructureFunctions");
		}
		#endif
		#if defined(__MIXED_VEL_STR_FUNC) 
		// Define the dimensions of the mixed velocity structure funciton array
		dims1D[0] = num_r_scales;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "MixedVorticityStructureFunctions", D1, dims1D, H5T_NATIVE_DOUBLE, stats_data->vort_mixed_str_func)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "MixedVorticityStructureFunctions");
		}
		#endif
		#if defined(__VEL_STR_FUNC) 
		// Define the dimensions of the velocity structure funciton array
		dims3D[0] = NUM_POW;
		dims3D[1] = num_r_scales;
		dims3D[2] = INCR_TYPES;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "VelocityStructureFunctions", D3, dims3D, H5T_NATIVE_DOUBLE, tmp_vel_str_functions)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "VelocityStructureFunctions");
		}
		fftw_free(tmp_vel_str_functions);
		#endif
		#if defined(__VORT_STR_FUNC) 
		// Define the dimensions of the vorticity structure funciton array
		dims3D[0] = NUM_POW;
		dims3D[1] = num_r_scales;
		dims3D[2] = INCR_TYPES;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "VorticityStructureFunctions", D3, dims3D, H5T_NATIVE_DOUBLE, tmp_vort_str_functions)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "VorticityStructureFunctions");
		}
		fftw_free(tmp_vort_str_functions);
		#endif
		#if defined(__VEL_INC) 
		// Define the dimensions of the velocity increment arrays
		dims3D[0] = NUM_INCR;
		dims3D[1] = VEL_NUM_BINS;
		dims3D[2] = INCR_TYPES;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "VelocityIncrementHist_Counts", D3, dims3D, H5T_NATIVE_DOUBLE, vel_inc_counts)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "VelocityIncrementHist_Counts");
		}
		dims3D[1] = VEL_NUM_BINS + 1;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "VelocityIncrementHist_Ranges", D3, dims3D, H5T_NATIVE_DOUBLE, vel_inc_ranges)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "VelocityIncrementHist_Ranges");
		}
		fftw_free(vel_inc_counts);
		fftw_free(vel_inc_ranges);
		#endif
		#if defined(__VORT_INC) 
		// Define the dimensions of the vorticity increment arrays
		dims3D[0] = NUM_INCR;
		dims3D[1] = VORT_NUM_BINS;
		dims3D[2] = INCR_TYPES;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "VorticityIncrementHist_Counts", D3, dims3D, H5T_NATIVE_DOUBLE, vort_inc_counts)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "VorticityIncrementHist_Counts");
		}
		dims3D[1] = VORT_NUM_BINS + 1;
		if ( (H5LTmake_dataset(file_info->stats_file_handle, "VorticityIncrementHist_Ranges", D3, dims3D, H5T_NATIVE_DOUBLE, vort_inc_ranges)) < 0) {
			printf("\n["MAGENTA"WARNING"RESET"] --- Failed to make dataset ["CYAN"%s"RESET"]\n", "VorticityIncrementHist_Ranges");
		}
		fftw_free(vort_inc_counts);
		fftw_free(vort_inc_ranges);
		#endif
		#endif
	}
	else {
		// Reduce all other process to rank 0
		#if defined(__SYS_MEASURES)
		MPI_Reduce(run_data->tot_energy, NULL,  sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_enstr, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_palin, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_forc, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_div, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enrg_diss, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enst_diss, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENST_FLUX)
		MPI_Reduce(run_data->d_enst_dt_sbst, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enst_flux_sbst, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enst_diss_sbst, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
		#if defined(__ENRG_FLUX)
		MPI_Reduce(run_data->d_enrg_dt_sbst, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enrg_flux_sbst, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enrg_diss_sbst, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		#endif
	}

	// -----------------------------------
	// Close Files for the final time
	// -----------------------------------
	if (!(sys_vars->rank)) {
		status = H5Fclose(file_info->sys_msr_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close system measures file: ["CYAN"%s"RESET"] \n-->> Exiting....\n", file_info->sys_msr_file_name);
			exit(1);
		}
	}
	#if defined(__PHASE_SYNC)
	if (!sys_vars->rank) {
		status = H5Fclose(file_info->stats_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close stats output file ["CYAN"%s"RESET"] \n-->> Exiting...\n", file_info->stats_file_name);
			exit(1);		
		}
	}
	#endif
	#if defined(DEBUG)
	if (!sys_vars->rank) {
		// Close test / debug file
	    H5Fclose(file_info->test_file_handle);
	}
	#endif
	#if defined(__VORT_FOUR) || defined(__MODES) || defined(__NONLIN)
	// Close the complex datatype identifier
	H5Tclose(file_info->COMPLEX_DTYPE);
	#endif
}
/**
 * Function to create and open the file for writinig test data to
 */
void OpenTestingFile(void) {

	// Initialzie variables
	char file_data[512];

	// Get filename
	sprintf(file_data, "Test_Data_N[%ld,%ld]_u0[%s].h5", sys_vars->N[0], sys_vars->N[1], sys_vars->u0);

	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();

	// ----------------------------------
	// Construct File Paths
	// ---------------------------------- 
	// Construct main file path
	strcpy(file_info->test_file_name, file_info->output_dir);
	strcat(file_info->test_file_name, file_data); 

	// ----------------------------------
	// Create file
	// ---------------------------------- 
	if (!sys_vars->rank) {
        file_info->test_file_handle = H5Fcreate(file_info->test_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_info->test_file_handle < 0) {
        	fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create test file: "CYAN"%s"RESET" \n-->> Exiting....\n", file_info->test_file_name);
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
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------