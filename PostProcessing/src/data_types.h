/**
* @file data_types.h 
* @author Enda Carroll
* @date Jun 2021
* @brief file containing the main data types and global variables
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#ifndef __DATA_TYPES

#ifndef __HDF5_HDR
#include <hdf5.h>
#include <hdf5_hl.h>
#define __HDF5_HDR
#endif
#ifndef __FFTW3
#include <fftw3-mpi.h>
#define __FFTW3
#endif

// ---------------------------------------------------------------------
//  Compile Time Macros and Definitions
// ---------------------------------------------------------------------
#define checkError(x) ({int __val = (x); __val == -1 ? \
	({fprintf(stderr, "ERROR ("__FILE__":%d) -- %s\n", __LINE__, strerror(errno)); \
	exit(-1);-1;}) : __val; })

// For coloured printing to screen
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define RESET   "\x1b[0m"
// ---------------------------------------------------------------------
//  Datasets to Write to File
// ---------------------------------------------------------------------
// These definitions control which datasets are to be computed and written to file
// Turning these on in this file means that they WILL be on at compilation time

// Choose whether to save the Real Space or Fourier Space vorticity
// #define __VORT_REAL
#define __VORT_FOUR
// Choose whether to save the Real or Fourier space velocitites
// #define __MODES
// #define __REALSPACE
// Choose whether to compute the Energy and Enstrophy spectra and flux spectra
#define __SPECT
// Choose whether to save the time, collocation points and wavenumbers
#define __TIME
#define __COLLOC_PTS
#define __WAVELIST
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// These definitions define some of the solver parameters.
#define SYS_DIM 2 				// The system dimension i.e., 2D
// ---------------------------------------------------------------------
//  Global Struct Definitions
// ---------------------------------------------------------------------
// System variables struct
typedef struct system_vars_struct {
	char u0[64];						// String to indicate the initial condition to use
	char forcing[64];					// String to indicate what type of forcing is selected
	long int N[SYS_DIM];				// Array holding the no. of collocation pts in each dim
	fftw_plan fftw_2d_dft_r2c;			// FFTW plan to perform transform from Real to Fourier
	fftw_plan fftw_2d_dft_c2r;			// FFTW plan to perform transform from Fourier to Real
	fftw_plan fftw_2d_dft_batch_r2c;	// FFTW plan to perform a batch transform from Real to Fourier
	fftw_plan fftw_2d_dft_batch_c2r;	// FFTW plan to perform a batch transform from Fourier to Real
	ptrdiff_t alloc_local;				// Variable to hold size of memory to allocate for local (on process) arrays for normal transform
	ptrdiff_t alloc_local_batch;		// Variable to hold size of memory to allocate for local (on process) arrays for batch transform
	ptrdiff_t local_Nx;					// Size of the first dimension for the local arrays
	ptrdiff_t local_Nx_start;			// Position where the local arrays start in the undistributed array
	int num_procs;						// Variable to hold the number of active provcesses
	int rank;							// Rank of the active processes
	long int num_t_steps;				// Number of iteration steps to perform
	long int num_print_steps;           // Number of times system was saved to file
	long int tot_iters;					// Records the total executed iterations
	long int tot_save_steps;			// Records the total saving iterations
	double t0;							// Intial time
	double T;							// Final time
	double t;							// Time variable
	double dt;							// Timestep
	double min_dt;						// Smallest timestep achieved when adaptive stepping is on
	double max_dt;						// Largest timestep achieved when adaptive stepping is on
	double dx;							// Collocation point spaceing in the x direction
	double dy;							// Collocation point spacing in the y direction
	double w_max_init;					// Max vorticity of the initial condition
	int n_spect;                        // Size of the spectra arrays
	int force_k; 						// The forcing wavenumber 
	int print_every;                    // Records how many iterations are performed before printing to file
	double EKMN_ALPHA; 					// The value of the Ekman drag coefficient
	double CFL_CONST;					// The CFL constant for the adaptive step
	double NU;							// The viscosity
	int SAVE_EVERY; 					// For specifying how often to print
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data_struct {
	double* x[SYS_DIM];      // Array to hold collocation pts
	int* k[SYS_DIM];		 // Array to hold wavenumbers
	fftw_complex* w_hat;     // Fourier space vorticity
	fftw_complex* u_hat;     // Fourier space velocity
	double* w;				 // Real space vorticity
	double* u;				 // Real space velocity
	double* time;			  // Array to hold the simulation times
} runtime_data_struct;

// HDF5 file info struct
typedef struct HDF_file_info_struct {
	char input_file_name[512];		// Array holding input file name
	char output_file_name[512];     // Output file name array
	char output_dir[512];			// Output directory
	char output_tag[64]; 			// Tag to be added to the output directory
	hid_t output_file_handle;		// Main file handle for the output file 
	hid_t spectra_file_handle;      // Spectra file handle
	hid_t COMPLEX_DTYPE;			// Complex datatype handle
	int file_only;					// Indicates if output should be file only with no output folder created
} HDF_file_info_struct;

// Complex datatype struct for HDF5
typedef struct complex_type_tmp {
	double re;   			 // real part 
	double im;   			 // imaginary part 
} complex_type_tmp;


// Declare the global variable pointers across all files
extern system_vars_struct *sys_vars; 		    // Global pointer to system parameters struct
extern runtime_data_struct *run_data; 			// Global pointer to system runtime variables struct 
extern HDF_file_info_struct *file_info; 		// Global pointer to system forcing variables struct 

#define __DATA_TYPES
#endif
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
