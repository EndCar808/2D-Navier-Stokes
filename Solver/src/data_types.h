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
//  Integration Functionality
// ---------------------------------------------------------------------
// #define __EULER
#define __NAVIER
#define __RK4
// #define __RK5
// #define __DPRK5
#define __ADAPTIVE_STEP
#define __CFL_STEP
#define __DEALIAS_23
// #define __DEALIAS_HOU_LI
#define __PRINT_SCREEN
#define SAVE_EVERY 100

#define __TESTING
#define TEST_PRINT 10

// ---------------------------------------------------------------------
//  Datasets to Write to File
// ---------------------------------------------------------------------
#define __VORT_REAL
#define __VORT_FOUR
// #define __MODES
// #define __REALSPACE
// #define __SPECT
#define __TIME
#define __COLLOC_PTS
#define __WAVELIST
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
#define SYS_DIM 2
#define NU 1.0
#define VIS_POW 1.0
#define EKMN_ALPHA 0.0
#define EKMN_POW 0.0

#define DP_ABS_TOL 1e-7
#define DP_REL_TOL 1e-7
#define DP_DELTA_MIN 0.01
#define DP_DELTA_MAX 1.5
#define DP_DELTA 0.8

#define CFL_CONST 1.73205080757 // = sqrt(3)

#define KAPPA 1.0
#define SIGMA 15.0 / M_PI
#define DELTA 0.005

#define MIN_STEP_SIZE 1e-10
#define MAX_ITERS 1e+6
#define MAX_VORT_LIM 1e+100
// ---------------------------------------------------------------------
//  Global Struct Definitions
// ---------------------------------------------------------------------
// System variables struct
typedef struct system_vars_struct {
	char* u0;							// String to indicate the initial condition to use
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
	double t0;							// Intial time
	double T;							// Final time
	double t;							// Time variable
	double dt;							// Timestep
	double dx;							// Collocation point spaceing in the x direction
	double dy;							// Collocation point spacing in the y direction
	double w_max_init;					// Max vorticity of the initial condition
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data_struct {
	double* x[SYS_DIM];      // Array to hold collocation pts
	int* k[SYS_DIM];		 // Array to hold wavenumbers
	fftw_complex* w_hat;     // Fourier space vorticity
	fftw_complex* u_hat;     // Fourier space velocity
	double* w;				 // Real space vorticity
	double* u;				 // Real space velocity
	double* tot_energy;      // Array to hold the total energy over the simulation
	double* tot_enstr;		 // Array to hold the total entrophy over the simulation
	double* tot_palin;		 // Array to hold the total palinstrophy over the simulaiotns
	double* time;			 // Array to hold the simulation times
	double* tg_soln;			 // Array for computing the Taylor Green vortex solution
} runtime_data_struct;

// Runge-Kutta Integration struct
typedef struct RK_data_struct {
	fftw_complex* RK1;		  // Array to hold the result of the first stage
	fftw_complex* RK2;		  // Array to hold the result of the second stage
	fftw_complex* RK3;		  // Array to hold the result of the third stage
	fftw_complex* RK4;		  // Array to hold the result of the fourth stage
	fftw_complex* RK5;		  // Array to hold the result of the fifth stage of RK5 scheme
	fftw_complex* RK6;		  // Array to hold the result of the sixth stage of RK5 scheme
	fftw_complex* RK7; 		  // Array to hold the result of the seventh stage of the Dormand Prince Scheme
	fftw_complex* RK_tmp;	  // Array to hold the tempory updates to w_hat - input to RHS function
	fftw_complex* w_hat_last; // Array to hold the values of the Fourier space vorticity from the previous iteration - used in the stepsize control in DP scheme
	double* nabla_psi;		  // Batch array the velocities u = d\psi_dy and v = -d\psi_dx
	double* nabla_w;		  // Batch array to hold \nabla\omega - the vorticity derivatives
	double DP_err; 			  // Variable to hold the error between the embedded methods in the Dormand Prince scheme
	int DP_fails;
} RK_data_struct;

// HDF5 file info struct
typedef struct HDF_file_info_struct {
	char input_file_name[512];		// Array holding input file name
	char output_file_name[512];     // Output file name array
	char output_dir[512];			// Output directory
	hid_t output_file_handle;		// File handle for the output file 
	hid_t COMPLEX_DTYPE;			// Complex datatype handle
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
