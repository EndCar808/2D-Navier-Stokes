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
	
#include <gsl/gsl_histogram.h> 
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_math.h>

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
// These definitions control the functionality of the solver. These definitions
// are also passed at compilation for more control of the solver. Definitions turned
// on in this file means that they WILL be turned on no matter what is passed at compilation

// Choose which system to solver 
// #define __EULER
// #define __NAVIER
// Choose which integrator to use
// #define __RK4
// #define __RK4CN
// #define __AB4CN
// #define __RK5CN
// #define __DPRK5CN
// Choose which filter to use
#define __DEALIAS_23
// #define __DEALIAS_HOU_LI
// Choose whether to print updates to screen
#define __PRINT_SCREEN
// Adaptive stepping Indicators 
#define ADAPTIVE_STEP 1         // Indicator for the adaptive stepping
#define CFL_STEP 1 				// Indicator to use a CFL like condition with the adaptive stepping
// Solver Types
#define HYPER_VISC 1			// Turned on hyperviscosity if called for at compilation time
#define VISC_POW 2.0            // The power of the hyperviscosity -> 1.0 means no hyperviscosity
#define EKMN_DRAG 1   			// Turn on Ekman drag if called for at compilation time
#define EKMN_POW -2.0 			// The power of the Eckman drag term -> 0.0 means no drag
#define EKMN_DRAG_K 6 			// This is the shell wavenumber that divides the drag values for the system
// For allow transient dynamics
#define TRANSIENT_ITERS 1       // Indicator for transient iterations
#define TRANSIENT_FRAC 0.2      // Fraction of total iteration = transient iterations
// Allow for Phase Only mode via direct integration
#if defined(__PHASE_ONLY)		// Turn on phase only mode if called for at compilation
#define PHASE_ONLY
#endif
// Allow for Phase Only mode by fixing the amplitudes
#if defined(__PHASE_ONLY_FXD_AMP) // Turn on phase only mode if called for at compilation
#define PHASE_ONLY_FXD_AMP
#endif
// Testing the solver will be decided at compilation
#if defined(__TESTING)
#define TESTING
#endif
// If debugging / testing is called for at compile time
#if defined(__DEBUG)
#define DEBUG
#endif

// ---------------------------------------------------------------------
//  Datasets to Write to File
// ---------------------------------------------------------------------
// These definitions control which datasets are to be computed and written to file
// Turning these on in this file means that they WILL be on at compilation time

// Choose whether to save the Real Space or Fourier Space vorticity
// #define __VORT_REAL
// #define __VORT_FOUR
// Choose whether to save the Nonlinear term or RHS of equation of motion
// #define __RHS
// #define __NONLIN
// Choose whether to save the Real or Fourier space velocitites
// #define __MODES
// #define __REALSPACE
// Choose whether to compute system measures
// #define __SYS_MEASURES
// Choose whether to compute the fluxes
// #define __ENST_FLUX
// #define __ENRG_FLUX
// Choose whether to compute the Energy and Enstrophy spectra and flux spectra
// #define __ENST_SPECT
// #define __ENRG_SPECT
// #define __ENST_FLUX_SPECT
// #define __ENRG_FLUX_SPECT
// #define __ENST_SPECT_T_AVG
// #define __ENRG_SPECT_T_AVG
// #define __ENST_FLUX_SPECT_T_AVG
// #define __ENRG_FLUX_SPECT_T_AVG
// Choose whether to compute the phase sync data
// #define __PHASE_SYNC
// If Stats is called for at compile time
#if defined(__STATS)
#define __VEL_INC
#define __VORT_INC
#define __VEL_GRAD
#define __VORT_GRAD
#define __VEL_STR_FUNC
#define __VORT_STR_FUNC
#define __VEL_VORT_STR_FUNC
#define __MIXED_VEL_STR_FUNC
#define __MIXED_VORT_STR_FUNC
#endif
// Choose whether to save the time, collocation points and wavenumbers
#define __TIME
#define __COLLOC_PTS
#define __WAVELIST
#define __AMPS_T_AVG
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// These definitions define some of the solver parameters.
#define SYS_DIM 2 				// The system dimension i.e., 2D
#define TRANS_FRAC 0.2          // Fraction of time to ignore before saving to file
// Dormand Prince integrator parameters
#define DP_ABS_TOL 1e-7		    // The absolute error tolerance for the Dormand Prince Scheme
#define DP_REL_TOL 1e-7         // The relative error tolerance for the Dormand Prince Scheme
#define DP_DELTA_MIN 0.01       // The min delta value for the Dormand Prince scheme
#define DP_DELTA_MAX 1.5 		// The max delta value for the Dormand Prince scheme
#define DP_DELTA 0.8 			// The scaling parameter of the error for the Dormand Prince Scheme
// Initial Conditions parameters
#define KAPPA 1.0 				// The wavenumber of the Taylor Green initial condition 
#define SIGMA 15.0 / M_PI 		// The sigma term for the Double Shear Layer initial condition
#define DELTA 0.005	 			// The delta term for the Double Shear Layer initial condition
#define BETA 1.0                // The aspect ratio for the Gaussian blob initial condition
#define S 6.0                   // The scale parameter for the Gaussian Blob initial condition 
#define DT_K0 6.0				// The peak wavenumber for the McWilliams decaying vortex turblence initial condition
#define DT_E0 0.5               // The initial energy for the McWilliams decaying vortex turbulence initial condition
#define DT2_K0 30				// The peak wavenumber for the second McWilliams decaying turbulence initial condition
#define DT2_E0 0.5				// The spectrum normalizing constant for the second McWilliams decaying turbulence initial condition
#define DTEXP_K0 8.0			// The peak of the initial spectrum for the decaying turbulence exponential spectrum initial condition
#define DTEXP_E0 343.0/96.0		// The initial energy for the decaying turbulence exponential spectrum initial condition
#define GDT_K0 5.0              // The peak wavenumber for the Gaussian decay turbulence initial condition
#define GDT_C0 0.06             // The intial energy of the Gaussian decaying turbulence initial condition
#define RING_MIN_K 3.0			// The minimum absolute wavevector value to set the ring initial condition
#define RING_MAX_K 10.0			// The maximum absolute wavevector value to set the ring initial condition
#define EXTRM_ENS_MIN_K 1.5 	// The minimum absolute wavevector value for the Exponential Enstrophy initial condition
#define EXTRM_ENS_POW 1.5 		// The power for wavevectors for the Exponential enstrophy distribution
#define UNIF_MIN_K 2.5 			// The lower bound of sqrt(k) for initializing the vorticity with random phases
#define UNIF_MAX_K 9.5 			// The upper bound of sqrt(k) for initializing the vorticity with random phases
#define RAND_ENST0 1.0 			// The initial enstrophy for the random initial condition
#define PO_ENRG0 1.0 			// The initial energy for the phase only zero and random initial conditions
// Forcing parameters
#define STOC_FORC_GAUSS_K_MIN 2.5		// The minimum value of the modulus forced wavevectors for the stochasitc (Gaussian) forcing
#define STOC_FORC_GAUSS_K_MAX 6.5    	// The maximum value of the modulus forced wavevectors for the stochastic (Gaussian) forcing
#define STOC_FORC_UNIF_K 15	        	// The peak of the power spectrum of the forcing field for the stochastic forcing
#define STOC_FORC_UNIF_DELTA_K 1.5   	// The standard deviation of the power spectrum for the forcing field of the stochastic forcing 
#define CONST_GAUSS_K_MIN 10    		// The minimum value of the mod of forced wavevectors for the Constant Gaussian Ring forcing
#define CONST_GAUSS_K_MAX 12    		// The minimum value of the mod of forced wavevectors for the Constant Gaussian Ring forcing
// System checking parameters
#define MIN_STEP_SIZE 1e-10 	// The minimum allowed stepsize for the solver 
#define MAX_ITERS 1e+12			// The maximum iterations to perform
#define MAX_VORT_LIM 1e+100     // The maximum allowed vorticity
// Dynamic Modes
#define UPR_SBST_LIM 64         // The upper mode limit of the energy/enstrophy flux
#define LWR_SBST_LIM 0  		// The lower mode limit of the energy/enstrophy flux
// Stats parameters
#define NUM_POW 6 				// The highest moment to compute for the structure function
#define NUM_RUN_STATS 7 		// The number of running stats moments to record
#define VEL_BIN_LIM	40			// The bin limits (in units of standard deviations) for the velocity histogram
#define VEL_NUM_BINS 1000		// The number of bins to use for the velocity histograms
#define VORT_BIN_LIM 40			// The bin limits (in units of standard deviations) for the velocity histogram
#define VORT_NUM_BINS 1000		// The number of bins to use for the velocity histograms
#define NUM_INCR 2              // The number of increment length scales e.g. 2 = smallest and largest
#define INCR_TYPES 2 			// The number of increment directions i.e., longitudinal and transverse
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
	ptrdiff_t local_Ny;					// Size of the first dimension for the local arrays
	ptrdiff_t local_Ny_start;			// Position where the local arrays start in the undistributed array
	int num_procs;						// Variable to hold the number of active provcesses
	int rank;							// Rank of the active processes
	long int num_t_steps;				// Number of iteration steps to perform
	long int num_print_steps;           // Number of times system was saved to file
	long int tot_iters;					// Records the total executed iterations
	long int tot_save_steps;			// Records the total saving iterations
	long int trans_iters;				// The number of transients iterations to perform before printing to file
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
	int print_every;                    // Records how many iterations are performed before printing to file
	int SAVE_EVERY; 					// For specifying how often to print
	int STATS_EVERY; 					// For specifying how often to compute stats
	double force_scale_var;				// The scaling variable for the forced modes
	int force_k; 						// The forcing wavenumber 
	int local_forcing_proc;				// Identifier used to indicate which process contains modes that need to be forced
	int num_forced_modes; 				// The number of modes to be forced on the local process
	int TRANS_ITERS_FLAG;				// Flag for indicating whether transient iterations are to be performed
	double TRANS_ITERS_FRAC;			// The fraction of total iterations that will be ignored
	int ADAPT_STEP_FLAG;			 	// Flag for indicating if adaptive stepping is to be used
	double CFL_CONST;					// The CFL constant for the adaptive step
	int CFL_COND_FLAG;					// Flag for indicating if the CFL like condition is to be used for the adaptive stepping
	double NU;							// The viscosity
	int HYPER_VISC_FLAG;				// Flag to indicate if hyperviscosity is to be used
	double HYPER_VISC_POW;				// The power of the hyper viscosity to use
	double EKMN_ALPHA_LOW_K; 			// The value of the Ekman drag coefficient for low wavenumbers
	double EKMN_ALPHA_HIGH_K;			// The value of the Ekman drag coefficient for high wavenumbers
	int EKMN_DRAG_FLAG;					// Flag for indicating if ekman drag is to be used
	double EKMN_DRAG_POW;				// The power of the hyper drag to be used
	int num_sys_msr_counts; 			// Counter for counting the number of system measures is called for averaging
	double PO_SLOPE;					// Slope of the amplitudes for the phase only model
	int argc;							// The number of CML arguements used to call the sovler
	char** argv;						// The array of CML arguements used to call the solver
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data_struct {
	double* x[SYS_DIM];       			// Array to hold collocation pts
	int* k[SYS_DIM];		  			// Array to hold wavenumbers
	fftw_complex* w_hat;      			// Fourier space vorticity
	fftw_complex* w_hat_tmp;   			// Temporary Fourier space vorticity array for performing DFT to real space w
	fftw_complex* u_hat;      			// Fourier space velocity
	fftw_complex* u_hat_tmp;  			// Temporary Fourier space velocity array for performing DFT to real space u
	fftw_complex* rhs; 		  			// Array to hold the RHS of the equation of motion
	fftw_complex* nonlinterm; 			// Array to hold the nonlinear term
	double* w;				  			// Real space vorticity
	double* u;				  			// Real space velocity
	double* a_k;			  			// Fourier vorticity amplitudes
	double* a_k_t_avg;					// Time averaged Fourier vorticity amplitudes
	double* phi_k;			  			// Fourier vorticity phases
	double* tot_div;		  			// Array to hold the total diverence
	double* tot_forc;		  			// Array to hold the total forcing input into the sytem over the simulation
	double* tot_energy;       			// Array to hold the total energy over the simulation
	double* tot_enstr;		  			// Array to hold the total entrophy over the simulation
	double* tot_palin;		  			// Array to hold the total palinstrophy over the simulaiotns
	double* enrg_diss; 		  			// Array to hold the energy dissipation rate 
	double* enst_diss;		  			// Array to hold the enstrophy dissipation rate
	double* time;			  			// Array to hold the simulation times
	double* d_enst_dt_sbst;   			// Array to hold the time derivative of the enstrophy in a subset of modes
	double* enst_flux_sbst;   			// Array to hold the enstrophy flux in/out of a subset of modes
	double* enst_diss_sbst;   			// Array to hold the enstrophy dissipation for a subset of modes
	double* d_enrg_dt_sbst;   			// Array to hold the time derivative of the energy in a subset of modes
	double* enrg_flux_sbst;   			// Array to hold the energy flux in/out of a subset of modes
	double* enrg_diss_sbst;   			// Array to hold the energy dissipation for a subset of modes
	double* enrg_spect;		  			// Array to hold the energy spectrum of the system 
	double* enst_spect;       			// Array to hold the enstrophy spectrum of the system
	double* enrg_spect_t_avg;			// Array to hold the time averaged energy spectrum of the system 
	double* enst_spect_t_avg;  			// Array to hold the time averaged enstrophy spectrum of the system
	double* d_enst_dt_spect;  			// Array to hold the spectrum of the time derivative of the enstorphy
	double* enst_flux_spect;  			// Array to hold the spectrum of enstrophy flux of the system
	double* enst_flux_spect_t_avg;		// Array to hold the time averaged spectrum of enstrophy flux of the system
	double* enst_diss_spect;  			// Array to hold the spectrum enstrophy dissipation of the system
	double* d_enrg_dt_spect;  			// Array to hold the spectrum of the time derivative  of energy
	double* enrg_flux_spect;  			// Array to hold the energy flux spectrum
	double* enrg_flux_spect_t_avg;		// Array to hold the time averaged energy flux spectrum
	double* enrg_diss_spect;  			// Array to hold the energy dissiaption spectrum
	double* mean_flow_x;				// Array to hold the mean flow of the velocity field in the x direction
	double* mean_flow_y;				// Array to hold the mean flow of the velocity field in the x direction
	fftw_complex* phase_order_k;		// Array to hold the scale dependent collective phase
	fftw_complex* normed_phase_order_k;	// Array to hold the scale dependent collective phase
	double* tg_soln;	  	  			// Array for computing the Taylor Green vortex solution
	fftw_complex* forcing;	  			// Array to hold the forcing for the current timestep
	double* forcing_scaling;  			// Array to hold the initial scaling for the forced modes
	int* forcing_indx;		  			// Array to hold the indices of the forced modes
	int* forcing_k[SYS_DIM];  			// Array containg the wavenumbers for the forced modes
} runtime_data_struct;

// Runge-Kutta Integration struct
typedef struct Int_data_struct {
	fftw_complex* RK1;		  		// Array to hold the result of the first stage
	fftw_complex* RK2;		  		// Array to hold the result of the second stage
	fftw_complex* RK3;		  		// Array to hold the result of the third stage
	fftw_complex* RK4;		  		// Array to hold the result of the fourth stage
	fftw_complex* RK5;		  		// Array to hold the result of the fifth stage of RK5 scheme
	fftw_complex* RK6;		  		// Array to hold the result of the sixth stage of RK5 scheme
	fftw_complex* RK7; 		  		// Array to hold the result of the seventh stage of the Dormand Prince Scheme
	fftw_complex* RK_tmp;	  		// Array to hold the tempory updates to w_hat - input to RHS function
	fftw_complex* w_hat_last; 		// Array to hold the values of the Fourier space vorticity from the previous iteration - used in the stepsize control in DP scheme
	double* nabla_psi;		  		// Batch array the velocities u = d\psi_dy and v = -d\psi_dx
	double* nonlin;           		// Array to hold the result of the nonlinear term in real space
	double DP_err; 			  		// Variable to hold the error between the embedded methods in the Dormand Prince scheme
	int DP_fails;			  		// Counter to count the number of failed steps for the Dormand Prince scheme
	fftw_complex* AB_tmp;	  		// Array to hold the result of the RHS/Nonlinear term for the Adams Bashforth scheme
	fftw_complex* AB_tmp_nonlin[3];	// Array to hold the previous 3 RHS/Nonlinear terms to update the Adams Bashforth scheme
	int AB_pre_steps;				// The number of derivatives to perform for the AB4 scheme
} Int_data_struct;

// Field stats struct -> for computing field stats
typedef struct field_stats {
	double min;						// Minimum value of the field
	double max; 					// Maximum value of the field
	double data_sqrd_sum;			// The sum of squared data values
	double data_sum;				// The sum of data values
	double var;						// The variance of the data values
	double mean; 					// The mean of the data values
	double skew;					// The skewness
	double kurt;					// The kurtosis
	double std;						// The standard deviation
	long int n;						// The number of data values
} field_stats;

// Stats data struct
typedef struct stats_data_struct {
	double* vel_str_func[INCR_TYPES][NUM_POW];					// Array to hold the structure functions of the Velocity increments
	double* vel_str_func_abs[INCR_TYPES][NUM_POW];				// Array to hold the structure functions of the absolute value of flux Velocity increments
	double* vort_str_func[INCR_TYPES][NUM_POW];					// Array to hold the structure functions of the Vorticity increments
	double* vort_str_func_abs[INCR_TYPES][NUM_POW];				// Array to hold the structure functions of the absolute value of flux Vorticity increments
	double* vel_mixed_str_func;									// Array to hold the mixed velocity structure function, this should be the same as 1/3 of the third order velocity structre function
	double* vort_mixed_str_func;								// Array to hold the mixed vorticity structure function, this should be the same as the third order vorticity structre function
	struct field_stats vel_inc_stats[INCR_TYPES][NUM_INCR];     // Struct to hold the running stats for the velocity increments
	struct field_stats vort_inc_stats[INCR_TYPES][NUM_INCR];	// Struct to hold the running stats for the vorticity increments
	struct field_stats vel_stats[INCR_TYPES];					// Struct to hold the running stats for the velocity field
	struct field_stats vort_stats[INCR_TYPES];				    // Struct to hold the running stats for the vorticity field
	gsl_histogram* vel_inc_hist[INCR_TYPES][NUM_INCR];			// Struct to hold the histogram info for the velocity increments
	gsl_histogram* vort_inc_hist[INCR_TYPES][NUM_INCR];			// Struct to hold the histogram info for the vorticity increments
	long int num_stats_steps;									// Counter for the number of steps statistics have been computed
	int set_stats_flag;											// Flag to indicate if the stats objects such as running stats and histogram limits need to be set
	double vel_inc_sqr[INCR_TYPES][NUM_POW];					// To hold the sqr value of the velocity increments, this is for setting of the bin limits
	double vel_inc_num_data[INCR_TYPES][NUM_POW];				// To hold the number of velocity incrments, for the setting of bin limits
	double vort_inc_sqr[INCR_TYPES][NUM_POW];					// To hold the sqr value of the vortcity increments, this is for setting of the bin limits
	double vort_inc_num_data[INCR_TYPES][NUM_POW];				// To hold the number of vortcity incrments, for the setting of bin limits
} stats_data_struct;


// HDF5 file info struct
typedef struct HDF_file_info_struct {
	char input_file_name[512];		// Array holding input file name
	char output_file_name[512];     // Output file name array
	char spectra_file_name[512];    // Spectra file name array
	char stats_file_name[512];    	// Stats file name array
	char sys_msr_file_name[512];    // System measures file name array
	char sync_file_name[512]; 	    // Phase Sync file name array
	char input_dir[512];			// Inputs directory
	char output_dir[512];			// Output directory
	char output_tag[64]; 			// Tag to be added to the output directory
	hid_t input_file_handle;		// File handle for the input file
	hid_t output_file_handle;		// Main file handle for the output file 
	hid_t spectra_file_handle;      // Spectra file handle
	hid_t stats_file_handle;      	// Stats file handle
	hid_t sys_msr_file_handle;     	// Stats file handle
	hid_t sync_file_handle;		    // Phase sync file handle
	hid_t COMPLEX_DTYPE;			// Complex datatype handle
	int file_only;					// Indicates if output should be file only with no output folder created
	hid_t test_file_handle;         // File handle for testing
	char test_file_name[512];       // File name for testing
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
extern stats_data_struct *stats_data;           // Globale pointer to the statistics struct

#define __DATA_TYPES
#endif
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
