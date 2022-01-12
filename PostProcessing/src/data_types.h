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
#include <fftw3.h>
#define __FFTW3
#endif
#ifndef __OPENMP
#include <omp.h>
#define __OPENMP
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

// Post processing Modes
#define __REAL_STATS
#define __FULL_FIELD
#define __SPECTRA
#define __VORT
#define __SEC_PHASE_SYNC
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// These definitions define some of the solver parameters.
#define SYS_DIM 2 				// The system dimension i.e., 2D
// Statistics definitions
#define N_BINS 1000				// The number of histogram bins to use
// Phase sync 
// #define N_SECTORS 40		 	// The number of phase sectors
#define N_BINS_SEC 1000         // The number of bins in the sector pdfs
#define N_BINS_SEC_INTIME 200   // The number of bins in the sector pdfs in time
#define NUM_TRIAD_TYPES 4 		// The number of triad types contributing to the flux
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
	long int num_snaps;					// Number of snapshots in the input data file
	long int kmax; 						// The largest dealiased wavenumber
	double kmax_sqr;                    // The largest dealiased wavenumber squared
	int num_sect;						// The number of sectors in wavenumber space to be used when computing the Kuramoto order parameter 
	double t0;							// Intial time
	double T;							// Final time
	double t;							// Time variable
	double dt;							// Timestep
	double min_dt;						// Smallest timestep achieved when adaptive stepping is on
	double max_dt;						// Largest timestep achieved when adaptive stepping is on
	double dx;							// Collocation point spaceing in the x direction
	double dy;							// Collocation point spacing in the y direction
	double w_max_init;					// Max vorticity of the initial condition
	int n_spec;                         // Size of the spectra arrays
	int force_k; 						// The forcing wavenumber 
	int print_every;                    // Records how many iterations are performed before printing to file
	double EKMN_ALPHA; 					// The value of the Ekman drag coefficient
	double CFL_CONST;					// The CFL constant for the adaptive step
	double NU;							// The viscosity
	int SAVE_EVERY; 					// For specifying how often to print
	int REAL_VORT_FLAG;					// Flag to indicate if the Real space vorticity exists in solver data
	int REAL_VEL_FLAG;					// Flag to indicate if the Real space velocity exists in solver data
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data_struct {
	double* x[SYS_DIM];      // Array to hold collocation pts
	int* k[SYS_DIM];		 // Array to hold wavenumbers
	fftw_complex* w_hat;     // Fourier space vorticity
	fftw_complex* u_hat;     // Fourier space velocity
	double* w;				 // Real space vorticity
	double* u;				 // Real space velocity
	double* time;			 // Array to hold the simulation times
	fftw_complex* psi_hat;   // Fourier stream function
} runtime_data_struct;

// Post processing data struct
typedef struct postprocess_data_struct {
	double* amps;												 // Array to hold the full field zero centred amplitudes
	double* phases;			   		 							 // Array to hold the full field zero centred phases
	double* enrg;			   		 							 // Array to hold the full field zero centred energy
	double* enst;			   		 							 // Array to hold the full field zero centred enstrophy
	double* enst_spec; 		   		 							 // Array to hold the enstrophy spectrum
    double* enrg_spec; 		   		 							 // Array to hold the energy spectrum
    double* enst_alt;		         							 // Array to hold the data for enstrophy spectrum computed using stream func
    double* enrg_alt;		         							 // Array to hold the data for energy spectrum computed using stream func
    double* theta;                   							 // Array to hold the angles for the sector boundaries
    double* k_angle;											 // Array to hold the pre computed arctangents of the k3 wavevectors to speed up triad computation
    double* k1_angle;											 // Array to hold the pre computed arctangents of the k1 wavevectors to speed up triad computation
    double* k2_angle;											 // Array to hold the pre computed arctangents of the k2 wavevectors to speed up triad computation
    double* k2_angle_neg;											 // Array to hold the pre computed arctangents of the negative k2 wavevectors to speed up triad computation    
    fftw_complex* phase_order;       							 // Array to hold the phase order parameter for each sector for the individual phases
    fftw_complex* triad_phase_order[NUM_TRIAD_TYPES + 1]; 		 // Array to hold the phase order parameter for each sector for each of the triad phase types including combined
    double* phase_R;				 							 // Array to hold the phase sync per sector for the individual phases
    double* phase_Phi;               							 // Array to hold the average phase per sector for the individual phases
    double* triad_R[NUM_TRIAD_TYPES + 1];						 // Array to hold the phase sync per sector for each of the triad phase types including all together
    double* triad_Phi[NUM_TRIAD_TYPES + 1];     				 // Array to hold the average phase per sector for each of the triad phase types including all together
    gsl_histogram** phase_sect_pdf;			    				 // Struct for the histogram of the individual phases in each sector over the simulation
    gsl_histogram** phase_sect_pdf_t;							 // Struct for the histogram of the individual phases in each sector over time
    gsl_histogram** triad_sect_pdf[NUM_TRIAD_TYPES + 1];  		 // Struct for the histogram of each triad phase type in each sector over the simulation
    gsl_histogram** triad_sect_pdf_t[NUM_TRIAD_TYPES + 1];		 // Struct for the histogram of each triad phase type in each sector over time
    gsl_histogram** phase_sect_wghtd_pdf_t;						 // Struct for the weighted histogram of the individual phases in each sector over time
	gsl_histogram** triad_sect_wghtd_pdf_t[NUM_TRIAD_TYPES + 1]; // Struct for the weighted histogram of each triad phase type in each sector over time

} postprocess_data_struct;

// Post processing stats data struct
typedef struct stats_data_struct {
	gsl_rstat_workspace r_stat;  // Workplace for the running stats
	gsl_histogram* w_pdf;		 // Histogram struct for the vorticity distribution
	gsl_histogram* u_pdf;		 // Histrogam struct for the velocity distribution
} stats_data_struct;

// HDF5 file info struct
typedef struct HDF_file_info_struct {
	char input_file_name[512];		// Array holding input file name
	char output_file_name[512];     // Output file name array
	char output_dir[512];			// Output directory
	char input_dir[512];			// Input directory
	char output_tag[64]; 			// Tag to be added to the output directory
	hid_t output_file_handle;		// File handle for the output file 
	hid_t input_file_handle;		// File handle for the input file 
	hid_t COMPLEX_DTYPE;			// Complex datatype handle
	int output_file_only;			// Indicates if output should be file only with no output folder created
	int input_file_only;			// Indicates if input is file only or input folder
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
extern postprocess_data_struct *proc_data;      // Global pointer to the post processing data struct
extern stats_data_struct *stats_data;           // Globale pointer to the statistics struct

#define __DATA_TYPES
#endif
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
