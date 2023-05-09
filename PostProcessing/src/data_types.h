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
#include <gsl/gsl_histogram2d.h> 
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

// System Modes
#define HYPER_VISC 1			// Turned on hyperviscosity if called for at compilation time
#define VISC_POW 2.0            // The power of the hyperviscosity -> 1.0 means no hyperviscosity
#define EKMN_DRAG 1   			// Turn on Ekman drag if called for at compilation time
#define EKMN_POW -2.0 			// The power of the Eckman drag term -> 0.0 means no drag
#define FORCING 1 				// Indicates if forcing was used in simulation
// Post processing Modes
#if defined(__POST_STATS)
// #define __REAL_STATS
#define __VEL_INC_STATS
#define __VORT_INC_STATS
#define __VEL_GRAD_STATS
#define __VORT_GRAD_STATS
#define __VEL_STR_FUNC_STATS
#define __VORT_STR_FUNC_STATS
// #define __VORT_RADIAL_STR_FUNC_STATS
#define __MIXED_VEL_STR_FUNC_STATS
#define __MIXED_VORT_STR_FUNC_STATS
#endif
#if defined(__POST_FIELD) || defined(__POST_SYNC)
#define __FULL_FIELD
#define __SPECTRA
#define __ENST_FLUX
#define __ENRG_FLUX
#endif
#if defined(__POST_SYNC)
// #define __PHASE_SYNC
#define __SEC_PHASE_SYNC
// #define __SEC_PHASE_SYNC_STATS
// #define __SEC_PHASE_SYNC_FLUX_STATS
// #define __SEC_PHASE_SYNC_STATS_IN_TIME
// #define __SEC_PHASE_SYNC_STATS_IN_TIME_ALL
// #define __SEC_PHASE_SYNC_STATS_IN_TIME_1D
// #define __SEC_PHASE_SYNC_STATS_IN_TIME_2D
#endif
// Postprocessing data sets
// #define __VORT_FOUR
// #define __VORT_REAL
// #define __MODES
// #define __REALSPACE
// #define __NONLIN
// #define __TIME
// #define __COLLOC_PTS
// #define __WAVELIST

#define __DEALIAS_23
// #define __DEALIAS_HOU_LI
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// These definitions define some of the solver parameters.
#define SYS_DIM 2 					// The system dimension i.e., 2D
// Statistics definitions	
#define N_BINS 1000					// The number of histogram bins to use
#define NUM_INCR 2              	// The number of increment length scales
#define INCR_TYPES 2 				// The number of increment directions i.e., longitudinal and transverse
#define STR_FUNC_MAX_POW 6      	// The maximum pow of the structure functions to compute
#define BIN_LIM 40              	// The limit of the bins for the velocity increments
// Phase sync 
#define PRE_COMPUTE 1   			// Indicator for pre compute mode
#define N_BINS_SEC 1000         	// The number of bins in the sector pdfs
#define NUM_MOMENTS 4 				// The number of moments to record for the 2d contribution stats
#define NUM_CONTRIB	3				// The number of types of contribution, i.e., 1d, 2d, both 1d and 2d combined
#define NUM_TRIAD_CLASS 2 			// The number of triad classes i.e., either normal triad or generalized triads
#define NUM_TRIAD_TYPES 6 			// The number of triad types contributing to the flux, these are dependent on the term and sign of the term 
#define NUM_K1_SECTORS 8			// The number of k1 sectors to search over in a reduced search @ +/- 30, 45, 60 & 90 degrees
#define NUM_K_DATA 17           	// The number of wavevector data to precompute and store
#define	K1_X	  0 				// The index for the k1_x wavenuber
#define	K1_Y	  1 				// The index for the k1_y wavenuber
#define	K2_X 	  2 				// The index for the k2_x wavenuber
#define	K2_Y  	  3 				// The index for the k2_y wavenuber
#define	K3_X  	  4 				// The index for the k3_x wavenuber
#define	K3_Y  	  5 				// The index for the k3_y wavenuber
#define	K1_SQR	  6 				// The index for the |k1|^2
#define	K2_SQR	  7 				// The index for the |k2|^2
#define	K3_SQR	  8 				// The index for the |k3|^2
#define	K1_ANGLE  9 				// The index for the anlge of k1
#define	K2_ANGLE  10 				// The index for the anlge of k2
#define	K3_ANGLE  11 				// The index for the anlge of k3
#define	K1_ANGLE_NEG  12 			// The index for the anlge of -k1
#define	K2_ANGLE_NEG  13 			// The index for the anlge of -k2
#define	K3_ANGLE_NEG  14 			// The index for the anlge of -k3
#define FLUX_TERM 15				// Indicator which identifies whether data is in postive or negative flux term
#define CONTRIB_TYPE 16         	// Indicator for which type of contribution to the flux, either 1d or 2d
#define POS_FLUX_TERM 0 			// Indicates postive flux term
#define NEG_FLUX_TERM 1         	// Indicates negative flux term
#define CONTRIB_1D 0 		    	// Indicates 1d contribution (same sector) to the flux 
#define CONTRIB_2D 1            	// Indicates 2d contribution (across sectors) to the flux
#define CONTRIB_ALL 2           	// Indicates when 1d and 2d contributions are combined
#define N_BINS_SEC_1D_T 50      	// The number of bins in the sector pdfs in time for 1D contributions
#define N_BINS_SEC_2D_T 50      	// The number of bins in the sector pdfs in time for 2D contributions
#define N_BINS_SEC_ALL_T 50     	// The number of bins in the sector pdfs in time for 1D and 2D contributions
#define N_BINS_TRIADS_ALL_T 50  	// The number of bins to use in the all triads pdf in time
#define N_BINS_X_JOINT_ALL_T 100	// The number of bins for the joint PDF in time in the x direction  
#define N_BINS_Y_JOINT_ALL_T 100	// The number of bins for the joint PDF in time in the y direction  
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
	double kmax_frac;					// Fraction of the maximum wavevector to consider for phase sync and enstrophy flux
	double kmax_C;						// The radius of the set C -> = kmax_frac * kmax
	double kmax_C_sqr;					// The sqr of the radius for the set C
	double kmax_sqr;                    // The largest dealiased wavenumber squared
	double kmin_sqr;                    // The smallest dealiased wavenumber squared
	int num_k3_sectors;					// The number of sectors in wavenumber space to be used when computing the Kuramoto order parameter 
	int num_k1_sectors;					// Variable to control the number of k1 sectors
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
	int FORCING_FLAG; 					// Indicates if forcing was used
	int force_k; 						// The forcing wavenumber 
	double force_scale_var;				// The scaling variable for the forced modes
	int num_forced_modes; 				// The number of modes to be forced on the local process
	int print_every;                    // Records how many iterations are performed before printing to file
	double CFL_CONST;					// The CFL constant for the adaptive step
	double EKMN_ALPHA; 					// The value of the Ekman drag coefficient
	int EKMN_DRAG_FLAG; 				// Indicator for Ekman drag
	double EKMN_DRAG_POW;				// Power of the hypo drag term
	double NU;							// The viscosity
	int HYPER_VISC_FLAG;				// Indicator for hyperviscosity 
	double HYPER_VISC_POW;				// The pow of the hyper viscosity
	int SAVE_EVERY; 					// For specifying how often to print
	int REAL_VORT_FLAG;					// Flag to indicate if the Real space vorticity exists in solver data
	int REAL_VEL_FLAG;					// Flag to indicate if the Real space velocity exists in solver data
	int num_threads;					// The number of OMP threads to use
	int num_fftw_threads;				// The number of FFTW threads to use
	int thread_id;						// The ID of the OMP threads
	int num_triad_per_sec_est;          // The estimate number of triads per sector
    int REDUCED_K1_SEARCH_FLAG;			// Flag to control whether we are doing a reduced search over specifc sectors of k1 or not
    int chk_pt_every;					// Checkpoint every n iterations
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data_struct {
	double* x[SYS_DIM];      // Array to hold collocation pts
	int* k[SYS_DIM];		 // Array to hold wavenumbers
	fftw_complex* w_hat;     // Fourier space vorticity
	fftw_complex* tmp_w_hat; // Temporary Fourier space vorticity
	fftw_complex* u_hat;     // Fourier space velocity
	fftw_complex* tmp_u_hat; // Fourier space velocity
	double* tmp_u; 			 // Temporary array to read & write in velocities
	double* w;				 // Real space vorticity
	double* u;				 // Real space velocity
	double* time;			 // Array to hold the simulation times
	fftw_complex* psi_hat;   // Fourier stream function
	fftw_complex* forcing;	 // Array to hold the forcing for the current timestep
	double* forcing_scaling; // Array to hold the initial scaling for the forced modes
	int* forcing_indx;		 // Array to hold the indices of the forced modes
	int* forcing_k[SYS_DIM]; // Array containg the wavenumbers for the forced modes
} runtime_data_struct;

// Post processing data struct
typedef struct postprocess_data_struct {
	double* amps;												             				// Array to hold the full field zero centred amplitudes
	double* phases;			   		 							             				// Array to hold the full field zero centred phases
	double* enrg;			   		 							             				// Array to hold the full field zero centred energy
	double* enst;			   		 							             				// Array to hold the full field zero centred enstrophy
	double* enst_spec; 		   		 							             				// Array to hold the enstrophy spectrum
    double* enrg_spec; 		   		 							             				// Array to hold the energy spectrum
    double dtheta_k3; 												         				// The angle between sector mid points for k3
    double dtheta_k1; 												         				// The angle between sector mid points for k1
	bool pos_flux_term_cond;									             				// Boolean to store the condition on the wavevectors for the first (positive) term in the enstrophy flux
	bool neg_flux_term_cond;									             				// Boolean to store the condition on the wavevectors for the second (negative) term in the enstrophy flux
    double* d_enst_dt_spec;										             				// Array to hold the time derivative of the enstrophy spectrum
    double* enst_flux_spec;										             				// Array to hold the enstrophy flux spectrum
    double* enst_diss_spec;										             				// Array to hold the enstrophy dissipation spectrum
    double* enst_flux_C;										             				// Array to hold the enstrophy flux out of the set C defined by radius sys_vars->kmax_frac * sys_vars->kmax
    double* enst_diss_C;										             				// Array to hold the enstrophy dissipation in the set C defined by radius sys_vars->kmax_frac * sys_vars->kmax
    double* enst_flux_C_theta;									             				// Array to hold the enstrophy flux in/out of the set C_\theta defined by radius sys_vars->kmax_frac * sys_vars->kmax
    double* enst_diss_C_theta;												 				// Array to hold the enstrophy diss in/out of the set C_\theta defined by radius sys_vars->kmax_frac * sys_vars->kmax
    double* d_enrg_dt_spec;										             				// Array to hold the time derivative of the energy spectrum
    double* enrg_flux_spec;										             				// Array to hold the energy flux spectrum
    double* enrg_diss_spec;										             				// Array to hold the energy dissipation spectrum
    double* enrg_flux_C;										             				// Array to hold the energy flux out of the set C defined by radius sys_vars->kmax_frac * sys_vars->kmax
    double* enrg_diss_C;										             				// Array to hold the energy dissipation in the set C defined by radius sys_vars->kmax_frac * sys_vars->kmax
    fftw_complex* enst_diss_field;											 				// Array to holde the enstrophy dissipation field in Fourier space.
    fftw_complex* phase_order_C_theta;										 				// Array to hold the phase order parameter for each C_theta
    fftw_complex* phase_order_C_theta_norm;													// Array to hold the phase order parameter for each C_theta normed
    fftw_complex phase_order_C_theta_triads_test[NUM_TRIAD_TYPES + 1];		 				// Array to hold the phase order parameter for each C_theta triads
    fftw_complex* phase_order_C_theta_triads[NUM_TRIAD_TYPES + 1];			 				// Array to hold the phase order parameter for each C_theta triads
    fftw_complex* phase_order_C_theta_triads_1d[NUM_TRIAD_TYPES + 1];		 				// Array to hold the phase order parameter for each C_theta triads for 1d contributions
    fftw_complex** phase_order_C_theta_triads_2d[NUM_TRIAD_TYPES + 1];		 				// Array to hold the phase order parameter for each C_theta triads for 2d contributions
    fftw_complex* phase_order_C_theta_triads_unidirec[NUM_TRIAD_TYPES + 1];					// Array to hold the unidirectional phase order parameter for each C_theta triads
    fftw_complex* phase_order_C_theta_triads_unidirec_1d[NUM_TRIAD_TYPES + 1];				// Array to hold the unidirectional phase order parameter for each C_theta triads for 1d contributions
    fftw_complex** phase_order_C_theta_triads_unidirec_2d[NUM_TRIAD_TYPES + 1];				// Array to hold the unidirectional phase order parameter for each C_theta triads for 2d contributions
    double** phase_order_norm_const[2][NUM_TRIAD_TYPES + 1];								// Array to hold the normalization constants for the collective phase order parameters				
    fftw_complex* dw_hat_dt; 									             				// Array to hold the RHS of the vorticity equation
    fftw_complex* grad_w_hat;												 				// Array to hold the derivative of the vorticity in the x and y direction in Fourier space     
    fftw_complex* grad_u_hat;												 				// Array to hold the derivative of the velocity in the x and y direction in Fourier space     
    double* grad_w;															 				// Array to hold the derivative of the vorticity in the x and y direction in Real space 
    double* grad_u;															 				// Array to hold the derivative of the vorticity in the x and y direction in Real space 
    double* nonlinterm;											 			 				// Array to hold the nonlinear term after multiplication in real space -> for nonlinear RHS funciotn
    double* nabla_w;											 			 				// Array to hold the gradient of the real space vorticity -> for nonlinear RHS function
    double* nabla_psi;											 			 				// Array to hold the gradient of the real space stream function -> for nonlinear RHS function
	double* theta_k3;                   							 		 				// Array to hold the angles for the sector mid points for k3
	double* theta_k1;                   							 		 				// Array to hold the angles for the sector mid points for k1
    double* mid_angle_sum;									     			 				// Array to hold the pre computed midpoint angle sums -> this will determine which sector k2 is in
    double* phase_angle;										 			 				// Array to hold the pre computed arctangents of the wavevectors for the individual phases   
    double* k1_sector_angles;									 			 				// Array to hold the pre computed arctangents of the wavevectors for the individual phases   
    double**** phase_sync_wave_vecs;										 				// Array of pointers to arrays to hold the wavevectors in a given sector
    double** phase_sync_wave_vecs_test;									     				// Array of pointers to arrays to hold the wavevectors in the test case
    int** num_wave_vecs;													 				// Array to hold the number of wavevector triads per secotr
    fftw_complex* phase_order;       							 			 				// Array to hold the phase order parameter for each sector for the individual phases
	fftw_complex* triad_phase_order[NUM_TRIAD_TYPES + 1]; 		 			 				// Array to hold the phase order parameter for each sector for each of the triad phase types including combined
	fftw_complex triad_phase_order_test[NUM_TRIAD_TYPES + 1]; 		 		 				// Array to hold the phase order parameter for each sector for each of the triad phase types including combined
	fftw_complex* triad_phase_order_1d[NUM_TRIAD_TYPES + 1]; 		 		 				// Array to hold the phase order parameter for 1d contributions for each sector for each of the triad phase types including combined
	fftw_complex** triad_phase_order_2d[NUM_TRIAD_TYPES + 1];		 		 				// Array to hold the phase order parameter for each sector for each of the triad phase types including combined
	double* phase_R;				 							 			 				// Array to hold the phase sync per sector for the individual phases
	double* phase_Phi;               							 			 				// Array to hold the average phase per sector for the individual phases
	double* enst_flux[NUM_TRIAD_TYPES + 1];						 			 				// Array to hold the flux of enstrophy for each triad type for each sector
	double enst_flux_test[NUM_TRIAD_TYPES + 1];						 		 				// Array to hold the flux of enstrophy for each triad type for each sector
	double* enst_flux_1d[NUM_TRIAD_TYPES + 1];					 			 				// Array to hold the flux of enstrophy for 1d contributoins for each triad type for each sector
    int* num_triads[NUM_TRIAD_TYPES + 1];						 			 				// Array to hold the number of triads for each triad type
    int num_triads_test[NUM_TRIAD_TYPES + 1];						 		 				// Array to hold the number of triads for each triad type
    int* num_triads_1d[NUM_TRIAD_TYPES + 1];					 			 				// Array to hold the number of triads for 1d contributions for each triad type
    double* triad_R[NUM_TRIAD_TYPES + 1];						 			 				// Array to hold the phase sync per sector for each of the triad phase types including all together
    double* triad_Phi[NUM_TRIAD_TYPES + 1];     				 			 				// Array to hold the average phase per sector for each of the triad phase types including all together
    double triad_R_test[NUM_TRIAD_TYPES + 1];						 		 				// Array to hold the phase sync per sector for each of the triad phase types including all together
    double triad_Phi_test[NUM_TRIAD_TYPES + 1];     				 		 				// Array to hold the average phase per sector for each of the triad phase types including all together
    double* triad_R_1d[NUM_TRIAD_TYPES + 1];						 		 				// Array to hold the phase sync for 1d contributions per sector for each of the triad phase types including all together
    double* triad_Phi_1d[NUM_TRIAD_TYPES + 1];     				 			 				// Array to hold the average phase for 1d contributions per sector for each of the triad phase types including all together
    double** enst_flux_2d[NUM_TRIAD_TYPES + 1];					 			 				// Array to hold the flux of enstrophy for each triad type for each sector
    int** num_triads_2d[NUM_TRIAD_TYPES + 1];					 			 				// Array to hold the number of triads for each triad type
    double** triad_R_2d[NUM_TRIAD_TYPES + 1];					 			 				// Array to hold the phase sync per sector for each of the triad phase types including all together
    double** triad_Phi_2d[NUM_TRIAD_TYPES + 1];		  				 	     				// Array to hold the average phase per sector for each of the triad phase types including all together
    double max_bin_enst_flux[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES - 1];							// Workplace for the running stats for enstrophy flux
	double* max_enst_flux[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES - 1];							// Workplace for the running stats for enstrophy flux
	gsl_histogram2d* triads_wghtd_2d_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES - 1];				// Workplace for the running stats for enstrophy flux
	// In time stats objects
	gsl_histogram* triads_all_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1];					// Array Structs to hold the pdfs for all triads both triad class: the normal triads and generalized triads, each triad type, for each contribution type in time
	gsl_histogram* triads_wghtd_all_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1];			// Array Structs to hold the pdfs for all triads both triad class: the normal triads and generalized triads, each triad type, for each contribution type in time
	gsl_histogram** triads_sect_all_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1];			// Array Structs to hold the over sectors pdfs for both triad class: the normal triads and generalized triads, each triad type, for each contribution type in time
	gsl_histogram** triads_sect_1d_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1];			    // Array Structs to hold the over sectors pdfs for both triad class    : the normal triads and generalized triads, each triad type, for each contribution type in time
	gsl_histogram*** triads_sect_2d_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1];			// Array Structs to hold the over sectors pdfs for both triad class: the normal triads and generalized triads, each triad type, for each contribution type in time
	gsl_histogram** triads_sect_wghtd_all_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1]; 	    // Array Struct for the weighted histogram over sectors  of each triad class, each triad phase type for each contribution type in each sector in time
	gsl_histogram** triads_sect_wghtd_1d_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1]; 	    // Array Struct for the weighted histogram over sectors  of each triad class, each triad phase type for each contribution type in each sector in time
	gsl_histogram*** triads_sect_wghtd_2d_pdf_t[NUM_TRIAD_CLASS][NUM_TRIAD_TYPES + 1]; 	    // Array Struct for the weighted histogram over sectors  of each triad class, each triad phase type for each contribution type in each sector in time
	// Over time stats objects
	gsl_histogram*** triad_R_2d_pdf;						 								// Array to hold the histogram objects for the 2D contributions for the Phase sync
	gsl_histogram*** triad_Phi_2d_pdf;														// Array to hold the histogram objects for the 2D contributions for the triads
	gsl_histogram*** enst_flux_2d_pdf;														// Array to hold the histogram objects for the 2D contributions for the enstrophy flux
	gsl_rstat_workspace*** triad_R_2d_stats;												// Workplace for the running stats for the 2d phase sync
	gsl_rstat_workspace*** triad_Phi_2d_stats;												// Workplace for the running stats for 2d average phase
	gsl_rstat_workspace*** enst_flux_2d_stats;												// Workplace for the running stats for the 2d enstrophy flux contribution
} postprocess_data_struct;

// Post processing stats data struct
typedef struct stats_data_struct {
	gsl_rstat_workspace* r_stat_grad_u[SYS_DIM + 1];		  		// Workplace for the running stats for the gradients of velocity (both for each direction and combined)
	gsl_rstat_workspace* r_stat_grad_w[SYS_DIM + 1];		  		// Workplace for the running stats for the gradients of vorticity (both for each direction and combined)
	gsl_rstat_workspace* r_stat_w;									// Workplace for the running stats for the velocity (both for each direction and combined)
	gsl_rstat_workspace* r_stat_u[SYS_DIM + 1];						// Workplace for the running stats for the vorticity (both for each direction and combined)
	gsl_rstat_workspace* r_stat_vel_incr[INCR_TYPES][NUM_INCR];		// Workplace for the running stats for the velocity increments
	gsl_rstat_workspace* r_stat_vort_incr[INCR_TYPES][NUM_INCR];	// Workplace for the running stats for the vorticity increments
	gsl_histogram* w_pdf;		 									// Histogram struct for the vorticity distribution
	gsl_histogram* u_pdf;		  									// Histrogam struct for the velocity distribution
	gsl_histogram* vel_grad[SYS_DIM + 1];		 					// Array to hold the PDFs of the longitudinal and transverse velocity gradients 
	gsl_histogram* vort_grad[SYS_DIM + 1];		 					// Array to hold the PDFs of the longitudinal and transverse vorticity gradients 
	gsl_histogram* vel_incr[INCR_TYPES][NUM_INCR]; 					// Array to hold the PDFs of the longitudinal and transverse velocity increments for each increment
	gsl_histogram* w_incr[INCR_TYPES][NUM_INCR]; 					// Array to hold the PDFs of the longitudinal and transverse vorticity increments for each increment
	double* str_func[INCR_TYPES][STR_FUNC_MAX_POW - 2];				// Array to hold the structure functions longitudinal and transverse velocity increments for each increment
	double* str_func_abs[INCR_TYPES][STR_FUNC_MAX_POW - 2];			// Array to hold the structure functions longitudinal and transverse velocity increments for each absolute increment
} stats_data_struct;

// HDF5 file info struct
typedef struct HDF_file_info_struct {
	char input_file_name[1024];		// Array holding input file name
	char output_file_name[1024];     // Output file name array
	char wave_vec_data_name[1024];   // File path for the phase sync wavector data
	char output_dir[1024];			// Output directory
	char input_dir[1024];			// Input directory
	char output_tag[64]; 			// Tag to be added to the output directory
	hid_t output_file_handle;		// File handle for the output file 
	hid_t input_file_handle;		// File handle for the input file 
	hid_t wave_vec_file_handle;		// Wavevector file handle
	hid_t COMPLEX_DTYPE;			// Complex datatype handle
	int output_file_only;			// Indicates if output should be file only with no output folder created
	int input_file_only;			// Indicates if input is file only or input folder
} HDF_file_info_struct;

// Complex datatype struct for HDF5
typedef struct complex_type_tmp {
	double re;   		// real part 
	double im;   		// imaginary part 
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