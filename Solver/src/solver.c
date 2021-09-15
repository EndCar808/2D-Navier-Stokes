/**
* @file solver.c 
* @author Enda Carroll
* @date Jun 2021
* @brief file containing the main functions used in the pseudopectral method
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"
#include "solver.h"

// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// Define RK4 variables
#if defined(__RK4)
static const double RK4_C2 = 0.5, 	  RK4_A21 = 0.5, \
				  	RK4_C3 = 0.5,           					RK4_A32 = 0.5, \
				  	RK4_C4 = 1.0,                      									   RK4_A43 = 1.0, \
				              	 	  RK4_B1 = 1.0/6.0, 		RK4_B2  = 1.0/3.0, 		   RK4_B3  = 1.0/3.0, 		RK4_B4 = 1.0/6.0;
// Define RK5 Dormand Prince variables
#elif defined(__RK5) || defined(__DPRK5)
static const double RK5_C2 = 0.2, 	  RK5_A21 = 0.2, \
				  	RK5_C3 = 0.3,     RK5_A31 = 3.0/40.0,       RK5_A32 = 0.5, \
				  	RK5_C4 = 0.8,     RK5_A41 = 44.0/45.0,      RK5_A42 = -56.0/15.0,	   RK5_A43 = 32.0/9.0, \
				  	RK5_C5 = 8.0/9.0, RK5_A51 = 19372.0/6561.0, RK5_A52 = -25360.0/2187.0, RK5_A53 = 64448.0/6561.0, RK5_A54 = -212.0/729.0, \
				  	RK5_C6 = 1.0,     RK5_A61 = 9017.0/3168.0,  RK5_A62 = -355.0/33.0,     RK5_A63 = 46732.0/5247.0, RK5_A64 = 49.0/176.0,    RK5_A65 = -5103.0/18656.0, \
				  	RK5_C7 = 1.0,     RK5_A71 = 35.0/384.0,								   RK5_A73 = 500.0/1113.0,   RK5_A74 = 125.0/192.0,   RK5_A75 = -2187.0/6784.0,    RK5_A76 = 11.0/84.0, \
				              		  RK5_B1  = 35.0/384.0, 							   RK5_B3  = 500.0/1113.0,   RK5_B4  = 125.0/192.0,   RK5_B5  = -2187.0/6784.0,    RK5_B6  = 11.0/84.0, \
				              		  RK5_Bs1 = 5179.0/57600.0, 						   RK5_Bs3 = 7571.0/16695.0, RK5_Bs4 = 393.0/640.0,   RK_Bs5  = -92097.0/339200.0, RK5_Bs6 = 187.0/2100.0, RK5_Bs7 = 1.0/40.0;
#endif
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Main function that performs the pseudospectral solver
 */
void SpectralSolve(void) {

	// Initialize variables
	int tmp;
	int indx;
	herr_t status;
	const long int N[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1]};
	const long int NBatch[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1] / 2 + 1};
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;
	#ifdef TESTING
	double max_vort;
	double norms[2];
	#endif

	// Initialize the Runge-Kutta struct
	struct RK_data_struct* RK_data;	   // Initialize pointer to a RK_data_struct
	struct RK_data_struct RK_data_tmp; // Initialize a RK_data_struct
	RK_data = &RK_data_tmp;		       // Point the ptr to this new RK_data_struct
	// -------------------------------
	// Allocate memory
	// -------------------------------
	AllocateMemory(NBatch, RK_data);

	// -------------------------------
	// FFTW Plans Setup
	// -------------------------------
	InitializeFFTWPlans(N);

	// -------------------------------
	// Initialize the System
	// -------------------------------
	// Initialize the collocation points and wavenumber space 
	InitializeSpaceVariables(run_data->x, run_data->k, N);

	// Get initial conditions
	InitialConditions(run_data->w_hat, run_data->u, run_data->u_hat, N);
	
	// PrintScalarReal(run_data->w, N, "w");
	// PrintScalarFourier(run_data->w_hat, N, "wh");

	// NonlinearRHSBatch(run_data->w_hat, RK_data->RK1, RK_data->nabla_psi, RK_data->nabla_w);
	// PrintScalarFourier(RK_data->RK1, N, "a_RHSh");

	// -------------------------------
	// Integration Variables
	// -------------------------------
	// Set the spatial increments
	sys_vars->dx = 2.0 * M_PI / (double )Nx;
	sys_vars->dy = 2.0 * M_PI / (double )Ny;

	// Get the timestep using a CFL like condition
	double umax          = GetMaxData("VEL");
	sys_vars->w_max_init = GetMaxData("VORT");
	#ifdef __ADAPTIVE_STEP
	GetTimestep(&(sys_vars->dt));
	#endif

	// Compute integration time variables
	double t0 = sys_vars->t0;
	double t  = t0;
	double dt = sys_vars->dt;
	double T  = sys_vars->T;

	// Number of iterations
	sys_vars->num_t_steps     = (T - t0) / dt;
	sys_vars->num_print_steps = sys_vars->num_t_steps / sys_vars->SAVE_EVERY + 1; // plus one to include initial condition
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps);
	}

	// Variable to control how ofter to print to screen -> 10% of num time steps
	int print_update = (sys_vars->num_t_steps >= 10 ) ? (int)((double)sys_vars->num_t_steps * 0.1) : 1;


	// -------------------------------
	// Create & Open Output File
	// -------------------------------
	#ifdef TESTING
	// If testing is enabled and TG initial condition selected -> computed TG solution for writing to file @ t = t0
	if(!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
		TaylorGreenSoln(t0, N);
	}
	#endif

	// Inialize system measurables
	InitializeSystemMeasurables(RK_data);

	// Create and open the output file - also write initial conditions to file
	CreateOutputFilesWriteICs(N, dt);

	printf("ToT E: %1.18lf\t ToT Enstr: %1.18lf\tToT Pal: %1.18lf\tEDiss: %g\tEnDiss: %g\n", run_data->tot_energy[0], run_data->tot_enstr[0], run_data->tot_palin[0], run_data->enst_flux_sbst[0], run_data->enst_diss_sbst[0]);
	if (!(sys_vars->rank)) {
		// Reduce on to rank 0
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_energy, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_enstr, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->tot_palin, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enrg_diss, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, run_data->enst_diss, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		printf("Iter: %d\tt: %1.6lf\tdt: %g\t Max Vort: %1.4lf \tTKE: %1.8lf\tENS: %1.8lf\tPAL: %g\tEDiss: %g\tEnDiss: %g\n", 0, 0.0, dt, sys_vars->w_max_init, run_data->tot_energy[0], run_data->tot_enstr[0], run_data->tot_palin[0], run_data->enst_flux_sbst[0], run_data->enst_diss_sbst[0]);
	}
	else {
		// Reduce all other process to rank 0
		MPI_Reduce(run_data->tot_energy, NULL,  sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_enstr, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->tot_palin, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enrg_diss, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(run_data->enst_diss, NULL, sys_vars->num_print_steps, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}



	// int spectrum_size = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	// int spec_indx;


	// // ------------------------------------
	// // Compute Spectrum
	// // ------------------------------------
	// if (!sys_vars->rank) {
	// 	printf("\nspecsize: %d \t kmax: %d", spectrum_size, Ny_Fourier - 1);
	// 	printf("\n");
	// 	for (int i = 0; i < sys_vars->local_Nx; ++i) {
	// 		tmp = i * Ny_Fourier;
	// 		for (int j = 0; j < Ny_Fourier; ++j) {
	// 			indx = tmp + j;

	// 			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
	// 			spec_indx = (int) sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));
	// 			printf("(%d, %d): %d - %1.4lf  ||  ", run_data->k[0][i], run_data->k[1][j], spec_indx, sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j])));
	// 		}
	// 		printf("\n");
	// 	}	
	// }
	



	
	//////////////////////////////
	// Begin Integration
	//////////////////////////////
	#ifdef __DPRK5
	try = 1;
	double dt_new;
	#endif
	int iters          = 1;
	int save_data_indx = 1;
	// while (t < T) {

	// 	// -------------------------------	
	// 	// Integration Step
	// 	// -------------------------------
	// 	#ifdef __RK4
	// 	RK4Step(dt, N, sys_vars->local_Nx, RK_data);
	// 	#elif defined(__RK5)
	// 	RK5DPStep(dt, N, sys_vars->local_Nx, RK_data);
	// 	#elif defined(__DPRK5)
	// 	while (try) {
	// 		// Try a Dormand Prince step and compute the local error
	// 		RK5DPStep(dt, N, sys_vars->local_Nx, RK_data);

	// 		// Compute the new timestep
	// 		dt_new = dt * DPMin(DP_DELTA_MAX, DPMax(DP_DELTA_MIN, DP_DELTA * pow(1.0 / RK_data->DP_errr, 0.2)))
			
	// 		// If error is bad repeat else move on
	// 		if (RK_data->DP_err < 1.0) {
	// 			RK->DP_fails++;
	// 			dt = dt_new;
	// 			continue;
	// 		}
	// 		else {
	// 			dt = dt_new;
	// 			break;
	// 		}
	// 	}
	// 	#endif

	// 	// -------------------------------
	// 	// Write To File
	// 	// -------------------------------
	// 	if (iters % sys_vars->SAVE_EVERY == 0) {
	// 		#ifdef TESTING
	// 		TaylorGreenSoln(t, N);
	// 		#endif

	// 		// Record System Measurables
	// 		RecordSystemMeasures(t, save_data_indx, RK_data);

	// 		// Write the appropriate datasets to file
	// 		WriteDataToFile(t, dt, save_data_indx);
			
	// 		// Update saving data index
	// 		save_data_indx++;
	// 	}

	// 	// -------------------------------
	// 	// Update & System Check
	// 	// -------------------------------
	// 	// Update timestep & iteration counter
	// 	#if defined(__ADAPTIVE_STEP) 
	// 	GetTimestep(&dt);
	// 	t += dt; 
	// 	#elif !defined(__DPRK5) && !defined(__ADAPTIVE_STEP)
	// 	t = iters * dt;
	// 	#endif
	// 	iters++;

	// 	// Check System: Determine if system has blown up or integration limits reached
	// 	SystemCheck(dt, iters);

	// 	// -------------------------------
	// 	// Print Update To Screen
	// 	// -------------------------------
	// 	#ifdef __PRINT_SCREEN
	// 	#ifdef TESTING
	// 	if (iters % TEST_PRINT == 0) {
	// 		// Call testing 
	// 		max_vort = GetMaxData("VORT");
	// 		if(!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
	// 			TestTaylorGreenVortex(t, N, norms);
	// 			RecordSystemMeasures(t, save_data_indx, RK_data);
	// 			if( !(sys_vars->rank) ) {	
	// 				// printf("Iter: %d\tt: %1.6lf\tdt: %g\t Max Vort: %1.4lf \tL2 Err: %g\tLinf Err: %g\n", iters, t, dt, max_vort, norms[0], norms[1]);
	// 				printf("Iter: %d\tt: %1.6lf\tdt: %g\t Max Vort: %1.4lf \tTKE: %1.8lf\tENS: %1.8lf\tPAL: %g\tEDiss: %g\tEnDiss: %g\n", iters, t, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx], run_data->enst_flux_sbst[save_data_indx], run_data->enst_diss_sbst[save_data_indx]);
	// 			}
	// 		}
	// 		else {
	// 			if ( iters % sys_vars->SAVE_EVERY != 0) {
	// 				RecordSystemMeasures(t, save_data_indx, RK_data);
	// 			}
	// 			if( !(sys_vars->rank) ) {	
	// 				printf("Iter: %d\tt: %1.6lf\tdt: %g\t Max Vort: %1.4lf \tKE: %1.5lf\tENS: %1.5lf\n", iters, t, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx]);
	// 			}
	// 		}
	// 	}
	// 	#else
	// 	if (iters % print_update == 0) {
	// 		// If needed compute system measures for printing to screen
	// 		if ( iters % sys_vars->SAVE_EVERY != 0) {
	// 			RecordSystemMeasures(t, save_data_indx, RK_data);
	// 		}
	// 		if( !(sys_vars->rank) ) {	
	// 			printf("Iter: %d/%ld\tt: %1.6lf/%1.3lf\tdt: %g\t ----------- \tKE: %1.5lf\tENS: %1.5lf\tPAL: %1.5lf\n", iters, sys_vars->num_t_steps, t, T, dt, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx]);
	// 		}
	// 	}
	// 	#endif	
	// 	#endif
	// }
	//////////////////////////////
	// End Integration
	//////////////////////////////
	

	// ------------------------------- 
	// Final Writes to Output File
	// -------------------------------
	FinalWriteAndCloseOutputFile(N);
	

	// -------------------------------
	// Clean Up 
	// -------------------------------
	FreeMemory(RK_data);
}
/**
 * Function to perform a single step of the RK5 or Dormand Prince scheme
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Nx Int indicating the local size of the first dimension of the arrays	
 * @param RK_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {


	// Initialize vairables
	int tmp;
	int indx;
	#ifdef __NAVIER
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny_Fourier = N[1] / 2 + 1;
	#ifdef __DPRK5
	const long int Nx = N[0];
	double dp_ho_step;
	#endif
	
	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHSBatch(run_data->w_hat, RK_data->RK1, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A21 * RK_data->RK1[indx];
		}
	}
	// ----------------------- Stage 2
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK2, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A31 * RK_data->RK1[indx] + dt * RK5_A32 * RK_data->RK2[indx];
		}
	}
	// ----------------------- Stage 3
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK3, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A41 * RK_data->RK1[indx] + dt * RK5_A42 * RK_data->RK2[indx] + dt * RK5_A43 * RK_data->RK3[indx];
		}
	}
	// ----------------------- Stage 4
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK4, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A51 * RK_data->RK1[indx] + dt * RK5_A52 * RK_data->RK2[indx] + dt * RK5_A53 * RK_data->RK3[indx] + dt * RK5_A54 * RK_data->RK4[indx];
		}
	}
	// ----------------------- Stage 5
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK5, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A61 * RK_data->RK1[indx] + dt * RK5_A62 * RK_data->RK2[indx] + dt * RK5_A63 * RK_data->RK3[indx] + dt * RK5_A64 * RK_data->RK4[indx] + dt * RK5_A65 * RK_data->RK5[indx];
		}
	}
	// ----------------------- Stage 6
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK6, RK_data->nabla_psi, RK_data->nabla_w);
	#ifdef __DPRK5
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A71 * RK_data->RK1[indx] + dt * RK5_A73 * RK_data->RK3[indx] + dt * RK5_A74 * RK_data->RK4[indx] + dt * RK5_A75 * RK_data->RK5[indx] + dt * RK5_A76 * RK_data->RK6[indx];
		}
	}
	// ----------------------- Stage 7
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK7, RK_data->nabla_psi, RK_data->nabla_w);
	#endif

	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;


			#ifdef __EULER
			// Update the Fourier space vorticity with the RHS
			run_data->w_hat[indx] = run_data->w_hat[indx] + (dt * (RK5_B1 * RK_data->RK1[indx]) + dt * (RK5_B3 * RK_data->RK3[indx]) + dt * (RK5_B4 * RK_data->RK4[indx]) + dt * (RK5_B5 * RK_data->RK5[indx]) + dt * (RK5_B6 * RK_data->RK6[indx]));
			#elif defined(__NAVIER)
			// Compute the pre factors for the RK4CN update step
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// Both Hyperviscosity and Ekman drag
			D_fac = dt * (sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW)); 
			#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// No hyperviscosity but we have Ekman drag
			D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW)); 
			#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
			// Hyperviscosity only
			D_fac = dt * (sys_vars->NU * pow(k_sqr, VIS_POW)); 
			#else 
			// No hyper viscosity or no ekman drag -> just normal viscosity
			D_fac = dt * (sys_vars->NU * k_sqr); 
			#endif

			// Complete the update step
			run_data->w_hat[indx] = run_data->w_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK5_B1 * RK_data->RK1[indx] + RK5_B3 * RK_data->RK3[indx] + RK5_B4 * RK_data->RK4[indx] + RK5_B5 * RK_data->RK5[indx] + RK5_B6 * RK_data->RK6[indx]);
			#endif
			#ifdef __DPRK5
			if (iters > 1) {
				// Get the higher order update step
				dp_ho_step = run_data->w_hat[indx] + (dt * (RK5_Bs1 * RK_data->RK1[indx]) + dt * (RK5_Bs3 * RK_data->RK3[indx]) + dt * (RK5_Bs4 * RK_data->RK4[indx]) + dt * (RK5_Bs5 * RK_data->RK5[indx]) + dt * (RK5_Bs6 * RK_data->RK6[indx])) + dt * (RK5_Bs7 * RK_data->RK7[indx]));

				// Denominator in the error
				err_denom = DP_ABS_TOL + DPMax(cabs(RK_data->w_hat_last[indx]), cabs(run_data->w_hat[indx])) * DP_REL_TOL;

				// Compute the sum for the error
				err_sum += pow((run_data->w_hat[indx] - dp_ho_step) /  err_denom, 2.0);
			}
			#endif
		}
	}
	#ifdef __DPRK5
	if (iters > 1) {
		// Reduce and sync the error sum across the processes
		MPI_Allreduce(MPI_IN_PLACE, &err_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Compute the error
		RK_data->DP_err = sqrt(1.0/ (Nx * Ny_Fourier) * err_sum);

		// Record the Fourier vorticity for the next step
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Record the vorticity
				RK_data->w_hat_last[indx] = run_data->w_hat[indx];
			}
		}
	}
	#endif
}
#endif
#ifdef __DPRK5
/**
 * Function used to find the max between two numbers -> used in the Dormand Prince scheme
 * @param  a Double that will be used to find the max
 * @param  b Double that will be used to find the max
 * @return   The max between the two inputs
 */
double DPMax(double a, double b) {

	// Initailize max
	double max;

	// Check Max
	if (a > b) {
		max = a;
	}
	else {
		max = b;
	}

	// Return max
	return max;
}
/**
 * Function used to find the min between two numbers
 * @param  a Double that will be used to find the min
 * @param  b Double that will be used to find the min
 * @return   The minimum of the two inputs
 */
double DPMin(double a, double b) {

	// Initialize min
	double min;

	if (a < b) {
		min = a;
	}
	else {
		min = b;
	}

	// return the result
	return min;
}
#endif
/**
 * Function to perform one step using the 4th order Runge-Kutta method
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Nx Int indicating the local size of the first dimension of the arrays	
 * @param RK_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#ifdef __RK4
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {

	// Initialize vairables
	int tmp;
	int indx;
	#ifdef __NAVIER
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny_Fourier = N[1] / 2 + 1;


	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHSBatch(run_data->w_hat, RK_data->RK1, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK4_A21 * RK_data->RK1[indx];
		}
	}
	// ----------------------- Stage 2
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK2, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK4_A32 * RK_data->RK2[indx];
		}
	}
	// ----------------------- Stage 3
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK3, RK_data->nabla_psi, RK_data->nabla_w);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK4_A43 * RK_data->RK3[indx];
		}
	}
	// ----------------------- Stage 4
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK4, RK_data->nabla_psi, RK_data->nabla_w);
	
	
	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;


			#ifdef __EULER
			// Update Fourier vorticity with the RHS
			run_data->w_hat[indx] = run_data->w_hat[indx] + (dt * (RK4_B1 * RK_data->RK1[indx]) + dt * (RK4_B2 * RK_data->RK2[indx]) + dt * (RK4_B3 * RK_data->RK3[indx]) + dt * (RK4_B4 * RK_data->RK4[indx]));
			#elif defined(__NAVIER)
			// Compute the pre factors for the RK4CN update step
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// Both Hyperviscosity and Ekman drag
			D_fac = dt * (sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW)); 
			#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// No hyperviscosity but we have Ekman drag
			D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW)); 
			#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
			// Hyperviscosity only
			D_fac = dt * (sys_vars->NU * pow(k_sqr, VIS_POW)); 
			#else 
			// No hyper viscosity or no ekman drag -> just normal viscosity
			D_fac = dt * (sys_vars->NU * k_sqr); 
			#endif

			// Update temporary input for nonlinear term
			run_data->w_hat[indx] = run_data->w_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * RK_data->RK1[indx] + RK4_B2 * RK_data->RK2[indx] + RK4_B3 * RK_data->RK3[indx] + RK4_B4 * RK_data->RK4[indx]);
			#endif
		}
	}
}
#endif
/**
 * Function that performs the evluation of the nonlinear term by transforming back to real space where 
 * multiplication is perform before transforming back to Fourier space. Dealiasing is applied to the result
 * 
 * @param w_hat     Input array: contains the current vorticity of the system
 * @param dw_hat_dt Output array: Contains the result of the dealiased nonlinear term. Also used as an intermediate array to save memory
 * @param u         Array to hold the real space velocities
 * @param nabla_w   Array to hold the real space vorticity derivatives
 */
void NonlinearRHSBatch(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u, double* nabla_w) {

	// Initialize variables
	int tmp, indx;
	const ptrdiff_t local_Nx  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	fftw_complex k_sqr;
	double vel1;
	double vel2;

	// Allocate temporay memory for the nonlinear term in Real Space
	double* nonlinterm = (double* )malloc(sizeof(double) * local_Nx * (Ny + 2));


	// -----------------------------------
	// Compute Fourier Space Velocities
	// -----------------------------------
	// Compute (-\Delta)^-1 \omega - i.e., u_hat = d\psi_dy = -I kx/|k|^2 \omegahat_k, v_hat = -d\psi_dx = I ky/|k|^2 \omegahat_k
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				// denominator
				k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Fill fill fourier velocities array
				dw_hat_dt[SYS_DIM * (indx) + 0] = k_sqr * ((double) run_data->k[1][j]) * w_hat[indx];
				dw_hat_dt[SYS_DIM * (indx) + 1] = -1.0 * k_sqr * ((double) run_data->k[0][i]) * w_hat[indx];
			}
			else {
				dw_hat_dt[SYS_DIM * (indx) + 0] = 0.0 + 0.0 * I;
				dw_hat_dt[SYS_DIM * (indx) + 1] = 0.0 + 0.0 * I;
			}
			// printf("-k_sqr[%d, %d]: %1.16lf %1.16lf I \t kx[%d]: %1.16lf \t wh[%d, %d]: %1.16lf %1.16lf I \n", i, j, creal(-1.0 * k_sqr), cimag(-1.0 * k_sqr), i, ((double) run_data->k[0][i]), i, j, creal(w_hat[indx]), cimag(w_hat[indx]));
		}
	}
	// printf("\n");
	// PrintVectorFourier(dw_hat_dt, sys_vars->N, "uh", "vh");

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier velocites to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), dw_hat_dt, u);
	// PrintVectorReal(u, sys_vars->N, "u", "v");

	// ---------------------------------------------
	// Compute Fourier Space Vorticity Derivatives
	// ---------------------------------------------
	// Compute \nabla\omega - i.e., d\omegahat_dx = -I kx \omegahat_k, d\omegahat_dy = -I ky \omegahat_k
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Fill vorticity derivatives array
			dw_hat_dt[SYS_DIM * indx + 0] = I * ((double) run_data->k[0][i]) * w_hat[indx];
			dw_hat_dt[SYS_DIM * indx + 1] = I * ((double) run_data->k[1][j]) * w_hat[indx]; 
			// if (!(sys_vars->rank)) {
			// 	printf("dwdx_h[%d]: %1.16lf %1.16lf I \t dwdy_h[%d]: %1.16lf %1.16lf I\n", indx, creal(dw_hat_dt[SYS_DIM * indx + 0]), cimag(dw_hat_dt[SYS_DIM * indx + 0]), indx, creal(dw_hat_dt[SYS_DIM * indx + 1]), cimag(dw_hat_dt[SYS_DIM * indx + 1]));
			// }
		}
	}
	// printf("\n\n");
	// PrintVectorFourier(dw_hat_dt, sys_vars->N, "dwh_dx", "dwh_dy");

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier vorticity derivatives to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), dw_hat_dt, nabla_w);
	
	// PrintVectorReal(nabla_w, sys_vars->N, "dw_dx", "dw_dy");

	// -----------------------------------
	// Perform Convolution in Real Space
	// -----------------------------------
	// Perform the multiplication in real space
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j; 
 			
 			// Perform multiplication of the nonlinear term 
 			vel1 = u[SYS_DIM * indx + 0];
 			vel2 = u[SYS_DIM * indx + 1];
 			nonlinterm[indx] = 1.0 * (vel1 * nabla_w[SYS_DIM * indx + 0] + vel2 * nabla_w[SYS_DIM * indx + 1]);
 			// nonlinterm[indx] *= 1.0 / pow((Nx * Ny), 1.0);
 		}
 	}

 	// PrintScalarReal(nonlinterm, sys_vars->N, "RHS");



 	// -------------------------------------
 	// Transform Nonlinear Term To Fourier
 	// -------------------------------------
 	// Transform Fourier nonlinear term back to Fourier space
 	fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), nonlinterm, dw_hat_dt);

 	// PrintScalarFourier(dw_hat_dt, sys_vars->N, "b_RHSh");

 	for (int i = 0; i < local_Nx; ++i) {
 		tmp = i * (Ny_Fourier);
 		for (int j = 0; j < Ny_Fourier; ++j) {
 			indx = tmp + j;

 			dw_hat_dt[indx] *= 1.0 / pow((Nx * Ny), 2.0);
 		}
 	}
 	// -------------------------------------
 	// Apply Dealiasing to Nonlinear Term
 	// -------------------------------------
 	// Apply dealiasing and DFT normalization to the new nonlinear term
 	ApplyDealiasing(dw_hat_dt, 1, sys_vars->N);

 	// Free memory
 	free(nonlinterm);
}
/**
 * Function to apply the selected dealiasing filter to the input array. Can be Fourier vorticity or 
 * @param array    	The array containing the Fourier modes to dealiased
 * @param array_dim The extra array dimension -> will be 1 for scalar or 2 for vector
 * @param N        	Array containing the dimensions of the system
 */
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N) {

	// Initialize variables
	int tmp, indx;
	ptrdiff_t local_Nx        = sys_vars->local_Nx;
	const long int Nx         = N[0];
	const long int Ny         = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;
	#ifdef __DEALIAS_HOU_LI
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter and Normalization
	// --------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = array_dim * (tmp + j);

			#ifdef __DEALIAS_23
			if (sqrt((double) run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) > Nx / 3) {
				for (int l = 0; l < array_dim; ++l) {
					// Set dealised modes to 0
					array[indx + l] = 0.0 + 0.0 * I;	
				}
			}
			else {
				for (int l = 0; l < array_dim; ++l) {
					// Apply DFT normaliztin to undealiased modes
					array[indx + l] = array[indx];	
				}				
			}
			#elif __DEALIAS_HOU_LI
			// Compute Hou-Li filter
			hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Nx / 2), 2.0) + pow(run_data->k[1][j] / (Ny / 2), 2.0))), 36.0));

			for (int l = 0; l < array_dim; ++l) {
				// Apply filter and DFT normaliztion
				array[indx + l] *= hou_li_filter;
			}
			#endif
		}
	}
}	
/**
 * Function to compute the initial condition for the integration
 * @param w_hat Fourier space vorticity
 * @param u     Real space velocities in batch layout - both u and v
 * @param u_hat Fourier space velocities in batch layout - both u_hat and v_hat
 * @param N     Array containing the dimensions of the system
 */
void InitialConditions(fftw_complex* w_hat, double* u, fftw_complex* u_hat, const long int* N) {

	// Initialize variables
	int tmp, indx;
	const long int Nx         = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1; 

	// Initialize local variables 
	ptrdiff_t local_Nx = sys_vars->local_Nx;
	

	if(!(strcmp(sys_vars->u0, "TG_VEL"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = (tmp + j);

				// Fill the velocities
				u[SYS_DIM * indx + 0] = cos(KAPPA * run_data->x[0][i]) * sin(KAPPA * run_data->x[1][j]);
				u[SYS_DIM * indx + 1] = -sin(KAPPA * run_data->x[0][i]) * cos(KAPPA * run_data->x[1][j]);		
			}
		}

		// Transform velocities to Fourier space & dealias
		fftw_mpi_execute_dft_r2c(sys_vars->fftw_2d_dft_batch_r2c, u, u_hat);
		ApplyDealiasing(u_hat, 2, N);

		// -------------------------------------------------
		// Taylor Green Initial Condition - Fourier Space
		// -------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				w_hat[indx] = I * (run_data->k[0][i] * u_hat[SYS_DIM * (indx) + 1] - run_data->k[1][j] * u_hat[SYS_DIM * (indx) + 0]);
			}
		}

		// ---------------------------------------------
		// Apply deliasing and transform to Real space 	
		// ---------------------------------------------
		ApplyDealiasing(w_hat, 1, N);
		#ifdef __VORT_REAL
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), w_hat, run_data->w);
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double )(Nx * Ny);
			}
		}
		#endif
	}
	else if(!(strcmp(sys_vars->u0, "TG_VORT"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = (tmp + j);

				// Compute the vorticity of the Taylor Green vortex
				run_data->w[indx] = 2.0 * KAPPA * cos(KAPPA * run_data->x[0][i]) * cos(KAPPA * run_data->x[1][j]); 
			}
		}

		// ---------------------------------------
		// Transform to Fourier Space
		// ---------------------------------------
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), run_data->w, w_hat);
	}
	else if(!(strcmp(sys_vars->u0, "DOUBLE_SHEAR_LAYER"))) {
		// ------------------------------------------------
		// Double Shear Lyaer - Real Space Vorticity
		// ------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = (tmp + j);

				// Compute the vorticity of the double shear layer
				if (run_data->x[1][j] <= M_PI) {
					run_data->w[indx] = DELTA * cos(run_data->x[1][j]) - SIGMA / pow(cosh(SIGMA * (run_data->x[0][i] - 0.5 * M_PI)), 2.0); 
				}
				else {
					run_data->w[indx] = DELTA * cos(run_data->x[1][j]) + SIGMA / pow(cosh(SIGMA * (1.5 * M_PI - run_data->x[0][i])), 2.0); 
				}
			}
		}

		// ---------------------------------------
		// Transform to Fourier Space
		// ---------------------------------------
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), run_data->w, w_hat);
	}
	else if (!(strcmp(sys_vars->u0, "DECAY_TURB"))) {
		// ---------------------------------------
		// McWilliams - Decaying Turbulence IC
		// ---------------------------------------
		// Initialize memory for the Stream function
		fftw_complex* psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * local_Nx * Ny_Fourier);
		double k_sqr;
		double u1, u2;
		double rand1, rand2;
		double wnum_func;

		// ---------------------------------------
		// Initialize Foureir Stream Function
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;	

				// Generate two standard normal variables using Box-Muller transform
				u1 = (double)rand() / (double) RAND_MAX;
				u2 = (double)rand() / (double) RAND_MAX;
				rand1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
				rand2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

				if (run_data->k[0][i] == 0.0 && run_data->k[1][j] == 0.0) {
					// Compute |k|^2 and the wavenumber function
					wnum_func = 0.0;
				}
				else {
					// Compute |k|^2 and the wavenumber function
					k_sqr 	  = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
					wnum_func = sqrt(k_sqr) / (1.0 + pow((k_sqr / pow(K0, 2.0)), 2.0));
				}

				// Compute the Fourier stream function so that variance is proportional to |k| / (1.0 + (|k|/k_0)^4)
				psi_hat[indx] = rand1 * wnum_func + rand2 * wnum_func * I;
			}
		}

		// ---------------------------------------
		// Normalize the Initial Energy
		// ---------------------------------------
		double spec_energy = 0.0;
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;	

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Compute the spectral energy -> the k = 0 and Nx/2 have no conjugate so onlt count their contribution once
					spec_energy += cabs(psi_hat[indx]) / pow((Nx * Ny), 2.0);
				}
				else {
					// Compute the spectral energy
					spec_energy += 2.0 * cabs(psi_hat[indx]) / pow((Nx * Ny), 2.0);	
				}
			}
		}
		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &spec_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// ---------------------------------------
		// Compute the Fourier Vorticity
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Wavenumber prefactor -> |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Compute the Fouorier vorticity
				w_hat[indx] = k_sqr * psi_hat[indx] / spec_energy;
			}
		}

		// free memory
		fftw_free(psi_hat);

		// ---------------------------------------------
		// Apply deliasing and transform to Real space 	
		// ---------------------------------------------
		ApplyDealiasing(w_hat, 1, N);
		#ifdef __VORT_REAL
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), w_hat, run_data->w);
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double )(Nx * Ny);
			}
		}
		#endif
	}
	else if (!(strcmp(sys_vars->u0, "RANDOM"))) {
		// ---------------------------------------
		// Random Initial Conditions
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				w_hat[indx] = ((double)rand() / (double) RAND_MAX) * 2.0 * M_PI + ((double)rand() / (double) RAND_MAX)* 2.0 * M_PI * I;
			}
		}		

		// ---------------------------------------------
		// Apply deliasing and transform to Real space 	
		// ---------------------------------------------
		ApplyDealiasing(w_hat, 1, N);
		#ifdef __VORT_REAL
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), w_hat, run_data->w);
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double )(Nx * Ny);
			}
		}
		#endif
	}
	else if (!(strcmp(sys_vars->u0, "TESTING"))) {
		// Initialize temp variables
		double k_sqr;

		// ---------------------------------------
		// Powerlaw Amplitude & Fixed Phase
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)){
					// Fill zero modes
					w_hat[indx] = 0.0 + 0.0 * I;
				}
				else if (j == 0 && run_data->k[0][i] < 0 ) {
					// Amplitudes
					k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill vorticity - this is for the kx axis - to enfore conjugate symmetry
					w_hat[indx] = k_sqr * cexp(-I * M_PI / 4.0);
				}
				else {
					// Amplitudes
					k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill vorticity - fill the rest of the modes
					w_hat[indx] = k_sqr * cexp(I * M_PI / 4.0);
				}
			}
		}

		// ---------------------------------------------
		// Apply deliasing and transform to Real space 	
		// ---------------------------------------------
		ApplyDealiasing(w_hat, 1, N);
		#ifdef __VORT_REAL
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), w_hat, run_data->w);
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double )(Nx * Ny);
			}
		}
		#endif
	}
	else {
		printf("\n["MAGENTA"WARNING"RESET"] --- No initial conditions specified\n---> Using random initial conditions...\n");
		// ---------------------------------------
		// Random Initial Conditions
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				w_hat[indx] = ((double)rand() / (double) RAND_MAX) * 2.0 * M_PI + ((double)rand() / (double) RAND_MAX) * 2.0 * M_PI * I;
			}
		}		

		// ---------------------------------------------
		// Apply deliasing and transform to Real space 	
		// ---------------------------------------------
		#ifdef __VORT_REAL
		ApplyDealiasing(w_hat, 1, N);
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), w_hat, run_data->w);
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double )(Nx * Ny);
			}
		}
		#endif
	}
	// -------------------------------------------------
	// Initialize the Dealiasing
	// -------------------------------------------------
	if (strcmp(sys_vars->u0, "TESTING") || strcmp(sys_vars->u0, "TG_VEL")) {
		ApplyDealiasing(w_hat, 1, N);
	}
}
/**
 * Function to initialize the Real space collocation points arrays and Fourier wavenumber arrays
 * 
 * @param x Array containing the collocation points in real space
 * @param k Array to contain the wavenumbers on both directions
 * @param N Array containging the dimensions of the system
 */
void InitializeSpaceVariables(double** x, int** k, const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Initialize local variables 
	ptrdiff_t local_Nx       = sys_vars->local_Nx;
	ptrdiff_t local_Nx_start = sys_vars->local_Nx_start;
	

	// -------------------------------
	// Fill the first dirction 
	// -------------------------------
	int j = 0;
	for (int i = 0; i < Nx; ++i) {
		if((i >= local_Nx_start) && ( i < local_Nx_start + local_Nx)) { // Ensure each process only writes to its local array slice
			x[0][j] = (double) i * 2.0 * M_PI / (double) Nx;
			j++;
		}
	}
	j = 0;
	for (int i = 0; i < local_Nx; ++i) {
		if (local_Nx_start + i <= Nx / 2) {   // Set the first half of array to the positive k
			k[0][j] = local_Nx_start + i;
			j++;
		}
		else if (local_Nx_start + i > Nx / 2) { // Set the second half of array to the negative k
			k[0][j] = local_Nx_start + i - Nx;
			j++;
		}
	}

	// -------------------------------
	// Fill the second dirction 
	// -------------------------------
	for (int i = 0; i < Ny; ++i) {
		if (i < Ny_Fourier) {
			k[1][i] = i;
		}
		x[1][i] = (double) i * 2.0 * M_PI / (double) Ny;
	}
}
/**
 * Function used to compute of either the velocity or vorticity
 * @return  Returns the computed maximum
 */
double GetMaxData(char* dtype) {

	// Initialize variables
	const long int Nx         = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	int tmp;
	int indx;
	fftw_complex k_sqr;

	// -------------------------------
	// Compute the Data 
	// -------------------------------
	if (strcmp(dtype, "VEL") == 0) {
		// Compute the velocity in Fourier space
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
					// Compute the prefactor				
					k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// compute the velocity in Fourier space
					run_data->u_hat[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * k_sqr * run_data->w_hat[indx];
					run_data->u_hat[SYS_DIM * indx + 1] = -1.0 * ((double) run_data->k[0][i]) * k_sqr * run_data->w_hat[indx];
				}
				else {
					run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
				}
			}
		}

		// Transform back to Fourier space
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat, run_data->u);
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->u[SYS_DIM * indx + 0] *= 1.0 / (double )(Nx * Ny);
				run_data->u[SYS_DIM * indx + 1] *= 1.0 / (double )(Nx * Ny);
			}
		}
	}
	else if (strcmp(dtype, "VORT") == 0) {
		// Perform transform back to real space for the vorticity
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->w_hat, run_data->w);
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0 / (double )(Nx * Ny);
			}
		}
	}

	// -------------------------------
	// Find the maximum
	// -------------------------------
	// Define the maximum
	double wmax = 0.0;

	// Loop over array to find the maximum
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		// Find the max of the Real Space vorticity
		if (strcmp(dtype, "VORT_FOUR") == 0) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Check if maximum
				if (cabs(run_data->w_hat[indx]) > wmax) {
					wmax = cabs(run_data->w_hat[indx]);
				}	
				else {
					continue;
				}
			}			
		}
		// Find the max of the Fourier Vorticity
		else if (strcmp(dtype, "VORT") == 0) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Check if maximum
				if (cabs(run_data->w[indx]) > wmax) {
					wmax = cabs(run_data->w[indx]);
				}	
				else {
					continue;
				}
			}			
		}
		// Find the max of the Real space velocity
		else if (strcmp(dtype, "VEL") == 0) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				for (int l = 0; l < SYS_DIM; ++l) {
					indx = tmp + j;

					// Check if maximum
					if (cabs(run_data->u[SYS_DIM * indx + l]) > wmax) {
						wmax = cabs(run_data->u[SYS_DIM * indx + l]);
					}	
					else {
						continue;
					}
				}
			}
		}
	}

	// Now synchronize the maximum over all process
	MPI_Allreduce(MPI_IN_PLACE, &wmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	
	return wmax;
}
/**
 * Function that checks the system to see if it is ok to continue integrations. Checks for blow up, timestep and iteration limits etc
 * @param dt    The updated timestep for the next iteration
 * @param iters The number of iterations for the next iteration
 */
void SystemCheck(double dt, int iters) {

	// Initialize variables
	double w_max;	

	// -------------------------------
	// Get Current Max Vorticity 
	// -------------------------------
	w_max = GetMaxData("VORT");

	// -------------------------------
	// Check Stopping Criteria 
	// -------------------------------
	if (w_max >= MAX_VORT_LIM)	{
		fprintf(stderr, "\n["YELLOW"SOVLER FAILURE"RESET"] --- System has reached maximum Vorticity limt at Iter: ["CYAN"%d"RESET"]\n-->> Exiting!!!n", iters);
		exit(1);
	}
	else if (dt <= MIN_STEP_SIZE) {
		fprintf(stderr, "\n["YELLOW"SOVLER FAILURE"RESET"]--- Timestep has become too small to continue at Iter: ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", iters);
		exit(1);		
	}
	else if (iters >= MAX_ITERS) {
		fprintf(stderr, "\n["YELLOW"SOVLER FAILURE"RESET"]--- The maximum number of iterations has been reached at Iter: ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", iters);
		exit(1);		
	}
}
/**
 * Function to update the timestep if adaptive timestepping is enabled
 * @param dt The current timestep
 */
void GetTimestep(double* dt) {

	// Initialize variables
	int tmp;
	int indx;
	
	// -------------------------------
	// Compute New Timestep
	// -------------------------------
	#ifdef __CFL_STEP
	fftw_complex k_sqr;
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	
	// -------------------------------
	// Compute the Velocity
	// -------------------------------
	// Compute the velocity in Fourier space
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * sys_vars->N[1] / 2 + 1;
		for (int j = 0; j < sys_vars->N[1] / 2 + 1; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				// Compute the prefactor				
				k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// compute the velocity in Fourier space
				run_data->u_hat[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * k_sqr * run_data->w_hat[indx];
				run_data->u_hat[SYS_DIM * indx + 1] = -1.0 * ((double) run_data->k[0][i]) * k_sqr * run_data->w_hat[indx];
			}
			else {
				// Get the zero mode
				run_data->u_hat[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
			}
		}
	}

	// Transform back to Fourier space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat, run_data->u);
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Normalize
			run_data->u[SYS_DIM * indx + 0] *= 1.0 / (double )(Nx * Ny);
			run_data->u[SYS_DIM * indx + 1] *= 1.0 / (double )(Nx * Ny);
		}
	}

	// -------------------------------
	// Find the Times Scales 
	// -------------------------------
	// Find the convective scales
	double scales_convect = 0.0;
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (sys_vars->N[1] + 2);
		for (int j = 0; j < sys_vars->N[1]; ++j) {
			indx = tmp + j;

			scales_convect = fmax(M_PI * (fabs(run_data->u[SYS_DIM * indx + 0]) / sys_vars->dx + fabs(run_data->u[SYS_DIM * indx + 1]) / sys_vars->dy), scales_convect);
		}
	}
	// Gather the maximum of the convective scales
	MPI_Allreduce(MPI_IN_PLACE, &scales_convect, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	// Find the diffusive scales
	double scales_diff = pow(M_PI, 2.0) * (sys_vars->NU / pow(sys_vars->dx, 2.0) + sys_vars->NU / pow(sys_vars->dy, 2.0));

	// -------------------------------
	// Update Timestep
	// -------------------------------
	(* dt) = sys_vars->CFL_CONST / (scales_convect + scales_diff);
	#else
	double dt_new;
	double w_max;


	// -------------------------------
	// Get Current Max Vorticity 
	// -------------------------------
	w_max = GetMaxData("VORT");

	// Find proposed timestep = h_0 * (max{w_hat(0)} / max{w_hat(t)}) -> this ensures that the maximum vorticity by the timestep is constant
	dt_new = (sys_vars->dt) * (sys_vars->w_max_init / w_max);	

	// Gather new timesteps from all processes -> pick the smallest one
	MPI_Allreduce(MPI_IN_PLACE, &dt_new, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	
	// -------------------------------
	// Update Timestep
	// -------------------------------
	// Check if new timestep meets approved criteria
	if (dt_new > 2.0 * (*dt)) {  
		// Timestep can be increased by no more than twice current timestep
		(*dt) = 2.0 * (*dt); 
	}
	else {
		// Update with new timestep - new timestep is checked for criteria in SystemsCheck() function 
		(*dt) = dt_new;
	}
	#endif
}
/**
 * Function to test the code against the Taylor-Green vortex solution
 * @param t The current time of the system - used to compute the exact solution
 * @param N Array containing the dimensions of the system
 * @return  Returns a pointer to an array holding the L2 and Linf norms
 */
void TestTaylorGreenVortex(const double t, const long int* N, double* norms) {

	// Initialize variables
	int tmp;
	int indx;
	const long int Nx         = N[0];
	const long int Ny         = N[1];
	double norm_const = 1.0 / (double)(Nx * Ny);
	double linf_norm  = 0.0;
	double l2_norm    = 0.0;
	double tg_exact;
	double abs_err;

	// --------------------------------
	// Get the Real Space Vorticity
	// --------------------------------
	// Transform back to Fourier space -> Don't normalize now - normalize in loop below
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->w_hat, run_data->w);
	
	// --------------------------------
	// Compute The Error
	// --------------------------------
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			if (!(strcmp(sys_vars->u0, "TG_VEL"))) {
				// Compute the exact solution
				tg_exact = -2.0 * KAPPA * sin(KAPPA * run_data->x[0][i]) * sin(KAPPA * run_data->x[1][j]) * exp(-2.0 * pow(KAPPA, 2.0) * sys_vars->NU * t);
			}
			else {
				// Compute the exact solution
				tg_exact = 2.0 * KAPPA * cos(KAPPA * run_data->x[0][i]) * cos(KAPPA * run_data->x[1][j]) * exp(-2.0 * pow(KAPPA, 2.0) * sys_vars->NU * t);
			}
			

			// Get the absolute error
			abs_err = fabs(run_data->w[indx] * norm_const - tg_exact);

			// Update L2-norm sum
			l2_norm += pow(abs_err, 2.0);

			// Update the Linf-norm 
			linf_norm = fmax(abs_err, linf_norm);
		}
	}	

	// --------------------------------
	// Return results to Root
	// --------------------------------
	if (!sys_vars->rank) {
		// Gather the norms from each process
		MPI_Reduce(MPI_IN_PLACE, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, &linf_norm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		// Compute the L2-norm
		l2_norm = sqrt(norm_const * l2_norm);

		// Save in norm array
		norms[0] = l2_norm;
		norms[1] = linf_norm;
	}
	else {
		// Send and reduce the norms from each process to root
		MPI_Reduce(&l2_norm, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&linf_norm, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	}
}
/**
 * Function used to compute the Taylor Green solution at the current iteration
 * @param t The current time in the iteration
 * @param N Array containing the dimensions of the system
 */
void TaylorGreenSoln(const double t, const long int* N) {

	// Initialize variables
	int tmp;
	int indx;
	const long int Ny = N[1];
	
	// --------------------------------
	// Compute The Taylor Green Soln
	// --------------------------------
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			if (!(strcmp(sys_vars->u0, "TG_VEL"))) {
				// Compute the exact solution
				run_data->tg_soln[indx] = -2.0 * KAPPA * sin(KAPPA * run_data->x[0][i]) * sin(KAPPA * run_data->x[1][j]) * exp(-2.0 * pow(KAPPA, 2.0) * sys_vars->NU * t);
			}
			else {
				// Compute the exact solution
				run_data->tg_soln[indx] = 2.0 * KAPPA * cos(KAPPA * run_data->x[0][i]) * cos(KAPPA * run_data->x[1][j]) * exp(-2.0 * pow(KAPPA, 2.0) * sys_vars->NU * t);
			}
		}
	}

}
/**
 * Function used to compute the energy spectrum of the current iteration. The energy spectrum is defined as all(sum) of the energy contained in concentric annuli in
 * wavenumber space. 	
 * @param  spectrum_size Variable used to store the size of the spectrum
 * @return               Pointer to the computed spectrum
 */
double* EnergySpectrum(int* spectrum_size) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	double u_hat;
	double v_hat;
	fftw_complex k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------
	// Allocate Memory
	// ------------------------------------
	// Size of the spectrum
	(* spectrum_size) = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	
	// Allocate memory and initialize
	double* spectrum = (double* )fftw_malloc(sizeof(double) * (* spectrum_size));
	if (spectrum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the spectrum array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Spectrum");
		exit(1);
	}
	for (int i = 0; i < (* spectrum_size); ++i) {
		spectrum[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute the prefactor
				k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Compute Fourier velocities
				u_hat = k_sqr * ((double) run_data->k[1][j]) * run_data->w_hat[indx];
				v_hat = -k_sqr * ((double) run_data->k[0][i]) * run_data->w_hat[indx];
			}
			else {
				// Compute the zero mode
				u_hat = 0.0 + 0.0 * I;
				v_hat = 0.0 + 0.0 * I;
			}
			
			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

			if ((j == 0) && (j = Ny_Fourier - 1)) {
				// Update the energy sum for the current mode
				spectrum[spec_indx] += cabs(u_hat * conj(u_hat)) + cabs(v_hat * conj(v_hat)) * norm_fac;
			}
			else {
				// Update the energy sum for the current mode
				spectrum[spec_indx] += 2.0 * cabs(u_hat * conj(u_hat)) + cabs(v_hat * conj(v_hat)) * norm_fac;
			}

		}
	}

	// Return spectrum
	return spectrum;
}
/**
 * Function used to compute the enstrophy spectrum for the current iteration. The enstrophy spectrum is defined as the total enstrophy contained in concentric annuli 
 * in wavenumber space
 * @param  spectrum_size The size of the enstrophy spectrum
 * @return               A pointer to the enstrophy spectrum
 */
double* EnstrophySpectrum(int* spectrum_size) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------
	// Allocate Memory
	// ------------------------------------
	// Size of the spectrum
	(* spectrum_size) = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	
	// Allocate memory and initialize
	double* spectrum = (double* )fftw_malloc(sizeof(double) * (* spectrum_size));
	if (spectrum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the spectrum array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Spectrum");
		exit(1);
	}
	for (int i = 0; i < (* spectrum_size); ++i) {
		spectrum[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

			if ((j == 0) && (j == Ny_Fourier - 1)) {
				// Update the sum of the enstrophy in the current mode
				spectrum[spec_indx] += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * norm_fac;
			}
			else {
				// Update the sum of the enstrophy in the current mode
				spectrum[spec_indx] += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * norm_fac;
			}
		}
	}

	// Return the spectrum
	return spectrum;
}
/**
 * Function to compute the total energy in the system at the current timestep
 * @return  The total energy in the system
 */
double TotalEnergy(void) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double tot_energy = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// ------------------------------------------
	// Compute Fourier Space Velocity & Energy
	// ------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// The 1/k^2 prefactor
				k_sqr = 1.0 / (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				if ((j == 0) && (j == Ny_Fourier - 1)) {
					// Update the sum for the total energy
					tot_energy += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
				else {
					// Update the sum for the total energy
					tot_energy += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
			}
			else {
				tot_energy += 0.0;
			}
		}
	}
	
	// Return result
	return 0.5 * tot_energy / pow(Nx * Ny, 2.0);
}
/**
 * Function to compute the total enstrophy in the system at the current timestep
 * @return  The total enstrophy in the system
 */
double TotalEnstrophy(void) {

	// Initialize variables
	int tmp;
	int indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;


	// -------------------------------
	// Compute The Total Energy 
	// -------------------------------
	// Initialize total enstrophy
	double tot_enstr = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 1; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((j == 0) && (j == Ny_Fourier - 1)) {
				// Update the sum for the total enstrophy -> only count the 0 and N/2 modes once as they have no conjugate
				tot_enstr += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}	
			else {
				// Update the sum for the total enstrophy -> factor of two for Fourier conjugates
				tot_enstr += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}	
		}
	}

	// Return result
	return 0.5 * tot_enstr / pow(Nx * Ny, 2.0);
}
/**
 * Function to compute the total palinstrophy at the current timestep on the local process.
 * Results are gathered on the master process at the end of the simulation run
 * @return  The total palinstrophy
 */
double TotalPalinstrophy(void) {
	
	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;


	// -------------------------------
	// Compute The Total Palinstrophy 
	// -------------------------------
	// Initialize total enstrophy
	double tot_palin = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Get the |k|^2 prefactor
				k_sqr = 1.0 * (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Update the running sum for the palinstrophy
				if((j == 0) && (j == Ny_Fourier - 1)) {
					tot_palin += k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					tot_palin += 2.0 * k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
			else {
				tot_palin += 0.0;
			}
		}
	}

	// Return result
	return 0.5 * tot_palin / pow(Nx * Ny, 2.0);
}
/**
 * Function to compute the energy dissipation rate \epsilon for the current iteration on the local processes
 * The results are gathered on the master process at the end of the simulation
 * @return  Returns the energy dissipation rate
 */
double EnergyDissipationRate(void) {

	// Initialize variables
	int tmp;
	int indx;
	double pre_fac;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -------------------------------
	// Compute The Energy Diss Rate
	// -------------------------------
	// Initialize total enstrophy
	double enrgy = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get the appropriate prefactor
			#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// Both Hyperviscosity and Ekman drag
			pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
			#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// No hyperviscosity but we have Ekman drag
			pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
			#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
			// Hyperviscosity only
			pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
			#else 
			// No hyper viscosity or no ekman drag -> just normal viscosity
			pre_fac = sys_vars->NU; 
			#endif

			// Update the running sum for the palinstrophy -> first and last modes have no conjugate so only count once
			if((j == 0) && (j == Ny_Fourier - 1)) {
				enrgy += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}
			else {
				enrgy += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}
		}
	}

	// Return result -> 2 (nu * 0.5 *<|w|^2>)
	return enrgy / pow(Nx * Ny, 2.0);
}
/**
 * Function to compute the enstrophy dissipation rate \eta which is equal to 2 * Palinstrophy for the current iteration on the local process
 * Results are gathered on the master process at the end of the simulation run
 * @return  The enstrophy dissipation rate on the local process
 */
double EnstrophyDissipationRate(void) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double pre_fac;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -------------------------------
	// Compute The Enstrophy Diss Rate 
	// -------------------------------
	// Initialize total enstrophy
	double tot_palin = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = 1.0 * (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// Both Hyperviscosity and Ekman drag
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// No hyperviscosity but we have Ekman drag
				pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
				// Hyperviscosity only
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
				#else 
				// No hyper viscosity or no ekman drag -> just normal viscosity
				pre_fac = sys_vars->NU * k_sqr; 
				#endif

				// Update the running sum for the palinstrophy -> first and last modes have no conjugate so only count once
				if((j == 0) && (j == Ny_Fourier - 1)) {
					tot_palin += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					tot_palin += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
			else {
				tot_palin += 0.0;
			}
		}
	}


	// Return result -> 2(palin) = 2(0.5 * <|grad \omega|^2>)
	return tot_palin / pow(Nx * Ny, 2.0);
}	
/**
 * Function to compute the enstrophy flux and enstrophy dissipation from a subset of modes on the local process for the current iteration 
 * The results will be gathered on the master process at the end of the simulation	
 * @param enst_flux The enstrophy flux 
 * @param enst_diss The enstrophy dissipation
 * @param RK_data   The Runge-Kutta struct containing the arrays for computing the nonlinear term
 */
void EnstrophyFlux(double* enst_flux, double* enst_diss, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double pre_fac;
	double tmp_enst_flux;
	double tmp_enst_diss;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Flux");
		exit(1);
	}

	// Compute the nonlinear term
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);

	// -------------------------------------
	// Compute the Enstrophy Flux & Diss
	// -------------------------------------
	// Initialize sums
	tmp_enst_flux = 0.0;
	tmp_enst_diss = 0.0;

	double tmp_flux = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Consider only a subset of modes
			if (((run_data->k[1][j] >= LWR_SBST_LIM) && (run_data->k[1][j] <= UPR_SBST_LIM)) && (abs(run_data->k[0][i]) <= UPR_SBST_LIM) && (abs(run_data->k[0][i]) >= LWR_SBST_LIM)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// Both Hyperviscosity and Ekman drag
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// No hyperviscosity but we have Ekman drag
				pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
				// Hyperviscosity only
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
				#else 
				// No hyper viscosity or no ekman drag -> just normal viscosity
				pre_fac = sys_vars->NU * k_sqr; 
				#endif

				// Update sums
				if ((j == 0) && (Ny_Fourier - 1)) {
					// Update the running sum for the flux of enstrophy
					tmp_enst_flux += creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]); 

					// Update the running sum for the enstrophy dissipation 
					tmp_enst_diss += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					// Update the running sum for the flux of enstrophy
					tmp_enst_flux += 2.0 * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);

					// Update the running sum for the enstrophy dissipation 
					tmp_enst_diss += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
		}
	}

	// -----------------------------------
	// Compute the Enstrophy Flux & Diss 
	// -----------------------------------
	// Compue the enstrophy dissipation
	(*enst_diss) = 0.5 * tmp_enst_diss / pow(Nx * Ny, 2.0);

	// Compute the enstrophy flux
	(*enst_flux) = 0.5 * tmp_enst_flux / pow(Nx * Ny, 2.0);

	// -----------------------------------
	// Free memory
	// -----------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the energy flux and dissipation in/out of a subset of modes for the current iteration on the local processes
 * The results will be gathered on the master process and written to file at the end of the simulation
 * @param enrg_flux The energy flux in/out of a subset of modes	
 * @param enrg_diss The energy dissipation of a subset of modes
 * @param RK_data   The struct containing the Runge-Kutta arrays to be used for computing the nonlinear term
 */
void EnergyFlux(double* enrg_flux, double* enrg_diss, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	double pre_fac;
	double tmp_enrgy_flux;
	double tmp_enrgy_diss;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Flux");
		exit(1);
	}

	// Compute the nonlinear term
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);

	// -------------------------------------
	// Compute the Energy Flux & Diss
	// -------------------------------------
	// Initialize sums
	tmp_enrgy_flux = 0.0;
	tmp_enrgy_diss = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Consider only a subset of modes
				if (((run_data->k[1][j] >= LWR_SBST_LIM) && (run_data->k[1][j] <= UPR_SBST_LIM)) && (abs(run_data->k[0][i]) <= UPR_SBST_LIM) && (abs(run_data->k[0][i]) >= LWR_SBST_LIM)) {
					// Compute |k|^2
					k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Get the appropriate prefactor
					#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
					// Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
					#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
					// No hyperviscosity but we have Ekman drag
					pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
					#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
					// Hyperviscosity only
					pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
					#else 
					// No hyper viscosity or no ekman drag -> just normal viscosity
					pre_fac = sys_vars->NU * k_sqr; 
					#endif

					// Update sums
					if ((j == 0) && (Ny_Fourier - 1)) {
						// Update the running sum for the flux of energy
						tmp_enrgy_flux += creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;

						// Update the running sum for the energy dissipation 
						tmp_enrgy_diss += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) / k_sqr;
					}
					else {
						// Update the running sum for the flux of energy
						tmp_enrgy_flux += 2.0 * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;

						// Update the running sum for the energy dissipation 
						tmp_enrgy_diss += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) / k_sqr;
					}
				}
			}
			else {
				tmp_enrgy_flux += 0.0;
				tmp_enrgy_diss += 0.0;
			}
		}
	}

	// -----------------------------------
	// Compute the Energy Flux & Diss 
	// -----------------------------------
	// Compue the energy dissipation
	(*enrg_diss) = 0.5 * tmp_enrgy_diss / pow(Nx * Ny, 2.0);

	// Compute the energy flux
	(*enrg_flux) = 0.5 * tmp_enrgy_flux / pow(Nx * Ny, 2.0);


	// -----------------------------------
	// Free memory
	// -----------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the energy flux spectrum for the current iteration on the local processes
 * The results of the complete spectrum will be gathered on the master rank before writing to file
 * @param  n_spect The size of the spectrum
 * @param  RK_data The struct containing the Runge-Kutta arrays needed for computing the nonlinear term
 * @return         Returns a pointer to the computed energy flux spectrum
 */
double* EnergyFluxSpectrum(int* n_spect, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	double pre_fac;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------
	// Allocate Memory
	// ------------------------------------
	// Size of the spectrum
	(* n_spect) = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	
	// Allocate memory & initialize
	double* spectrum = (double* )fftw_malloc(sizeof(double) * (* n_spect));
	if (spectrum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the spectrum array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}
	for (int i = 0; i < (* n_spect); ++i) {
		spectrum[i] = 0.0;
	}

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear term array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}

	// Compute the nonlinear term
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);

	// -------------------------------------
	// Compute the Energy Flux Spectrum
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// Both Hyperviscosity and Ekman drag
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// No hyperviscosity but we have Ekman drag
				pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
				// Hyperviscosity only
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
				#else 
				// No hyper viscosity or no ekman drag -> just normal viscosity
				pre_fac = sys_vars->NU * k_sqr; 
				#endif

				// Get the spectrum index
				spec_indx = (int) sqrt(k_sqr);

				// Update spectrum bin
				if ((j == 0) && (Ny_Fourier - 1)) {
					// Update the running sum for the flux of energy
					spectrum[spec_indx] += norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;
				}
				else {
					// Update the running sum for the flux of energy
					spectrum[spec_indx] += 2.0 * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;
				}
			}
		}
	}

	// -------------------------------------
	// Free Temp Memory
	// -------------------------------------
	fftw_free(dwhat_dt);

	// -------------------------------------
	// Return Spectrum
	// -------------------------------------
	return spectrum;
}
/**
 * Function to compute the enstrophy flux spectrum for the current iteration on the local processes
 * The results are gathered on the master rank before being written to file
 * @param  n_spect The size of the spectrum
 * @param  RK_data The struct containing the Runge-Kutta arrays for computing the nonlinear term
 * @return         Returns a pointer to the computed spectrum
 */
double* EnstrophyFluxSpectrum(int* n_spect, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	double pre_fac;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------
	// Allocate Memory
	// ------------------------------------
	// Size of the spectrum
	(* n_spect) = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	
	// Allocate memory & initialize
	double* spectrum = (double* )fftw_malloc(sizeof(double) * (* n_spect));
	if (spectrum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the spectrum array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}
	for (int i = 0; i < (* n_spect); ++i) {
		spectrum[i] = 0.0;
	}

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear term array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}

	// Compute the nonlinear term
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);

	// -------------------------------------
	// Compute the Energy Flux Spectrum
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the appropriate prefactor
				#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// Both Hyperviscosity and Ekman drag
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
				// No hyperviscosity but we have Ekman drag
				pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
				#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
				// Hyperviscosity only
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
				#else 
				// No hyper viscosity or no ekman drag -> just normal viscosity
				pre_fac = sys_vars->NU * k_sqr; 
				#endif

				// Get the spectrum index
				spec_indx = (int) sqrt(k_sqr);

				// Update spectrum bin
				if ((j == 0) && (Ny_Fourier - 1)) {
					// Update the running sum for the flux of energy
					spectrum[spec_indx] += norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);
				}
				else {
					// Update the running sum for the flux of energy
					spectrum[spec_indx] += 2.0 * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);
				}
			}
		}
	}

	// -------------------------------------
	// Free Temp Memory
	// -------------------------------------
	fftw_free(dwhat_dt);

	// -------------------------------------
	// Return Spectrum
	// -------------------------------------
	return spectrum;
}
/**
 * Function to record the system measures for the current timestep 
 * @param t          The current time in the simulation
 * @param print_indx The current index of the measurables arrays
 * @param RK_data 	 The Runge-Kutta struct containing the arrays to compute the nonlinear term for the fluxes
 */
void RecordSystemMeasures(double t, int print_indx, RK_data_struct* RK_data) {

	// -------------------------------
	// Record the System Measures 
	// -------------------------------
	// The integration time
	#ifdef __TIME
	if (!(sys_vars->rank)) {
		run_data->time[print_indx] = t;
	}
	#endif

	// Check if within memory limits
	if (print_indx < sys_vars->num_print_steps) {
		// Total Energy, enstrophy and palinstrophy
		run_data->tot_enstr[print_indx]  = TotalEnstrophy();
		run_data->tot_energy[print_indx] = TotalEnergy();
		run_data->tot_palin[print_indx]  = TotalPalinstrophy();
		// Energy and enstrophy dissipation rates
		run_data->enrg_diss[print_indx] = EnergyDissipationRate();
		run_data->enst_diss[print_indx] = EnstrophyDissipationRate();
		// Enstrophy and energy flux in/out and dissipation of a subset of modes
		EnstrophyFlux(&(run_data->enst_flux_sbst[print_indx]), &(run_data->enst_diss_sbst[print_indx]), RK_data);
		EnergyFlux(&(run_data->enrg_flux_sbst[print_indx]), &(run_data->enrg_diss_sbst[print_indx]), RK_data);
	}
	else {
		printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", print_indx, t);
	}
	
	// -------------------------------
	// Record the Spectra 
	// -------------------------------
	#ifdef __SPECT 
	// Call spectra functions
	run_data->enst_spect 	  = (double* )EnstrophySpectrum(&(sys_vars->n_spect));
	run_data->enrg_spect 	  = (double* )EnergySpectrum(&(sys_vars->n_spect));
	run_data->enrg_flux_spect = (double* )EnergyFluxSpectrum(&(sys_vars->n_spect), RK_data);
	run_data->enst_flux_spect = (double* )EnstrophyFluxSpectrum(&(sys_vars->n_spect), RK_data);
	#endif
}
/**
 * Wrapper function used to allocate memory all the nessecary local and global system and integration arrays
 * @param NBatch  Array holding the dimensions of the Fourier space arrays
 * @param RK_data Pointer to struct containing the integration arrays
 */
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data) {

	// Initialize variables
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;	

	// -------------------------------
	// Get Local Array Sizes - FFTW 
	// -------------------------------
	//  Find the size of memory for the FFTW transforms - use these to allocate appropriate memory
	sys_vars->alloc_local       = fftw_mpi_local_size_2d(Nx, Ny_Fourier, MPI_COMM_WORLD, &(sys_vars->local_Nx), &(sys_vars->local_Nx_start));
	sys_vars->alloc_local_batch = fftw_mpi_local_size_many((int)SYS_DIM, NBatch, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &(sys_vars->local_Nx), &(sys_vars->local_Nx_start));
	if (sys_vars->local_Nx == 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- FFTW was unable to allocate local memory for each process -->> Code will run but will be slow\n");
	}
	
	// -------------------------------
	// Allocate Space Variables 
	// -------------------------------
	// Allocate the wavenumber arrays
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->local_Nx);  // kx
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * Ny_Fourier);     		// ky
	if (run_data->k[0] == NULL || run_data->k[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "wavenumber list");
		exit(1);
	}

	// Allocate the collocation points
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->local_Nx);  // x direction 
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * Ny);     			  // y direction
	if (run_data->x[0] == NULL || run_data->x[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "collocation points");
		exit(1);
	}
	// -------------------------------
	// Allocate System Variables 
	// -------------------------------
	// Allocate the Real and Fourier space vorticity
	run_data->w     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Space Vorticity" );
		exit(1);
	}
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Space Vorticity");
		exit(1);
	}

	// Allocate the Real and Fourier space velocities
	run_data->u     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Space Velocities");
		exit(1);
	}
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Space Velocities");
		exit(1);
	}
	#ifdef TESTING
	run_data->tg_soln = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->tg_soln == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Taylor Green vortex solution");
		exit(1);
	}
	#endif

	// -------------------------------
	// Allocate Integration Variables 
	// -------------------------------
	// Runge-Kutta Integration arrays
	RK_data->nabla_psi = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->nabla_psi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "nabla_psi");
		exit(1);
	}
	RK_data->nabla_w   = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->nabla_w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "nabla_w");
		exit(1);
	}
	RK_data->RK1       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK1 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK1");
		exit(1);
	}
	RK_data->RK2       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK2 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK2");
		exit(1);
	}
	RK_data->RK3       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK3 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK3");
		exit(1);
	}
	RK_data->RK4       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK4 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK4");
		exit(1);
	}
	RK_data->RK_tmp    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK_tmp");
		exit(1);
	}
	#ifdef __RK5
	RK_data->RK5       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK5 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK5");
		exit(1);
	}
	RK_data->RK6       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK6 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK6");
		exit(1);
	}
	#endif
	#ifdef __DPRK5
	RK_data->RK7       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK7 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK7");
		exit(1);
	}
	RK_data->w_hat_last = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->w_hat_last == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "w_hat_last");
		exit(1);
	}
	#endif

	// -------------------------------
	// Initialize All Data 
	// -------------------------------
	int tmp_real, tmp_four;
	int indx_real, indx_four;
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp_real = i * (Ny + 2);
		tmp_four = i * Ny_Fourier;
		
		for (int j = 0; j < Ny; ++j){
			indx_real = tmp_real + j;
			indx_four = tmp_four + j;
			
			run_data->u[SYS_DIM * indx_real + 0]        = 0.0;
			run_data->u[SYS_DIM * indx_real + 1] 	  	= 0.0;
			RK_data->nabla_w[SYS_DIM * indx_real + 0] 	= 0.0;
			RK_data->nabla_w[SYS_DIM * indx_real + 1] 	= 0.0;
			RK_data->nabla_psi[SYS_DIM * indx_real + 0] = 0.0;
			RK_data->nabla_psi[SYS_DIM * indx_real + 1] = 0.0;
			#ifdef TESTING
			run_data->tg_soln[indx_real]                = 0.0;
			#endif
			run_data->w[indx_real]                      = 0.0;
			if (j < Ny_Fourier) {
				run_data->w_hat[indx_four]               = 0.0 + 0.0 * I;
				RK_data->RK_tmp[indx_four]    			 = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 1] = 0.0 + 0.0 * I;
				RK_data->RK1[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK1[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				RK_data->RK2[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK2[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				RK_data->RK3[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK3[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				RK_data->RK4[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK4[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				#ifdef __RK5
				RK_data->RK5[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK5[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				RK_data->RK6[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK6[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				#endif
				#ifdef __DPRK5
				RK_data->RK7[SYS_DIM * indx_four + 0]    = 0.0 + 0.0 * I;
				RK_data->RK7[SYS_DIM * indx_four + 1]    = 0.0 + 0.0 * I;
				RK_data->w_hat_last[indx_four]			 = 0.0 + 0.0 * I;
				#endif
				if (i == 0) {
					if (j < Ny_Fourier) {
						run_data->k[1][j] = 0;
					}
					run_data->x[1][j] = 0.0;
				}
			}
		}
		run_data->k[0][i] = 0; 
		run_data->x[0][i] = 0.0;
	}
}
/**
 * Wrapper function that initializes the FFTW plans using MPI
 * @param N Array containing the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// -----------------------------------
	// Initialize Plans for Vorticity 
	// -----------------------------------
	// Set up FFTW plans for normal transform - vorticity field
	sys_vars->fftw_2d_dft_r2c = fftw_mpi_plan_dft_r2c_2d(Nx, Ny, run_data->w, run_data->w_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	sys_vars->fftw_2d_dft_c2r = fftw_mpi_plan_dft_c2r_2d(Nx, Ny, run_data->w_hat, run_data->w, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	if (sys_vars->fftw_2d_dft_r2c == NULL || sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}

	// -------------------------------------
	// Initialize batch Plans for Velocity 
	// -------------------------------------
	// Set up FFTW plans for batch transform - velocity fields
	sys_vars->fftw_2d_dft_batch_r2c = fftw_mpi_plan_many_dft_r2c((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u, run_data->u_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	sys_vars->fftw_2d_dft_batch_c2r = fftw_mpi_plan_many_dft_c2r((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u_hat, run_data->u, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	if (sys_vars->fftw_2d_dft_batch_r2c == NULL || sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
}
/**
 * Function to initialize and compute the system measurables and spectra of the initial conditions
 * @param RK_data The struct containing the Runge-Kutta arrays to compute the nonlinear term for the fluxes
 */
void InitializeSystemMeasurables(RK_data_struct* RK_data) {

	// Set the size of the arrays to twice the number of printing steps to account for extra steps due to adaptive stepping
	#ifdef __ADAPTIVE_STEP
	sys_vars->num_print_steps = 2 * sys_vars->num_print_steps;
	#else
	sys_vars->num_print_steps = sys_vars->num_print_steps;
	#endif
	int print_steps = sys_vars->num_print_steps;
		
	// ------------------------
	// Allocate Memory
	// ------------------------
	// Total Energy in the system
	run_data->tot_energy = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_energy == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Energy");
		exit(1);
	}	

	// Total Enstrophy
	run_data->tot_enstr = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_enstr == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Enstrophy");
		exit(1);
	}	

	// Total Palinstrophy
	run_data->tot_palin = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_palin == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Palinstrophy");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate");
		exit(1);
	}	

	// Enstrophy flux
	run_data->enst_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Subset");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate Subset");
		exit(1);
	}	

	// Energy Flux
	run_data->enrg_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Subset");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate Subset");
		exit(1);
	}
	
	// Time
	#ifdef __TIME
	if (!(sys_vars->rank)){
		run_data->time = (double* )fftw_malloc(sizeof(double) * print_steps);
		if (run_data->time == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time");
			exit(1);
		}	
	}
	#endif

	// ----------------------------
	// Get Measurables of the ICs
	// ----------------------------
	// Total Energy
	run_data->tot_energy[0] = TotalEnergy();

	// Total Enstrophy
	run_data->tot_enstr[0] = TotalEnstrophy();

	// Total Palinstrophy
	run_data->tot_palin[0] = TotalPalinstrophy();

	// Energy dissipation rate
	run_data->enrg_diss[0] = EnergyDissipationRate();

	// Enstrophy dissipation rate
	run_data->enst_diss[0] = EnstrophyDissipationRate();

	// Enstrophy Flux and dissipation from/to Subset of modes
	EnstrophyFlux(&(run_data->enst_flux_sbst[0]), &(run_data->enst_diss_sbst[0]), RK_data);

	// Energy Flux and dissipation from/to a subset of modes
	EnergyFlux(&(run_data->enrg_flux_sbst[0]), &(run_data->enrg_diss_sbst[0]), RK_data);

	// Time
	#ifdef __TIME
	if (!(sys_vars->rank)) {
		run_data->time[0] = sys_vars->t0;
	}
	#endif

	// ----------------------------
	// Get Spectra of the ICs
	// ----------------------------
	#ifdef __SPECT
	// Call spectra functions
	run_data->enst_spect 	  = (double* )EnstrophySpectrum(&(sys_vars->n_spect));
	run_data->enrg_spect 	  = (double* )EnergySpectrum(&(sys_vars->n_spect));
	run_data->enrg_flux_spect = (double* )EnergyFluxSpectrum(&(sys_vars->n_spect), RK_data);
	run_data->enst_flux_spect = (double* )EnstrophyFluxSpectrum(&(sys_vars->n_spect), RK_data);
	#endif
}
/**
 * Wrapper function that frees any memory dynamcially allocated in the programme
 * @param RK_data Pointer to a struct contaiing the integraiont arrays
 */
void FreeMemory(RK_data_struct* RK_data) {

	// ------------------------
	// Free memory 
	// ------------------------
	// Free space variables
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}

	// Free system variables
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	fftw_free(run_data->w);
	fftw_free(run_data->w_hat);
	fftw_free(run_data->tot_energy);
	fftw_free(run_data->tot_enstr);
	fftw_free(run_data->tot_palin);
	fftw_free(run_data->enrg_diss);
	fftw_free(run_data->enst_diss);
	fftw_free(run_data->enst_flux_sbst);
	fftw_free(run_data->enst_diss_sbst);
	fftw_free(run_data->enrg_flux_sbst);
	fftw_free(run_data->enrg_diss_sbst);
	#ifdef TESTING
	fftw_free(run_data->tg_soln);
	#endif
	#ifdef __TIME
	if (!(sys_vars->rank)){
		fftw_free(run_data->time);
	}
	#endif

	// Free integration variables
	fftw_free(RK_data->RK1);
	fftw_free(RK_data->RK2);
	fftw_free(RK_data->RK3);
	fftw_free(RK_data->RK4);
	#ifdef __RK5
	fftw_free(RK_data->RK5);
	fftw_free(RK_data->RK6);
	#endif 
	#ifdef __DPRK5
	fftw_free(RK_data->RK7);
	fftw_free(RK_data->w_hat_last);
	#endif
	fftw_free(RK_data->RK_tmp);
	fftw_free(RK_data->nabla_w);
	fftw_free(RK_data->nabla_psi);

	// ------------------------
	// Destroy FFTW plans 
	// ------------------------
	fftw_destroy_plan(sys_vars->fftw_2d_dft_r2c);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_r2c);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------