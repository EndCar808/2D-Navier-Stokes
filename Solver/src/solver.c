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
				  	RK4_C3 = 0.5,	           					RK4_A32 = 0.5, \
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
				              		  RK5_Bs1 = 5179.0/57600.0, 						   RK5_Bs3 = 7571.0/16695.0, RK5_Bs4 = 393.0/640.0,   RK5_Bs5 = -92097.0/339200.0, RK5_Bs6 = 187.0/2100.0, RK5_Bs7 = 1.0/40.0;
#endif
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Main function that performs the pseudospectral solver
 */
void SpectralSolve(void) {

	// Initialize variables
	const long int N[SYS_DIM]      = {sys_vars->N[0], sys_vars->N[1]};
	const long int NBatch[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1] / 2 + 1};

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
	// If in testing / debug mode - create/ open test file
	#if defined(DEBUG)
	OpenTestingFile();
	#endif

	// Initialize the collocation points and wavenumber space 
	InitializeSpaceVariables(run_data->x, run_data->k, N);

	// Get initial conditions
	InitialConditions(run_data->w_hat, run_data->u, run_data->u_hat, N);

	// Initialize the forcing 
	InitializeForcing();
		
	// -------------------------------
	// Integration Variables
	// -------------------------------
	// Initialize integration variables
	double t0;
	double t;
	double dt;
	double T;
	long int trans_steps;
	#if defined(__DPRK5)
	int try = 1;
	double dt_new;
	#endif

	// Get timestep and other integration variables
	InitializeIntegrationVariables(&t0, &t, &dt, &T, &trans_steps);
	
	// -------------------------------
	// Create & Open Output File
	// -------------------------------
	// Inialize system measurables
	InitializeSystemMeasurables(RK_data);
    
	// Create and open the output file - also write initial conditions to file
	CreateOutputFilesWriteICs(N, dt);

	// -------------------------------------------------
	// Print IC to Screen 
	// -------------------------------------------------
	#if defined(__PRINT_SCREEN)
	PrintUpdateToTerminal(0, t0, dt, T, 0);
	#endif	
	
	//////////////////////////////
	// Begin Integration
	//////////////////////////////
	t 				   += dt;
	int iters          = 1;
	#if defined(TRANSIENTS)
	int save_data_indx = 0;
	#else
	int save_data_indx = 1;
	#endif
	// while (t <= T) {

	// 	// -------------------------------	
	// 	// Integration Step
	// 	// -------------------------------
	// 	#if defined(__RK4)
	// 	RK4Step(dt, N, sys_vars->local_Nx, RK_data);
	// 	#elif defined(__RK5)
	// 	RK5DPStep(dt, N, iters, sys_vars->local_Nx, RK_data);
	// 	#elif defined(__DPRK5)
	// 	while (try) {
	// 		// Try a Dormand Prince step and compute the local error
	// 		RK5DPStep(dt, N, iters, sys_vars->local_Nx, RK_data);

	// 		// Compute the new timestep
	// 		dt_new = dt * DPMin(DP_DELTA_MAX, DPMax(DP_DELTA_MIN, DP_DELTA * pow(1.0 / RK_data->DP_err, 0.2)));
			
	// 		// If error is bad repeat else move on
	// 		if (RK_data->DP_err < 1.0) {
	// 			RK_data->DP_fails++;
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
	// 	if ((iters > trans_steps) && (iters % sys_vars->SAVE_EVERY == 0)) {
	// 		#if defined(TESTING)
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
	// 	// Print Update To Screen
	// 	// -------------------------------
	// 	#if defined(__PRINT_SCREEN)
	// 	#if defined(TRANSIENTS)
	// 	if (iters == trans_steps && !(sys_vars->rank)) {
	// 		printf("\n\n...Transient Iterations Complete!\n\n");
	// 	}
	// 	#endif
	// 	if (iters % sys_vars->SAVE_EVERY == 0) {
	// 		#if defined(TRANSIENTS)
	// 		if (iters <= sys_vars->trans_iters) {
	// 			// If currently performing transient iters, call system measure for printing to screen
	// 			RecordSystemMeasures(t, save_data_indx, RK_data);
	// 		}
	// 		#endif
	// 		PrintUpdateToTerminal(iters, t, dt, T, save_data_indx - 1);
	// 	}
	// 	#endif

	// 	// -------------------------------
	// 	// Update & System Check
	// 	// -------------------------------
	// 	// Update timestep & iteration counter
	// 	iters++;
	// 	#if defined(__ADAPTIVE_STEP) 
	// 	GetTimestep(&dt);
	// 	t += dt; 
	// 	#elif !defined(__DPRK5) && !defined(__ADAPTIVE_STEP)
	// 	t = iters * dt;
	// 	#endif

	// 	// Check System: Determine if system has blown up or integration limits reached
	// 	SystemCheck(dt, iters);
	// }
	//////////////////////////////
	// End Integration
	//////////////////////////////
	

	// ------------------------------- 
	// Final Writes to Output File
	// -------------------------------
	FinalWriteAndCloseOutputFile(N, iters, save_data_indx);
	

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
	#if defined(__NAVIER)
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny_Fourier = N[1] / 2 + 1;
	#if defined(__DPRK5)
	const long int Nx = N[0];
	double dp_ho_step;
	double err_sum;
	double err_denom;
	#endif
	
	//------------------- Pre-record the amplitudes so they can be reset after update step
	#if defined(PHASE_ONLY)
	double tmp_a_k_norm;

	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// record amplitudes
			run_data->tmp_a_k[indx] = cabs(run_data->w_hat[indx]);
		}
	}
	#endif

	//------------------- Get the forcing
	ComputeForcing();
	
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
	#if defined(__DPRK5)
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

			#if defined(PHASE_ONLY)
			tmp_a_k_norm = cabs(run_data->w_hat[indx]);
			#endif


			#if defined(__EULER)
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
			#if defined(PHASE_ONLY)
			// Reset the amplitudes
			run_data->w_hat[indx] *= (tmp_a_k_norm / cabs(run_data->w_hat[indx]));
			#endif
			#if defined(__DPRK5)
			if (iters > 1) {
				// Get the higher order update step
				dp_ho_step = run_data->w_hat[indx] + (dt * (RK5_Bs1 * RK_data->RK1[indx]) + dt * (RK5_Bs3 * RK_data->RK3[indx]) + dt * (RK5_Bs4 * RK_data->RK4[indx]) + dt * (RK5_Bs5 * RK_data->RK5[indx]) + dt * (RK5_Bs6 * RK_data->RK6[indx])) + dt * (RK5_Bs7 * RK_data->RK7[indx]));
				#if defined(PHASE_ONLY)
				// Reset the amplitudes
				dp_ho_step *= (tmp_a_k_norm / cabs(run_data->w_hat[indx]))
				#endif

				// Denominator in the error
				err_denom = DP_ABS_TOL + DPMax(cabs(RK_data->w_hat_last[indx]), cabs(run_data->w_hat[indx])) * DP_REL_TOL;

				// Compute the sum for the error
				err_sum += pow((run_data->w_hat[indx] - dp_ho_step) /  err_denom, 2.0);
			}
			#endif
		}
	}
	#if defined(__NONLIN)
	// Record the nonlinear for the updated Fourier vorticity
	NonlinearRHSBatch(run_data->w_hat, run_data->nonlinterm, RK_data->nabla_psi, RK_data->nabla_w);
	#endif
	#if defined(__DPRK5)
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
#if defined(__DPRK5)
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
#if defined(__RK4)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {

	// Initialize vairables
	int tmp;
	int indx;
	#if defined(__NAVIER)
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny_Fourier = N[1] / 2 + 1;

	//------------------- Pre-record the amplitudes so they can be reset after update step
	#if defined(PHASE_ONLY)
	double tmp_a_k_norm;

	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// record amplitudes
			run_data->tmp_a_k[indx] = cabs(run_data->w_hat[indx]);
		}
	}
	#endif

	//------------------- Get the forcing
	ComputeForcing();


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

			#if defined(PHASE_ONLY)
			// Pre-record the amplitudes
			tmp_a_k_norm = cabs(run_data->w_hat[indx]);
			#endif

			#if defined(__EULER)
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

			// Update Fourier vorticity
			run_data->w_hat[indx] = run_data->w_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * RK_data->RK1[indx] + RK4_B2 * RK_data->RK2[indx] + RK4_B3 * RK_data->RK3[indx] + RK4_B4 * RK_data->RK4[indx]);
			#endif
			#if defined(PHASE_ONLY)
			run_data->w_hat[indx] *= (tmp_a_k_norm / cabs(run_data->w_hat[indx]));
			#endif
		}
	}
	#if defined(__NONLIN)
	// Record the nonlinear term with the updated Fourier vorticity
	NonlinearRHSBatch(run_data->w_hat, run_data->nonlinterm, RK_data->nabla_psi, RK_data->nabla_w);
	#endif
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
 	// Apply Dealiasing & Forcing
 	// -------------------------------------
 	// Apply dealiasing 
 	ApplyDealiasing(dw_hat_dt, 1, sys_vars->N);

 	// Add the forcing
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dw_hat_dt[run_data->forcing_indx[i]] += run_data->forcing[i]; 
		}
	}

 	// Free memory
 	free(nonlinterm);
}
/**
 * Function to apply the selected dealiasing filter to the input array. Can be Fourier vorticity or velocity
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
	const long int Ny_Fourier = Ny / 2 + 1;
	#if defined(__DEALIAS_HOU_LI)
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter 
	// --------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = array_dim * (tmp + j);

			#if defined(__DEALIAS_23)
			if (sqrt((double) run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) > Nx / 3) {
				for (int l = 0; l < array_dim; ++l) {
					// Set dealised modes to 0
					array[indx + l] = 0.0 + 0.0 * I;	
				}
			}
			else {
				for (int l = 0; l < array_dim; ++l) {
					// Apply DFT normaliztin to undealiased modes
					array[indx + l] = array[indx + l];	
				}				
			}
			#elif defined(__DEALIAS_HOU_LI)
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
 * Function that comutes the forcing for the current timestep
 */
void ComputeForcing(void) {

	// Initialize variables
	int tmp, indx;
	double r1, r2;
	double re_f, im_f;
	ptrdiff_t local_Nx        = sys_vars->local_Nx;
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = Ny / 2 + 1;

	// --------------------------------------------
	// Compute Forcing
	// --------------------------------------------
	// Compute the forcing on the local process containing the forced modes
	if (sys_vars->local_forcing_proc) {
		//---------------------------- Compute Zero forcing -> specified modes are killed/set to 0
		if(!(strcmp(sys_vars->forcing, "ZERO"))) {
			// Loop over the forced modes
			for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
				run_data->w_hat[run_data->forcing_indx[i]] = 0.0 + 0.0 * I;
			}
		}
		//---------------------------- Compute Kolmogorov forcing -> f(u) = (sin(n y), 0); f(w) = -n cos(n y) -> f_k = -1/2 * n \delta(n)
		else if(!(strcmp(sys_vars->forcing, "KOLM"))) {
			// Compute the Kolmogorov forcing
			run_data->forcing[0] = -0.5 * sys_vars->force_k * (sys_vars->force_k + 0.0 * I);
		}
		//---------------------------- Compute Stochastic forcing
		else if(!(strcmp(sys_vars->forcing, "STOC"))) {
			// Loop over the forced modes
			for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
				// Generate two uniform random numbers
				r1 = (double) rand() / (double) RAND_MAX;
				r2 = (double) rand() / (double) RAND_MAX;

				// Convert to Gaussian using Box-Muller transform
				re_f = sqrt(-2.0 * log(r1)) * cos(r2 * 2.0 * M_PI);
				im_f = sqrt(-2.0 * log(r1)) * sin(r2 * 2.0 * M_PI);

				// Now compute the forcing 
				run_data->forcing[i] = run_data->forcing_scaling[i] * (re_f + im_f * I);
			}		
		}
		//---------------------------- No forcing
		else {

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

    // ------------------------------------------------
    // Set Seed for RNG
    // ------------------------------------------------
    srand(123456789);

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

				// Top Layer
				run_data->w[indx] = DELTA * cos(run_data->x[1][j]) - SIGMA / pow(cosh(SIGMA * (run_data->x[0][i] - 0.5 * M_PI)), 2.0); 

				// Bottom Layer
				run_data->w[indx] += DELTA * cos(run_data->x[1][j]) + SIGMA / pow(cosh(SIGMA * (1.5 * M_PI - run_data->x[0][i])), 2.0); 
			}
		}

		// ---------------------------------------
		// Transform to Fourier Space
		// ---------------------------------------
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), run_data->w, w_hat);
	}
	else if (!(strcmp(sys_vars->u0, "DECAY_TURB")) || !(strcmp(sys_vars->u0, "DECAY_TURB_II")) || !(strcmp(sys_vars->u0, "DECAY_TURB_EXP")) || !(strcmp(sys_vars->u0, "DECAY_TURB_EXP_II"))) {
		// --------------------------------------------------------
		// Decaying Turbulence ICs
		// --------------------------------------------------------
		// Initialize variables
		double sqrt_k;
		double inv_k_sqr;
		double u1;
		double spec_1d;

		#if defined(DEBUG)
		double* rand_u = (double*)fftw_malloc(sizeof(double) * Nx * Ny_Fourier);
		#endif
		// ---------------------------------------------------------------
		// Initialize Vorticity with Specific Spectrum and Random Phases
		// ---------------------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;	

				if (run_data->k[0][i] == 0.0 && run_data->k[1][j] == 0.0) {
					// Compute the energy
					sqrt_k  = 0.0;
					spec_1d = 0.0;

					// Fill the vorticity	
					w_hat[indx] = 0.0 + 0.0 * I;
				}
				else {
					// Compute the form of the initial energy
					sqrt_k = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

					if (!(strcmp(sys_vars->u0, "DECAY_TURB"))) {
						// Computet the Broad band initial spectrum
						spec_1d = sqrt_k / (1.0 + pow(sqrt_k, 4.0) / DT_K0);
					} 
					else if (!(strcmp(sys_vars->u0, "DECAY_TURB_II"))) {
						// Compute the Narrow Band initial spectrum
						spec_1d = pow(sqrt_k, 6.0) / pow((1.0 + sqrt_k / (2.0 * DT2_K0)), 18.0);
					}
					else if (!(strcmp(sys_vars->u0, "DECAY_TURB_EXP"))) {
						// Computet the Broad band initial spectrum
						spec_1d = pow(sqrt_k, 7.0) / pow(DTEXP_K0, 8.0) * cexp( - 3.5 * pow(sqrt_k / DTEXP_K0, 2.0));
					} 
					else if (!(strcmp(sys_vars->u0, "DECAY_TURB_EXP_II"))) {
						// Computet the Broad band initial spectrum
						// spec_1d = pow(sqrt_k, 7.0) / pow(DTEXP_K0, 8.0) * cexp( - 3.5 * pow(sqrt_k / DTEXP_K0, 2.0));
					} 
					
					// Generate uniform random number between 0, 1
					u1 = (double)rand() / (double) RAND_MAX;
					#if defined(DEBUG)
					rand_u[indx] = u1;
					#endif

					if (!(strcmp(sys_vars->u0, "DECAY_TURB_EXP"))) {
						// Fill the vorticity	
						w_hat[indx] = sqrt(sqrt_k * spec_1d / (M_PI)) * cexp(2.0 * M_PI * u1 * I); 
					}
					else {
						// Fill the vorticity -. the factor of k/2pi is to account for the area of the annulus
						w_hat[indx] = sqrt(spec_1d * (sqrt_k / (2.0 * M_PI))) * cexp(2.0 * M_PI * u1 * I);
					}
				}
			}
		}
		#if defined(DEBUG)
		int dims[2] = {Nx, Ny_Fourier};
		WriteTestDataReal(rand_u, "rand_u", 2, dims, local_Nx);
		WriteTestDataFourier(w_hat, "w_hat", Nx, Ny_Fourier, local_Nx);
		fftw_free(rand_u);
		#endif

		// ---------------------------------------
		// Compute the Initial Energy
		// ---------------------------------------
		double enrg = 0.0;
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;	

				if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
					// Wavenumber prefactor -> 1 / |k|^2
					inv_k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					if ((j == 0) || (j == Ny_Fourier - 1)) {
						enrg += inv_k_sqr * cabs(w_hat[indx] * conj(w_hat[indx]));
					}
					else {
						enrg += 2.0 * inv_k_sqr * cabs(w_hat[indx] * conj(w_hat[indx]));
					}
				}
			}
		}
		// Normalize 
		enrg *= (0.5 / pow(Nx * Ny, 2.0)) * 4.0 * pow(M_PI, 2.0);

		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &enrg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// -------------------------------------------
		// Normalize & Compute the Fourier Vorticity
		// -------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if (!(strcmp(sys_vars->u0, "DECAY_TURB"))) {
					// Compute the Fouorier vorticity
					w_hat[indx] *= sqrt(DT_E0 / enrg);
				}
				else if (!(strcmp(sys_vars->u0, "DECAY_TURB_II"))) {
					// Compute the Fouorier vorticity
					w_hat[indx] *= sqrt(DT2_E0 / enrg);
				}
				else if (!(strcmp(sys_vars->u0, "DECAY_TURB_EXP"))) {
					// Compute the Fouorier vorticity
					w_hat[indx] *= sqrt(DTEXP_E0 / enrg);
				}
			}
		}
	}
	else if (!(strcmp(sys_vars->u0, "DECAY_TURB_ALT"))) {
		// ---------------------------------------------
		// Gaussian IC with Prescribed initial Spectrum
		// ---------------------------------------------
		double k_sqrt;
		double k_sqrd;
		double u1, u2, z1, z2;
		double spec_1d;

		#if defined(DEBUG)
		fftw_complex* rand_u = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
		#endif

		// Allocate memory for the stream function
		double* psi = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
		if (psi == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Stream Function");
			exit(1);
		}	
		fftw_complex* psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
		if (psi_hat == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Stream Function");
			exit(1);
		}	

		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;	

				// Generate Standard normal random variable
				u1 = (double) rand() / (double) RAND_MAX;
				u2 = (double) rand() / (double) RAND_MAX;
				z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
				z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

				#if defined(DEBUG)
				rand_u[indx] = (z1  + z2 * I);
				#endif

				if (run_data->k[0][i] == 0 && run_data->k[1][j] == 0) {
					psi_hat[indx] = 0.0 + 0.0 * I;
				}
				else {
					// Compute the mod of k -> |k|
					k_sqrt = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

					// Get the initial form of the fourier stream function sqrd -> definition in the paper
					spec_1d = 1.0 / (k_sqrt * (1.0 + pow(k_sqrt / 6.0, 4.0)));
					
					// Fill the stream function -> spec_1d * k_sqrt^2 is the form of the initial energy spectrum
					psi_hat[indx] = (z1  + z2 * I) * sqrt(spec_1d * pow(k_sqrt, 2.0) / (pow(k_sqrt, 3.0) * 2.0 * M_PI));
				}
			}
		}
		#if defined(DEBUG)
		WriteTestDataFourier(rand_u, "rand_u", Nx, Ny_Fourier, local_Nx);
		WriteTestDataFourier(psi_hat, "psi_hat", Nx, Ny_Fourier, local_Nx);
		fftw_free(rand_u);
		#endif
		
		// ---------------------------------------------
		// Ensure Zero Mean Field
		// ---------------------------------------------
		// Variable to compute the mean
		double mean = 0.0;

		// Transform and Normalize while also gathering the mean
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), psi_hat, psi);
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;	

				// Normalize
				psi[indx] /= (Nx * Ny);

				// Update the sum for the mean
				mean += psi[indx];
			}
		}
		

		// Reduce all local mean sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Enforce the zero mean of the stream function field
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;	

				// Normalize
				psi[indx] -=  (mean / (Nx * Ny));
			}
		}

		// Transform zero mean field back to Fourier space
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), psi, psi_hat);

		
		// ---------------------------------------------
		// Compute the Energy
		// ---------------------------------------------
		double enrg = 0.0;
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Wavenumber prefactor -> |k|^2
				k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					enrg += k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
				}
				else {
					enrg += 2.0 * k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
				}
			}
		}
		// Normalize the energy
		enrg *= 4.0 * pow(M_PI, 2.0) * (0.5 / pow(Nx * Ny, 2.0));
		
		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &enrg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// ---------------------------------------------
		// Normalize Initial Condition - Compute the Vorticity
		// ---------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Normalize the initial condition
				psi_hat[indx] *= sqrt(0.5 / enrg);

				// Get |k|^2
				k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Compute the vorticity 
				w_hat[indx] = -k_sqrd * psi_hat[indx];
			}
		}
        

		// Free memory
		fftw_free(psi_hat);
		fftw_free(psi);
	}
	else if (!(strcmp(sys_vars->u0, "GAUSS_DECAY_TURB"))) {
		// ---------------------------------------------
		// Gaussian IC with Prescribed initial Spectrum
		// ---------------------------------------------
		double k_sqrt;
		double k_sqrd;
		double u1;
		double spec_1d;

		// Allocate memory for the stream function
		fftw_complex* psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);

		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;	

				if (run_data->k[0][i] == 0 && run_data->k[1][j] == 0) {
					psi_hat[indx] = 0.0 + 0.0 * I;
				}
				else {
					k_sqrt = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

					// spec_1d = k_sqrt / (1.0 + pow(k_sqrt, 4.0) / 6.0);
					spec_1d = pow(k_sqrt, 6.0) / pow((1.0 + k_sqrt / 60.0), 18.0);

					// Fill the Fourier vorticity with Gaussian data
					u1 = (double) rand() / (double) RAND_MAX;
					psi_hat[indx] = sqrt(spec_1d * (1.0 / k_sqrt)) * cexp(I * 2.0 * M_PI * u1); //  * 
				}
			}
		}

		// ---------------------------------------------
		// Compute the Energy
		// ---------------------------------------------
		double enrg = 0.0;
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
					// Wavenumber prefactor -> |k|^2
					k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					if ((j == 0) || (j == Ny_Fourier - 1)) {
						enrg += k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
					}
					else {
						enrg += 2.0 * k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
					}
				}
			}
		}
		// Normalize the energy
		enrg *= 4.0 * pow(M_PI, 2.0) * (0.5 / pow(Nx * Ny, 2.0));
		
		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &enrg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// ---------------------------------------------
		// Normalize Initial Condition - Compute the Vorticity
		// ---------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Normalize the initial condition
				psi_hat[indx] /=  sqrt(enrg);
				psi_hat[indx] *=  sqrt(0.5);

				// Get |k|^2
				k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Compute the vorticity 
				w_hat[indx] = k_sqrd * psi_hat[indx];
			}
		}

		// Free memory
		fftw_free(psi_hat);
	}
	else if (!(strcmp(sys_vars->u0, "GAUSS_BLOB"))) {
	
		// ---------------------------------------
		// Define the Gaussian Blob
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Fill vorticity
				run_data->w[indx] = -exp(-(pow(run_data->k[0][i] - M_PI, 2.0) + BETA * pow(run_data->k[1][j] - M_PI, 2.0)) / pow(2.0 * M_PI / S, 2.0));
			}
		}		

		// ---------------------------------------------
		// Transform to Fourier Space
		// ---------------------------------------------
		// Transform
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), run_data->w, w_hat);
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
				w_hat[indx] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX)* 2.0 * M_PI * I);
			}
		}		
	}
	else if (!(strcmp(sys_vars->u0, "TESTING"))) {
		// Initialize temp variables
		double inv_k_sqr;

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
					inv_k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill vorticity - this is for the kx axis - to enfore conjugate symmetry
					w_hat[indx] = inv_k_sqr * cexp(-I * M_PI / 4.0);
				}
				else {
					// Amplitudes
					inv_k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill vorticity - fill the rest of the modes
					w_hat[indx] = inv_k_sqr * cexp(I * M_PI / 4.0);
				}
			}
		}
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
				w_hat[indx] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX) * 2.0 * M_PI * I);
			}
		}		
	}
	// -------------------------------------------------
	// Initialize the Dealiasing & Force Conjugacy
	// -------------------------------------------------
	// Apply dealiasing to initial condition
	ApplyDealiasing(w_hat, 1, N);
    
    // Ensure conjugacy in the ky = 0 modes of the intial condition
    ForceConjugacy(w_hat, N);

 
   	// -------------------------------------------------
   	// Get Max of Initial Condition
   	// -------------------------------------------------
   	sys_vars->w_max_init = GetMaxData("VORT");

	// -------------------------------------------------
	// Initialize Taylor Green Vortex Soln 
	// -------------------------------------------------
	// If testing is enabled and TG initial condition selected -> compute TG solution for writing to file @ t = t0
	#if defined(TESTING)
	if(!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
		TaylorGreenSoln(0.0, N);
	}
	#endif
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
	
	// Set the spatial increments
	sys_vars->dx = 2.0 * M_PI / (double )Nx;
	sys_vars->dy = 2.0 * M_PI / (double )Ny;

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
	// Fill the second direction 
	// -------------------------------
	for (int i = 0; i < Ny; ++i) {
		if (i < Ny_Fourier) {
			k[1][i] = i;
		}
		x[1][i] = (double) i * 2.0 * M_PI / (double) Ny;
	}
}
/**
 * Function to force conjugacy of the initial condition
 * @param w_hat The Fourier space worticity field
 * @param N     The array containing the size of the system in each dimension
 */
void ForceConjugacy(fftw_complex* w_hat, const long int* N) {

	// Initialize variables
	int tmp;
	int local_Nx       = (int) sys_vars->local_Nx;
	int local_Nx_start = (int) sys_vars->local_Nx_start;
	const long int Nx         = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate tmp memory to hold the data to be conjugated
	fftw_complex* conj_data = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx);

	// Loop through local process and store data in appropriate location in conj_data
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		conj_data[local_Nx_start + i] = run_data->w_hat[tmp];
	}

	// Gather the data on all process
	MPI_Allgather(MPI_IN_PLACE, (int)local_Nx, MPI_C_DOUBLE_COMPLEX, conj_data, (int)local_Nx, MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);

	// Now ensure the 
	for (int i = 0; i < local_Nx; ++i) {
		if (run_data->k[0][i] < 0) {
			tmp = i * Ny_Fourier;

			// Fill the conjugate modes with the conjugate of the postive k modes
			run_data->w_hat[tmp] = conj(conj_data[abs(run_data->k[0][i])]);
		}
	}	
}
/**
 * Function to initialize all the integration time variables
 * @param t0           The initial time of the simulation
 * @param t            The current time of the simulaiton
 * @param dt           The timestep
 * @param T            The final time of the simulation
 * @param trans_steps  The number of iterations to perform before saving to file begins
 */
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps) {
	
	// -------------------------------
	// Get the Timestep
	// -------------------------------
	#if defined(__ADAPTIVE_STEP)
	GetTimestep(&(sys_vars->dt));
	#endif

	// -------------------------------
	// Get Time variables
	// -------------------------------
	// Compute integration time variables
	(*t0) = sys_vars->t0;
	(*t ) = sys_vars->t0;
	(*dt) = sys_vars->dt;
	(*T ) = sys_vars->T;
	sys_vars->min_dt = 10;
	sys_vars->max_dt = MIN_STEP_SIZE;

	// -------------------------------
	// Integration Counters
	// -------------------------------
	// Number of time steps and saving steps
	sys_vars->num_t_steps = ((*T) - (*t0)) / (*dt);
	#if defined(TRANSIENTS)
	// Get the transient iterations
	(* trans_steps)       = (long int)(TRANS_FRAC * sys_vars->num_t_steps);
	sys_vars->trans_iters = (* trans_steps);

	// Get the number of steps to perform before printing to file -> allowing for a transient fraction of these to be ignored
	sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? (sys_vars->num_t_steps - sys_vars->trans_iters) / sys_vars->SAVE_EVERY : sys_vars->num_t_steps - sys_vars->trans_iters;	 
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\t Transient Steps: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps, sys_vars->trans_iters);
	}
	#else
	// Get the transient iterations
	(* trans_steps)       = 0;
	sys_vars->trans_iters = (* trans_steps);

	// Get the number of steps to perform before printing to file
	sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? sys_vars->num_t_steps / sys_vars->SAVE_EVERY + 1 : sys_vars->num_t_steps + 1; // plus one to include initial condition
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps);
	}
	#endif

	// Variable to control how ofter to print to screen -> set it to half the saving to file steps
	sys_vars->print_every = (sys_vars->num_t_steps >= 10 ) ? (int)sys_vars->SAVE_EVERY : 1;
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
	fftw_complex I_over_k_sqr;

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
					I_over_k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// compute the velocity in Fourier space
					run_data->u_hat[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * I_over_k_sqr * run_data->w_hat[indx];
					run_data->u_hat[SYS_DIM * indx + 1] = -1.0 * ((double) run_data->k[0][i]) * I_over_k_sqr * run_data->w_hat[indx];
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
			tmp = i * (Ny + 2);
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
 * Used to print update to screen
 * @param iters          The current iteration of the integration
 * @param t              The current time in the simulation
 * @param dt             The current timestep in the simulation
 * @param T              The final time of the simulation
 * @param save_data_indx The saving index for output data
 * @param RK_data        Struct containing arrays for the Runge-Kutta integration
 */
void PrintUpdateToTerminal(int iters, double t, double dt, double T, int save_data_indx) {

	// Initialize variables
	double max_vort;

	#if defined(TESTING)
	// Initialize norms array
	double norms[2];
	
	// Get max vorticity
	max_vort = GetMaxData("VORT");

	if(!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
		// Get Taylor Green Solution
		TestTaylorGreenVortex(t, sys_vars->N, norms);

		// Print Update to screen
		if( !(sys_vars->rank) ) {	
			printf("Iter: %d\tt: %1.6lf/%1.3lf\tdt: %g\t Max Vort: %1.4lf\tKE: %1.5lf\tENS: %1.5lf\tPAL: %1.5lf\tL2: %g\tLinf: %g\n", iters, t, dt, T, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx], norms[0], norms[1]);
		}
	}
	else {
		// Print Update to screen
		if( !(sys_vars->rank) ) {	
			printf("Iter: %d\tt: %1.6lf/%1.3lf\tdt: %g\t Max Vort: %1.4lf\tTKE: %1.8lf\tENS: %1.8lf\tPAL: %g\tE_Diss: %g\tEns_Diss: %g\n", iters, t, T, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx], run_data->enrg_diss[save_data_indx], run_data->enst_diss[save_data_indx]);
		}
	}
	#else
	// Get max vorticity
	max_vort = GetMaxData("VORT");

	// Print to screen
	if( !(sys_vars->rank) ) {	
		printf("Iter: %d/%ld\tt: %1.6lf/%1.3lf\tdt: %g\tMax Vort: %1.4lf\tKE: %1.5lf\tENS: %1.5lf\tPAL: %1.5lf\n", iters, sys_vars->num_t_steps, t, T, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx]);
	}
	#endif	
}
/**
 * Function to update the timestep if adaptive timestepping is enabled
 * @param dt The current timestep
 */
void GetTimestep(double* dt) {

	// -------------------------------
	// Compute New Timestep
	// -------------------------------
	#if defined(__CFL_STEP)
	// Initialize variables
	int tmp;
	int indx;
	fftw_complex I_over_k_sqr;
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
				I_over_k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// compute the velocity in Fourier space
				run_data->u_hat[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * I_over_k_sqr * run_data->w_hat[indx];
				run_data->u_hat[SYS_DIM * indx + 1] = -1.0 * ((double) run_data->k[0][i]) * I_over_k_sqr * run_data->w_hat[indx];
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

	// Find proposed timestep = h_0 * (max{w(0)} / max{w(t)}) -> this ensures that the maximum vorticity by the timestep is constant
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

	// -------------------------------
	// Record Min/Max Timestep
	// -------------------------------
	#if !defined(__DPRK5)
	// Get min timestep
	sys_vars->min_dt = fmin((*dt), sys_vars->min_dt);
	// Get max timestep
	sys_vars->max_dt = fmax((*dt), sys_vars->max_dt);
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
 */
void EnergySpectrum(void) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	double k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enrg_spect[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				run_data->enrg_spect[spec_indx] += 0.0;
			}
			else {
				// Compute |k|^2
				k_sqr = 1.0 / ((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the current bin for mode
					run_data->enrg_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
				else {
					// Update the energy sum for the current mode
					run_data->enrg_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
				}
			}
		}
	}
}
/**
 * Function used to compute the enstrophy spectrum for the current iteration. The enstrophy spectrum is defined as the total enstrophy contained in concentric annuli 
 * in wavenumber space
 */
void EnstrophySpectrum(void) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enst_spect[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				run_data->enst_spect[spec_indx] += 0.0;
			}
			else {
				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the current bin for mode
					run_data->enst_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					// Update the enstrophy sum for the current mode
					run_data->enst_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
		}
	}
}
/**
 * Function to compute the total divergence of the velocity field
 * @return  Total divergence
 */
double TotalDivergence(void) {

	// Initialize variables
	int tmp;
	int indx;
	fftw_complex k_sqr_inv;
	fftw_complex u_z, v_z, div_u_z;
	double tot_div = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);


	// ------------------------------------------
	// Compute Fourier Space Velocity & Divergence
	// ------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// The I/k^2 prefactor
				k_sqr_inv = 1.0 / (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the fourier velocities
				u_z = k_sqr_inv * ((double) run_data->k[1][j]) * run_data->w_hat[indx];
				v_z = -k_sqr_inv * ((double) run_data->k[0][i]) * run_data->w_hat[indx];

				// Get the divergence of u_z
				div_u_z = I * ((double )run_data->k[0][i] * u_z + (double )run_data->k[1][j] * v_z);

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the sum for the total energy
					tot_div += cabs(div_u_z * conj(div_u_z));
				}
				else {
					// Update the sum for the total energy
					tot_div += 2.0 * cabs(div_u_z * conj(div_u_z));
				}
			}
			else {
				tot_div += 0.0;
			}
		}
	}
	
	// Return result
	return 4.0 * M_PI * M_PI * tot_div * norm_fac;
}
/**
 * Function to compute the total forcing input into the system
 * @return  The total forcing input
 */
double TotalForcing(void) {

	// Initialize variables
	double tot_forcing = 0.0;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------------
	// Compute The Total Forcing
	// ------------------------------------------
	for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
		tot_forcing += 4.0 * cabs(sys_vars->forcing[i] * conj(run_data->w_hat[run_data->forcing_indx[i]]));
	}

	return norm_fac * tot_forcing;
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
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

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

				if ((j == 0) || (j == Ny_Fourier - 1)) {
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
	return 4.0 * M_PI * M_PI * tot_energy * norm_fac;
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
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

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

			if ((j == 0) || (j == Ny_Fourier - 1)) {
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
	return 4.0 * M_PI * M_PI * tot_enstr * norm_fac;
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
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

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
				if((j == 0) || (j == Ny_Fourier - 1)) {
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
	return 4.0 * M_PI * M_PI * tot_palin * norm_fac;
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
	#if defined(HYPER_VISC) || defined(EKMN_DRAG)
	double k_sqr;
	#endif
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -------------------------------
	// Compute The Energy Diss Rate
	// -------------------------------
	// Initialize total enstrophy
	double enrgy_diss = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			#if defined(HYPER_VISC) || defined(EKMN_DRAG)
			// Compute |k|^2
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			#endif

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
			if((j == 0) || (j == Ny_Fourier - 1)) {
				enrgy_diss += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}
			else {
				enrgy_diss += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
			}
		}
	}

	// Return result -> 2 (nu * 0.5 *<|w|^2>)
	return 2.0 * (4.0 * M_PI * M_PI * enrgy_diss * norm_fac);
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
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -------------------------------
	// Compute The Enstrophy Diss Rate 
	// -------------------------------
	// Initialize total enstrophy
	double tot_enst_diss = 0.0;

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

				// Update the running sum for the enst_dissstrophy -> first and last modes have no conjugate so only count once
				if((j == 0) || (j == Ny_Fourier - 1)) {
					tot_enst_diss += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					tot_enst_diss += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
			else {
				tot_enst_diss += 0.0;
			}
		}
	}


	// Return result -> 2(enst_diss) = 2(0.5 * <|grad \omega|^2>)
	return 2.0 * (4.0 * M_PI * M_PI * tot_enst_diss * norm_fac);
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
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Enstrophy Flux");
		exit(1);
	}

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

	// -------------------------------------
	// Compute the Enstrophy Flux & Diss
	// -------------------------------------
	// Initialize sums
	tmp_enst_flux = 0.0;
	tmp_enst_diss = 0.0;

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
				if ((j == 0) || (Ny_Fourier - 1)) {
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
	(*enst_diss) = 4.0 * M_PI * M_PI * tmp_enst_diss * norm_fac;

	// Compute the enstrophy flux
	(*enst_flux) = 8.0 * M_PI * M_PI * tmp_enst_flux * norm_fac;

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
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// -----------------------------------
	// Compute the Derivative
	// -----------------------------------
	// Allocate memory
	fftw_complex* dwhat_dt = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (dwhat_dt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the nonlinear array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Energy Flux");
		exit(1);
	}

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

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
					if ((j == 0) || (Ny_Fourier - 1)) {
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
	(*enrg_diss) = 4.0 * M_PI * M_PI * tmp_enrgy_diss * norm_fac;

	// Compute the energy flux
	(*enrg_flux) = 8.0 * M_PI * M_PI * tmp_enrgy_flux * norm_fac;


	// -----------------------------------
	// Free memory
	// -----------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the energy flux spectrum for the current iteration on the local processes
 * The results of the complete spectrum will be gathered on the master rank before writing to file
 * @param  RK_data The struct containing the Runge-Kutta arrays needed for computing the nonlinear term
 */
void EnergyFluxSpectrum(RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double pre_fac = 0.0;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enrg_flux_spect[i] = 0.0;
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

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

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
				spec_indx = (int) ceil(sqrt(k_sqr));

				// Update spectrum bin
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the running sum for the flux of energy
					run_data->enrg_flux_spect[spec_indx] += 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;
				}
				else {
					// Update the running sum for the flux of energy
					run_data->enrg_flux_spect[spec_indx] += 2.0 * 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]) / k_sqr;
				}
			}
		}
	}

	// -------------------------------------
	// Free Temp Memory
	// -------------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the enstrophy flux spectrum for the current iteration on the local processes
 * The results are gathered on the master rank before being written to file
 * @param  RK_data The struct containing the Runge-Kutta arrays for computing the nonlinear term
 */
void EnstrophyFluxSpectrum(RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	double k_sqr;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double pre_fac = 0.0;
	double norm_fac = 0.5 / pow(Nx * Ny, 2.0);

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enst_flux_spect[i] = 0.0;

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

	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, dwhat_dt, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			dwhat_dt[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}

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
				spec_indx = (int) ceil(sqrt(k_sqr));

				// Update spectrum bin
				if ((j == 0) || (Ny_Fourier - 1)) {
					// Update the current bin sum 
					run_data->enst_flux_spect[spec_indx] += 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);
				}
				else {
					// Update the running sum for the flux of energy
					run_data->enst_flux_spect[spec_indx] += 2.0 * 4.0 * M_PI * M_PI * norm_fac * creal(run_data->w_hat[indx] * conj(dwhat_dt[indx]) + conj(run_data->w_hat[indx]) * dwhat_dt[indx]);
				}
			}
		}
	}

	// -------------------------------------
	// Free Temp Memory
	// -------------------------------------
	fftw_free(dwhat_dt);
}
/**
 * Function to compute the system measurables such as energy, enstrophy, palinstrophy, helicity, energy and enstrophy dissipation rates, and spectra at once on the local processes for the current timestep
 * @param t 		The current time in the simulation
 * @param iter 		The index in the system arrays for the current timestep
 * @param RK_data 	Struct containing the integration varaiables needed for the nonlinear term function
 */
void ComputeSystemMeasurables(double t, int iter, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double k_sqr, pre_fac;
	fftw_complex u_z, v_z, div_u_z;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
    double lwr_sbst_lim_sqr = pow(LWR_SBST_LIM, 2.0);
    double upr_sbst_lim_sqr = pow(UPR_SBST_LIM, 2.0);

    // Record the initial time
    #if defined(__TIME) && !defined(TRANSIENTS)
    if (!(sys_vars->rank)) {
    	run_data->time[iter] = t;
    }
    #endif

    // If adaptive stepping check if within memory limits
    if ((iter >= sys_vars->num_print_steps) && (iter % 100 == 0)) {
    	// Print warning to screen if we have exceeded the memory limits for the system measurables arrays
    	printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", iter, t);
    }

	// ------------------------------------
	// Initialize Measurables
	// ------------------------------------
	#if defined(__SYS_MEASURES)
	if (iter < sys_vars->num_print_steps) {
		// Initialize totals
		run_data->tot_enstr[iter]  = 0.0;
		run_data->tot_palin[iter]  = 0.0;
		run_data->tot_energy[iter] = 0.0;
		run_data->tot_forc[iter]   = 0.0;
		run_data->tot_div[iter]    = 0.0;
		run_data->enrg_diss[iter]  = 0.0;
		run_data->enst_diss[iter]  = 0.0;
		#if defined(__ENRG_FLUX)
		run_data->enrg_diss_sbst[iter] = 0.0;
		run_data->enrg_diss_sbst[iter] = 0.0;
		#endif
		#if defined(__ENST_FLUX)
		run_data->enst_flux_sbst[iter] = 0.0;
		run_data->enst_diss_sbst[iter] = 0.0;
		#endif
	}
	#endif 
	#if defined(__ENRG_SPECT) || defined(__ENST_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	// Initialize spectra
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		#if defined(__ENRG_SPECT)
		run_data->enrg_spect[i] = 0.0;
		#endif
		#if defined(__ENST_SPECT)
		run_data->enst_spect[i] = 0.0;
		#endif
		#if defined(__ENST_FLUX_SPECT)
		run_data->enst_flux_spect[i] = 0.0;
		#endif
		#if defined(__ENRG_FLUX_SPECT)
		run_data->enrg_flux_spect[i] = 0.0;
		#endif
	}
	#endif

	#if defined(__ENRG_FLUX) || defined(__ENST_FLUX) || defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT)
	// Compute the nonlinear term & subtract the forcing as the flux computation should ignore focring
	NonlinearRHSBatch(run_data->w_hat, RK_data->RK1, RK_data->nabla_psi, RK_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			RK_data->RK1[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}
	#endif

	// Compute the total forcing
	for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
		run_data->tot_forc[iter] += cabs(run_data->forcing[i] * run_data->w_hat[run_data->forcing_indx[i]]);
	}

	// -------------------------------------
	// Compute Measurables in Fourier Space
	// -------------------------------------
	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;


			///--------------------------------- System Measures
			#if defined(__SYS_MEASURES)
		    if (iter < sys_vars->num_print_steps) {
				if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
					// The |k|^2 prefactor
					k_sqr = (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Get the appropriate prefactor
					#if defined(HYPER_VISC) && defined(EKMN_DRAG)  // Both Hyperviscosity and Ekman drag
					pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
					#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) // No hyperviscosity but we have Ekman drag
					pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, EKMN_POW);
					#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) // Hyperviscosity only
					pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW);
					#else // No hyper viscosity or no ekman drag -> just normal viscosity
					pre_fac = sys_vars->NU * k_sqr; 
					#endif

					// Get the fourier velocities
					u_z = I * ((double )run_data->k[1][j]) * run_data->w_hat[indx] / k_sqr;
					v_z = -I * ((double )run_data->k[0][i]) * run_data->w_hat[indx] / k_sqr;

					// Get the diverence of the Fourier velocity
					div_u_z = I * ((double )run_data->k[0][i] * u_z + (double )run_data->k[1][j] * v_z);

					// Update the sums
					if ((j == 0) || (j == Ny_Fourier - 1)) { // only count the 0 and N/2 modes once as they have no conjugate
						run_data->tot_energy[iter] += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
						run_data->tot_enstr[iter]  += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->tot_div[iter]    += cabs(div_u_z * conj(div_u_z));
						run_data->tot_palin[iter]  += k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->enrg_diss[iter]  += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
						run_data->enst_diss[iter]  += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						if ((k_sqr >= lwr_sbst_lim_sqr) && (k_sqr < upr_sbst_lim_sqr)) { // define the subset to consider for the flux and dissipation
							#if defined(__ENRG_FLUX)
							run_data->enrg_flux_sbst[iter] += creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]) * (1.0 / k_sqr);
							run_data->enrg_diss_sbst[iter] += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
							#endif
							#if defined(__ENST_FLUX)
							run_data->enst_flux_sbst[iter] += creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]); 
							run_data->enst_diss_sbst[iter] += pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
							#endif
						}
					}
					else {
						run_data->tot_energy[iter] += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
						run_data->tot_enstr[iter]  += 2.0 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->tot_div[iter]    += 2.0 * cabs(div_u_z * conj(div_u_z));
						run_data->tot_palin[iter]  += 2.0 * k_sqr * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						run_data->enrg_diss[iter]  += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
						run_data->enst_diss[iter]  += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
						if ((k_sqr >= lwr_sbst_lim_sqr) && (k_sqr < upr_sbst_lim_sqr)) { // define the subset to consider for the flux and dissipation
							#if defined(__ENRG_FLUX)
							run_data->enrg_flux_sbst[iter] += 2.0 * creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]) * (1.0 / k_sqr);
							run_data->enrg_diss_sbst[iter] += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
							#endif
							#if defined(__ENST_FLUX)
							run_data->enst_flux_sbst[iter] += 2.0 * creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]); 
							run_data->enst_diss_sbst[iter] += 2.0 * pre_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
							#endif
						}
					}
				}
				else {
					continue;
				}
			}
			#endif

			///--------------------------------- Spectra
			#if defined(__ENRG_SPECT) || defined(__ENST_SPECT) || defined(__ENRG_FLUX_SPECT) || defined(__ENST_FLUX_SPECT)
			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)) {
				// Compute |k|^2
				k_sqr = ((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				if ((j == 0) || (j == Ny_Fourier - 1)) {
					// Update the current bin
					#if defined(__ENRG_SPECT)
					run_data->enrg_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
					#endif
					#if defined(__ENST_SPECT)
					run_data->enst_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
					#endif
					#if defined(__ENST_FLUX_SPECT)
					run_data->enst_flux_spect[spec_indx] += const_fac * norm_fac * creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]);
					#endif
					#if defined(__ENRG_FLUX_SPECT)
					run_data->enrg_flux_spect[spec_indx] += const_fac * norm_fac * creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]) * (1.0 / k_sqr);
					#endif
				}
				else {
					// Update the spectra sums for the current mode
					#if defined(__ENRG_SPECT)
					run_data->enrg_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * (1.0 / k_sqr);
					#endif
					#if defined(__ENST_SPECT)
					run_data->enst_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
					#endif
					#if defined(__ENST_FLUX_SPECT)
					run_data->enst_flux_spect[spec_indx] += 2.0 * const_fac * norm_fac * creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]);
					#endif
					#if defined(__ENRG_FLUX_SPECT)
					run_data->enrg_flux_spect[spec_indx] += 2.0 * const_fac * norm_fac * creal(run_data->w_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->w_hat[indx]) * RK_data->RK1[indx]) * (1.0 / k_sqr);
					#endif
				}
			}
			else {
				continue;
			}
			#endif
		}
	}

	// ------------------------------------
	// Normalize Measureables 
	// ------------------------------------	
	#if defined(__SYS_MEASURES)
	if (iter < sys_vars->num_print_steps) {
		// Normalize results and take into account computation in Fourier space
		run_data->enrg_diss[iter]  *= 2.0 * const_fac * norm_fac;
		run_data->enst_diss[iter]  *= 2.0 * const_fac * norm_fac;
		run_data->tot_enstr[iter]  *= const_fac * norm_fac;
		run_data->tot_palin[iter]  *= const_fac * norm_fac;
		run_data->tot_forc[iter]   *= const_fac * norm_fac;
		run_data->tot_div[iter]    *= const_fac * norm_fac;
		run_data->tot_energy[iter] *= const_fac * norm_fac;
	}
	#endif
	#if defined(__ENRG_FLUX)
	run_data->enrg_flux_sbst[iter] *= const_fac * norm_fac;
	run_data->enrg_diss_sbst[iter] *= 2.0 * const_fac * norm_fac;
	#endif
	#if defined(__ENST_FLUX)
	run_data->enst_flux_sbst[iter] *= const_fac * norm_fac;
	run_data->enst_diss_sbst[iter] *= 2.0 * const_fac * norm_fac;
	#endif
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
	#if defined(__TIME)
	if (!(sys_vars->rank)) {
		run_data->time[print_indx] = t;
	}
	#endif

	// Check if within memory limits
	if (print_indx < sys_vars->num_print_steps) {
		#if defined(__SYS_MEASURES)
		// Total Energy, enstrophy and palinstrophy
		run_data->tot_enstr[print_indx]  = TotalEnstrophy();
		run_data->tot_energy[print_indx] = TotalEnergy();
		run_data->tot_palin[print_indx]  = TotalPalinstrophy();
		// Total Forcing and Divergence
		run_data->tot_forc[print_indx] = TotalForcing();
		run_data->tot_div[print_indx]    = TotalDivergence();
		// Energy and enstrophy dissipation rates
		run_data->enrg_diss[print_indx] = EnergyDissipationRate();
		run_data->enst_diss[print_indx] = EnstrophyDissipationRate();
		#endif
		#if defined(__ENST_FLUX)
		// Enstrophy and energy flux in/out and dissipation of a subset of modes
		EnstrophyFlux(&(run_data->enst_flux_sbst[print_indx]), &(run_data->enst_diss_sbst[print_indx]), RK_data);
		#endif
		#if defined(__ENRG_FLUX)
		EnergyFlux(&(run_data->enrg_flux_sbst[print_indx]), &(run_data->enrg_diss_sbst[print_indx]), RK_data);
		#endif
	}
	else {
		printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", print_indx, t);
	}
	
	// -------------------------------
	// Record the Spectra 
	// -------------------------------
	// Call spectra functions
	#if defined(__ENST_SPECT )
	EnstrophySpectrum();
	#endif
	#if defined(__ENRG_SPECT)
	EnergySpectrum();
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	EnergyFluxSpectrum(RK_data);
	#endif
	#if defined(__ENST_FLUX_SPECT)
	EnstrophyFluxSpectrum(RK_data);
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
	#if defined(PHASE_ONLY)
	// Allocate array for the Fourier amplitudes
	run_data->a_k = (double* )fftw_malloc(sizeof(double) * sys_vars->alloc_local_batch);
	if (run_data->a_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Amplitudes");
		exit(1);
	}
	// Allocate array for the Fourier phases
	run_data->phi_k = (double* )fftw_malloc(sizeof(double) * sys_vars->alloc_local_batch);
	if (run_data->phi_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Phases");
		exit(1);
	}
	// Allocate array for the Fourier phases
	run_data->tmp_a_k = (double* )fftw_malloc(sizeof(double) * sys_vars->alloc_local_batch);
	if (run_data->tmp_a_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Tmp Amplitudes");
		exit(1);
	}
	#endif
	#if defined(TESTING)
	// Allocate array for the taylor green solution
	run_data->tg_soln = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->tg_soln == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Taylor Green vortex solution");
		exit(1);
	}
	#endif
	#if defined(__NONLIN)
	// Allocate memory for recording the nonlinear term
	run_data->nonlinterm = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->nonlinterm == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Nonlinear Term");
		exit(1);
	}
	#endif
	#if defined(__RHS)
	// Allocate memory for recording the RHS
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
	#if defined(__RK5) || defined(__DPRK5)
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
	#if defined(__DPRK5)
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
			#if defined(TESTING)
			run_data->tg_soln[indx_real]                = 0.0;
			#endif
			run_data->w[indx_real]                      = 0.0;
			if (j < Ny_Fourier) {
				#if defined(PHASE_ONLY)
				run_data->a_k[indx_four] 				= 0.0;
				run_data->phi_k[indx_four]			    = 0.0;
				run_data->tmp_a_k[indx_four] 			= 0.0;
				#endif
				run_data->w_hat[indx_four]               	  = 0.0 + 0.0 * I;
				RK_data->RK_tmp[indx_four]    			 	  = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 0] 	  = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 1] 	  = 0.0 + 0.0 * I;
				RK_data->RK1[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				RK_data->RK1[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				RK_data->RK2[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				RK_data->RK2[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				RK_data->RK3[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				RK_data->RK3[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				RK_data->RK4[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				RK_data->RK4[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				#if defined(__NONLIN)
				run_data->nonlinterm[SYS_DIM * indx_four + 0] = 0.0 + 0.0 * I;
				run_data->nonlinterm[SYS_DIM * indx_four + 1] = 0.0 + 0.0 * I;
				#endif
				#if defined(__RK5)
				RK_data->RK5[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				RK_data->RK5[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				RK_data->RK6[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				RK_data->RK6[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				#endif
				#if defined(__DPRK5)
				RK_data->RK7[SYS_DIM * indx_four + 0]    	 = 0.0 + 0.0 * I;
				RK_data->RK7[SYS_DIM * indx_four + 1]    	 = 0.0 + 0.0 * I;
				RK_data->w_hat_last[indx_four]			 	 = 0.0 + 0.0 * I;
				#endif
			}
			if (i == 0) {
				if (j < Ny_Fourier) {
					run_data->k[1][j] = 0;
				}
				run_data->x[1][j] = 0.0;
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
 * Function to initialize the forcing variables and arrays, indentify the local forcing processes and the number of forced modes
 */
void InitializeForcing(void) {

	// Initialize variables
	int tmp, indx;
	double k_abs;
	double scale_fac_f0;
	int num_forced_modes   = 0;
	int force_mode_counter = 0;
	double sum_k_pow       = 0.0;
	double scaling_exp     = 0.0;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -----------------------------------
	// Initialize Forcing Objects 
	// -----------------------------------
	//--------------------------------- Apply Kolmogorov forcing
	if(!(strcmp(sys_vars->forcing, "KOLM"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			if (run_data->k[0][i] == 0) {
				sys_vars->local_forcing_proc = 1;
				num_forced_modes++;
			}
			else {
				sys_vars->local_forcing_proc = 0;
			}
		}

		// Get the number of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}

			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			// Get the forcing index
			run_data->forcing_indx[0] = sys_vars->force_k;

			// Get the forcing wavenumbers
			run_data->forcing_k[0][0] = 0;
			run_data->forcing_k[0][1] = run_data->k[1][sys_vars->force_k];

			// Get the forcing scaling 
			run_data->forcing_scaling[0] = 1.0;
		}
	}
	//--------------------------------- Apply Stochastic forcing
	else if(!(strcmp(sys_vars->forcing, "STOC"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				// Compute |k|
				k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				// Count the forced modes
				if ((k_abs > STOC_FORC_K_MIN && k_abs < STOC_FORC_K_MAX) && (run_data->k[1][j] != 0 || run_data->k[0][i] > 0)) {
					sum_k_pow += pow(k_abs, 2.0 * scaling_exp);
					num_forced_modes++;
				}
			}
		}

		// Get count of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// Sync sum of forced wavenumbers
		MPI_Allreduce(MPI_IN_PLACE, &sum_k_pow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}


			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			// Initialize variables
			scale_fac_f0       = sqrt(sys_vars->force_scale_var / (2.0 * sum_k_pow));
			force_mode_counter = 0;
			for (int i = 0; i < sys_vars->local_Nx; ++i) {
				tmp = i * Ny_Fourier;
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;

					// Compute |k|
					k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

					// Record the data for the forced modes
					if ((k_abs > STOC_FORC_K_MIN && k_abs < STOC_FORC_K_MAX) && (run_data->k[1][j] != 0 || run_data->k[0][i] > 0)) {
						run_data->forcing_scaling[force_mode_counter] = scale_fac_f0 * pow(k_abs, scaling_exp);
						run_data->forcing_indx[force_mode_counter]    = indx;
						run_data->forcing_k[0][force_mode_counter]    = run_data->k[0][i];
						run_data->forcing_k[1][force_mode_counter]    = run_data->k[1][j];
						force_mode_counter++;
					}
				}
			}
		}
	} 
	//--------------------------------- If ZERO modes forcing selected
	else if(!(strcmp(sys_vars->forcing, "ZERO"))) {
		// Loop through modes to identify local process(es) containing the modes to be forced
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {

				// Count the forced modes
				if ((abs(run_data->k[0][i]) <= sys_vars->force_k) || (abs(run_data->k[1][j]) <= sys_vars->force_k)) {
					sys_vars->local_forcing_proc = 1;
					num_forced_modes++;		
				}
			}
		}

		// Get the number of forced modes
		sys_vars->num_forced_modes = num_forced_modes;

		// Allocate forcing data on the local forcing process only
		if (sys_vars->local_forcing_proc) {
			// -----------------------------------
			// Allocate Memory 
			// -----------------------------------
			// Allocate the forcing array to hold the forcing
			run_data->forcing = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_forced_modes);
			if (run_data->forcing == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing");
				exit(1);
			}
			// Allocate array for the forced mode index
			run_data->forcing_indx = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
			if (run_data->forcing_indx == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Indices");
				exit(1);
			}
			// Allocate array for the scaling 
			run_data->forcing_scaling = (double* )fftw_malloc(sizeof(double) * num_forced_modes);
			if (run_data->forcing_scaling == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Scaling");
				exit(1);
			}
			// Allocate array for the forced mode wavenumbers
			for (int i = 0; i < SYS_DIM; ++i) {
				run_data->forcing_k[i] = (int* )fftw_malloc(sizeof(int) * num_forced_modes);
				if (run_data->forcing_k[i] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Forcing Wavevectors");
					exit(1);
				}
			}

			// -----------------------------------
			// Fill Forcing Info
			// -----------------------------------
			force_mode_counter = 0;
			for (int i = 0; i < sys_vars->local_Nx; ++i) {
				tmp = i * Ny_Fourier;
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;

					// Record the data for the forced modes
					if ((abs(run_data->k[0][i]) <= sys_vars->force_k) || (abs(run_data->k[1][j]) <= sys_vars->force_k)) {
						run_data->forcing[force_mode_counter]         = 0.0 + 0.0 * I;
						run_data->forcing_scaling[force_mode_counter] = 0.0;
						run_data->forcing_indx[force_mode_counter]    = indx;
						run_data->forcing_k[0][force_mode_counter]    = run_data->k[0][i];
						run_data->forcing_k[1][force_mode_counter]    = run_data->k[1][j];
						force_mode_counter++;
					}
				}
			}
		}
	}
	//--------------------------------- No forcing selected
	else {
		// Set number of forced modes and local forcing processes
		sys_vars->num_forced_modes   = 0;
		sys_vars->local_forcing_proc = 0;
	}
}
/**
 * Function to initialize and compute the system measurables and spectra of the initial conditions
 * @param RK_data The struct containing the Runge-Kutta arrays to compute the nonlinear term for the fluxes
 */
void InitializeSystemMeasurables(RK_data_struct* RK_data) {

	// Set the size of the arrays to twice the number of printing steps to account for extra steps due to adaptive stepping
	#if defined(__ADAPTIVE_STEP)
	sys_vars->num_print_steps = 2 * sys_vars->num_print_steps;
	#else
	sys_vars->num_print_steps = sys_vars->num_print_steps;
	#endif
	int print_steps = sys_vars->num_print_steps;

	// Get the size of the spectrum arrays
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];

	sys_vars->n_spect = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0)) + 1;
	int n_spect = sys_vars->n_spect;
	#endif
		
	// ------------------------
	// Allocate Memory
	// ------------------------
	#if defined(__SYS_MEASURES)
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

	// Total Forcing Input
	run_data->tot_forc = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_forc == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Forcing Input");
		exit(1);
	}	

	// Total Divergence
	run_data->tot_div = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_div == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Divergence");
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
	#endif
	#if defined(__ENST_SPECT)
	// Enstrophy Spectrum
	run_data->enst_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_SPECT)
	// Energy Spectrum
	run_data->enrg_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENST_FLUX)
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
	#endif
	#if defined(__ENRG_FLUX)
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
	#endif
	// Time
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		run_data->time = (double* )fftw_malloc(sizeof(double) * print_steps);
		if (run_data->time == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time");
			exit(1);
		}	
	}
	#endif
	#if defined(__ENST_FLUX_SPECT)
	// Enstrophy Spectrum
	run_data->enst_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	// Energy Spectrum
	run_data->enrg_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}	
	#endif

	// ----------------------------
	// Get Measurables of the ICs
	// ----------------------------
	#if defined(__SYS_MEASURES)
	// Total Energy
	run_data->tot_energy[0] = TotalEnergy();

	// Total Enstrophy
	run_data->tot_enstr[0] = TotalEnstrophy();

	// Total Palinstrophy
	run_data->tot_palin[0] = TotalPalinstrophy();

	// Total Forcing Input
	run_data->tot_forc[0] = TotalForcing();

	// Total Divergence
	run_data->tot_div[0] = TotalDivergence();

	// Energy dissipation rate
	run_data->enrg_diss[0] = EnergyDissipationRate();

	// Enstrophy dissipation rate
	run_data->enst_diss[0] = EnstrophyDissipationRate();
	#endif
	#if defined(__ENST_FLUX)
	// Enstrophy Flux and dissipation from/to Subset of modes
	EnstrophyFlux(&(run_data->enst_flux_sbst[0]), &(run_data->enst_diss_sbst[0]), RK_data);
	#endif
	#if defined(__ENRG_FLUX)
	// Energy Flux and dissipation from/to a subset of modes
	EnergyFlux(&(run_data->enrg_flux_sbst[0]), &(run_data->enrg_diss_sbst[0]), RK_data);
	#endif
	// Time
	#if defined(__TIME)
	if (!(sys_vars->rank)) {
		run_data->time[0] = sys_vars->t0;
	}
	#endif

	// ----------------------------
	// Get Spectra of the ICs
	// ----------------------------
	// Call spectra functions
	#if defined(__ENST_SPECT)
	EnstrophySpectrum();
	#endif
	#if defined(__ENRG_SPECT)
	EnergySpectrum();
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	EnergyFluxSpectrum(RK_data);
	#endif
	#if defined(__ENST_FLUX_SPECT)
	EnstrophyFluxSpectrum(RK_data);
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
		if (sys_vars->local_forcing_proc) {
			fftw_free(run_data->forcing_k);
		}
	}

	// Free system variables
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	fftw_free(run_data->w);
	fftw_free(run_data->w_hat);
	if (sys_vars->local_forcing_proc) {
		fftw_free(run_data->forcing);
		fftw_free(run_data->forcing_indx);
		fftw_free(run_data->forcing_scaling);
	}
	#if defined(PHASE_ONLY)
	fftw_free(run_data->a_k);
	fftw_free(run_data->phi_k);
	fftw_free(run_data->tmp_a_k);
	#endif
	#if defined(__NONLIN)
	fftw_free(run_data->nonlinterm);
	#endif
	#if defined(__SYS_MEASURES)
	fftw_free(run_data->tot_energy);
	fftw_free(run_data->tot_enstr);
	fftw_free(run_data->tot_div);
	fftw_free(run_data->tot_forc);
	fftw_free(run_data->tot_palin);
	fftw_free(run_data->enrg_diss);
	fftw_free(run_data->enst_diss);
	#endif
	#if defined(__ENST_FLUX)
	fftw_free(run_data->enst_flux_sbst);
	fftw_free(run_data->enst_diss_sbst);
	#endif
	#if defined(__ENRG_FLUX)
	fftw_free(run_data->enrg_flux_sbst);
	fftw_free(run_data->enrg_diss_sbst);
	#endif
	#if defined(__ENRG_SPECT)
	fftw_free(run_data->enrg_spect);
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	fftw_free(run_data->enrg_flux_spect);
	#endif
	#if defined(__ENST_SPECT)
	fftw_free(run_data->enst_spect);
	#endif
	#if defined(__ENST_FLUX_SPECT)
	fftw_free(run_data->enst_flux_spect);
	#endif
	#if defined(TESTING)
	fftw_free(run_data->tg_soln);
	#endif
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		fftw_free(run_data->time);
	}
	#endif

	// Free integration variables
	fftw_free(RK_data->RK1);
	fftw_free(RK_data->RK2);
	fftw_free(RK_data->RK3);
	fftw_free(RK_data->RK4);
	#if defined(__RK5) || defined(__DPRK5)
	fftw_free(RK_data->RK5);
	fftw_free(RK_data->RK6);
	#endif 
	#if defined(__DPRK5)
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