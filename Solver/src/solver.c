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
#include "sys_msr.h"
#include "force.h"
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// Define RK4 variables - Butcher Tableau
#if defined(__RK4) || defined(__AB4) || defined(__RK4CN)
static const double RK4_C2 = 0.5, 	  RK4_A21 = 0.5, \
				  	RK4_C3 = 0.5,	           					RK4_A32 = 0.5, \
				  	RK4_C4 = 1.0,                      									   RK4_A43 = 1.0, \
				              	 	  RK4_B1 = 1.0/6.0, 		RK4_B2  = 1.0/3.0, 		   RK4_B3  = 1.0/3.0, 		RK4_B4 = 1.0/6.0;
// Define RK5 Dormand Prince variables - Butcher Tableau
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
#if defined(__AB4)
static const double AB4_1 = 55.0/24.0, AB4_2 = -59.0/24.0,		AB4_3 = 37.0/24.0,			AB4_4 = -3.0/8.0;
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
	struct Int_data_struct* Int_data;	    // Initialize pointer to a Int_data_struct
	struct Int_data_struct Int_data_tmp; 	// Initialize a Int_data_struct
	Int_data = &Int_data_tmp;		        // Point the ptr to this new Int_data_struct

	// -------------------------------
	// Allocate memory
	// -------------------------------
	AllocateMemory(NBatch, Int_data);

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

	// Get initial conditions - seed for random number generator is set here
	InitialConditions(run_data->w_hat, run_data->u, run_data->u_hat, N);

	// // Initialize the forcing 
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
	InitializeSystemMeasurables(Int_data);


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
	t         += dt;
	int iters = 1;
	int save_data_indx;
	if (sys_vars->TRANS_ITERS_FLAG == TRANSIENT_ITERS) {
		save_data_indx = 0;
	}
	else {
		save_data_indx = 1;
	}
	while (t <= T) {

		// -------------------------------	
		// Integration Step
		// -------------------------------
		#if defined(__RK4) || defined(__RK4CN)
		RK4Step(dt, N, sys_vars->local_Ny, Int_data);
		#elif defined(__AB4)
		AB4Step(dt, N, iters, sys_vars->local_Ny, Int_data);
		#elif defined(__RK5)
		RK5DPStep(dt, N, iters, sys_vars->local_Ny, Int_data);
		#elif defined(__DPRK5)
		while (try) {
			// Try a Dormand Prince step and compute the local error
			RK5DPStep(dt, N, iters, sys_vars->local_Ny, Int_data);

			// Compute the new timestep
			dt_new = dt * DPMin(DP_DELTA_MAX, DPMax(DP_DELTA_MIN, DP_DELTA * pow(1.0 / Int_data->DP_err, 0.2)));
			
			// If error is bad repeat else move on
			if (Int_data->DP_err < 1.0) {
				Int_data->DP_fails++;
				dt = dt_new;
				continue;
			}
			else {
				dt = dt_new;
				break;
			}
		}
		#endif

		// -------------------------------
		// Write To File
		// -------------------------------
		if (iters % sys_vars->SAVE_EVERY == 0) {
			#if defined(TESTING)
			TaylorGreenSoln(t, N);
			#endif

			// Record System Measurables
			ComputeSystemMeasurables(t, save_data_indx, Int_data);

			// If and when transient steps are complete write to file
			if (iters > trans_steps) {
				// // Write the appropriate datasets to file 
				WriteDataToFile(t, dt, save_data_indx);
				
				// Update saving data index
				save_data_indx++;
			}
		}
		// -------------------------------
		// Print Update To Screen
		// -------------------------------
		#if defined(__PRINT_SCREEN)
		if (sys_vars->TRANS_ITERS_FLAG == TRANSIENT_ITERS) {
			// Print update that transient iters have been complete
			if ((iters == trans_steps) && !(sys_vars->rank)) {
				printf("\n\n...Transient Iterations Complete!\n\n");
			}
		}
		if (iters % sys_vars->SAVE_EVERY == 0) {
			// Print update of the system to the terminal 
			if (iters <= sys_vars->trans_iters) {
				// If performing transient iterations the system measures are stored in the 0th index
				PrintUpdateToTerminal(iters, t, dt, T, 0);
			}
			else {
				// Print update of the non transient iterations to the terminal 
				PrintUpdateToTerminal(iters, t, dt, T, save_data_indx - 1);
			}
		}
		#endif

		// -------------------------------
		// Update & System Check
		// -------------------------------
		// Update timestep & iteration counter
		iters++;
		if (sys_vars->ADAPT_STEP_FLAG == ADAPTIVE_STEP) {
			GetTimestep(&dt);
			t += dt; 
		}
		else {
			#if defined(__DPRK5)
			t += dt;
			#else
			t = iters * dt;
			#endif
		}

		// Check System: Determine if system has blown up or integration limits reached
		SystemCheck(dt, iters);
	}
	//////////////////////////////
	// End Integration
	//////////////////////////////
		
	// ------------------------------- 
	// Final Writes to Output File
	// -------------------------------
	FinalWriteAndCloseOutputFiles(N, iters, save_data_indx);
	

	// -------------------------------
	// Clean Up 
	// -------------------------------
	FreeMemory(Int_data);
}
#if defined(__AB4)
/**
 * Function to perform an integration step using the 4th order Adams Bashforth scheme
 * @param dt       The current timestep of the system
 * @param N        Array containing the size of each dimension
 * @param iters    The current iteration of the system
 * @param local_Ny The size of the first dimension for the local (to each process) arrays
 * @param Int_data Struct containing the integration arrays
 */
void AB4Step(const double dt, const long int* N, const int iters, const ptrdiff_t local_Ny, Int_data_struct* Int_data) {

	// Initialize vairables
	int tmp;
	int indx;
	#if defined(__NAVIER)
	double k_sqr;
	double D_fac;
	#endif
	const long int Nx_Fourier = N[1] / 2 + 1;

	/////////////////////
	/// AB Pre Steps
	/////////////////////
	if (iters <= Int_data->AB_pre_steps) {
		// -----------------------------------
		// Perform RK4 Step
		// -----------------------------------
		// March the vorticity forward in time using RK4 step
		RK4Step(dt, N, sys_vars->local_Ny, Int_data);

		// Save the nonlinear term for each pre step for use in the update step of the AB4 scheme
		memcpy(&(Int_data->AB_tmp_nonlin[iters - 1][0]), Int_data->RK1, sizeof(fftw_complex) * (long int)local_Ny * Nx_Fourier);
	}
	else {
		// -----------------------------------
		// Compute Forcing
		// -----------------------------------
		// Compute the forcing for the current iteration
		ComputeForcing(dt);

		// -----------------------------------
		// Compute Nonlinear Term
		// -----------------------------------
		// Get the nonlinear term for the current step
		NonlinearRHSBatch(run_data->w_hat, Int_data->AB_tmp, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);


		/////////////////////
		/// AB Update Step
		/////////////////////
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
					#if defined(__EULER)
					// Update the Fourier space vorticity with the RHS
					run_data->w_hat[indx] = run_data->w_hat[indx] + dt * (AB4_1 * Int_data->AB_tmp[indx]) + dt * (AB4_2 * Int_data->AB_tmp_nonlin[2][indx]) + dt * (AB4_3 * Int_data->AB_tmp_nonlin[1][indx]) + dt * (AB4_4 * Int_data->AB_tmp_nonlin[0][indx]);
					#elif defined(__NAVIER)
					// Compute the pre factors for the RK4CN update step
					k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
					
					if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
						// Both Hyperviscosity and Ekman drag
						D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
					}
					else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
						// No hyperviscosity but we have Ekman drag
						D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
					}
					else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
						// Hyperviscosity only
						D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW)); 
					}
					else { 
						// No hyper viscosity or no ekman drag -> just normal viscosity
						D_fac = dt * (sys_vars->NU * k_sqr); 
					}
					
					// Complete the update step
					run_data->w_hat[indx] = run_data->w_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * ((AB4_1 * Int_data->AB_tmp[indx]) + (AB4_2 * Int_data->AB_tmp_nonlin[2][indx]) + (AB4_3 * Int_data->AB_tmp_nonlin[1][indx]) + (AB4_4 * Int_data->AB_tmp_nonlin[0][indx]));
					#endif
				}
				else {
					run_data->w_hat[indx] = 0.0 + 0.0 * I;
				}
			}
		}

		// -----------------------------------
		// Update Previous Nonlinear Terms
		// -----------------------------------
		// Update the previous Nonlinear term arrays for next iteration
		memcpy(&(Int_data->AB_tmp_nonlin[0][0]), &(Int_data->AB_tmp_nonlin[1][0]), sizeof(fftw_complex) * (long int)local_Ny * Nx_Fourier);
		memcpy(&(Int_data->AB_tmp_nonlin[1][0]), &(Int_data->AB_tmp_nonlin[2][0]), sizeof(fftw_complex) * (long int)local_Ny * Nx_Fourier);
		memcpy(&(Int_data->AB_tmp_nonlin[2][0]), Int_data->AB_tmp, sizeof(fftw_complex) * (long int)local_Ny * Nx_Fourier);
	}
}
#endif
/**
 * Function to perform a single step of the RK5 or Dormand Prince scheme
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Ny Int indicating the local size of the first dimension of the arrays	
 * @param Int_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Ny, Int_data_struct* Int_data) {


	// Initialize vairables
	int tmp;
	int indx;
	#if defined(__NAVIER)
	double k_sqr;
	double D_fac;
	#endif
	const long int Nx_Fourier = N[1] / 2 + 1;
	#if defined(__DPRK5)
	const long int Ny = N[0];
	double dp_ho_step;
	double err_sum;
	double err_denom;
	#endif
	
	//------------------- Pre-record the amplitudes so they can be reset after update step
	#if defined(PHASE_ONLY)
	double tmp_a_k_norm;

	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// record amplitudes
			run_data->tmp_a_k[indx] = cabs(run_data->w_hat[indx]);
		}
	}
	#endif

	//------------------- Get the forcing
	ComputeForcing(dt);
	
	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHSBatch(run_data->w_hat, Int_data->RK1, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A21 * Int_data->RK1[indx];
		}
	}
	// ----------------------- Stage 2
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK2, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A31 * Int_data->RK1[indx] + dt * RK5_A32 * Int_data->RK2[indx];
		}
	}
	// ----------------------- Stage 3
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK3, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A41 * Int_data->RK1[indx] + dt * RK5_A42 * Int_data->RK2[indx] + dt * RK5_A43 * Int_data->RK3[indx];
		}
	}
	// ----------------------- Stage 4
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK4, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A51 * Int_data->RK1[indx] + dt * RK5_A52 * Int_data->RK2[indx] + dt * RK5_A53 * Int_data->RK3[indx] + dt * RK5_A54 * Int_data->RK4[indx];
		}
	}
	// ----------------------- Stage 5
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK5, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A61 * Int_data->RK1[indx] + dt * RK5_A62 * Int_data->RK2[indx] + dt * RK5_A63 * Int_data->RK3[indx] + dt * RK5_A64 * Int_data->RK4[indx] + dt * RK5_A65 * Int_data->RK5[indx];
		}
	}
	// ----------------------- Stage 6
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK6, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	#if defined(__DPRK5)
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + dt * RK5_A71 * Int_data->RK1[indx] + dt * RK5_A73 * Int_data->RK3[indx] + dt * RK5_A74 * Int_data->RK4[indx] + dt * RK5_A75 * Int_data->RK5[indx] + dt * RK5_A76 * Int_data->RK6[indx];
		}
	}
	// ----------------------- Stage 7
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK7, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	#endif

	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				#if defined(PHASE_ONLY)
				tmp_a_k_norm = cabs(run_data->w_hat[indx]);
				#endif

				#if defined(__EULER)
				// Update the Fourier space vorticity with the RHS
				run_data->w_hat[indx] = run_data->w_hat[indx] + (dt * (RK5_B1 * Int_data->RK1[indx]) + dt * (RK5_B3 * Int_data->RK3[indx]) + dt * (RK5_B4 * Int_data->RK4[indx]) + dt * (RK5_B5 * Int_data->RK5[indx]) + dt * (RK5_B6 * Int_data->RK6[indx]));
				#elif defined(__NAVIER)
				// Compute the pre factors for the RK4CN update step
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
				
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// Both Hyperviscosity and Ekman drag
					D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// No hyperviscosity but we have Ekman drag
					D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
				}
				else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
					// Hyperviscosity only
					D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW)); 
				}
				else { 
					// No hyper viscosity or no ekman drag -> just normal viscosity
					D_fac = dt * (sys_vars->NU * k_sqr); 
				}
				
				// Complete the update step
				run_data->w_hat[indx] = run_data->w_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK5_B1 * Int_data->RK1[indx] + RK5_B3 * Int_data->RK3[indx] + RK5_B4 * Int_data->RK4[indx] + RK5_B5 * Int_data->RK5[indx] + RK5_B6 * Int_data->RK6[indx]);
				#endif
				#if defined(PHASE_ONLY)
				// Reset the amplitudes
				run_data->w_hat[indx] *= (tmp_a_k_norm / cabs(run_data->w_hat[indx]));
				#endif
				#if defined(__DPRK5)
				if (iters > 1) {
					// Get the higher order update step
					dp_ho_step = run_data->w_hat[indx] + dt * (RK5_Bs1 * Int_data->RK1[indx]) + dt * (RK5_Bs3 * Int_data->RK3[indx]) + dt * (RK5_Bs4 * Int_data->RK4[indx]) + dt * (RK5_Bs5 * Int_data->RK5[indx]) + dt * (RK5_Bs6 * Int_data->RK6[indx]) + dt * (RK5_Bs7 * Int_data->RK7[indx]);
					#if defined(PHASE_ONLY)
					// Reset the amplitudes
					dp_ho_step *= (tmp_a_k_norm / cabs(run_data->w_hat[indx]))
					#endif

					// Denominator in the error
					err_denom = DP_ABS_TOL + DPMax(cabs(Int_data->w_hat_last[indx]), cabs(run_data->w_hat[indx])) * DP_REL_TOL;

					// Compute the sum for the error
					err_sum += pow((run_data->w_hat[indx] - dp_ho_step) /  err_denom, 2.0);
				}
				#endif
			}
			else {
				run_data->w_hat[indx] = 0.0 + 0.0 * I;
			}
		}
	}
	#if defined(__NONLIN)
	// Record the nonlinear for the updated Fourier vorticity
	NonlinearRHSBatch(run_data->w_hat, run_data->nonlinterm, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			run_data->nonlinterm[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}
	#endif
	#if defined(__DPRK5)
	if (iters > 1) {
		// Reduce and sync the error sum across the processes
		MPI_Allreduce(MPI_IN_PLACE, &err_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Compute the error
		Int_data->DP_err = sqrt(1.0/ (Ny * Nx_Fourier) * err_sum);

		// Record the Fourier vorticity for the next step
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
					// Record the vorticity
					Int_data->w_hat_last[indx] = run_data->w_hat[indx];
				}
				else {
					run_data->w_hat_last[indx] = 0.0 + 0.0 * I;
				}
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
 * @param local_Ny Int indicating the local size of the first dimension of the arrays	
 * @param Int_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK4) || defined(__RK4CN) || defined(__AB4)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Ny, Int_data_struct* Int_data) {

	// Initialize vairables
	int tmp;
	int indx;
	double k_sqr;
	double D_fac;
	const long int Nx_Fourier = N[1] / 2 + 1;

	//------------------- Pre-record the amplitudes so they can be reset after update step
	#if defined(PHASE_ONLY)
	double tmp_a_k_norm;

	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// record amplitudes
			run_data->tmp_a_k[indx] = cabs(run_data->w_hat[indx]);
		}
	}
	#endif

	//------------------- Get the forcing
	ComputeForcing(dt);


	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHSBatch(run_data->w_hat, Int_data->RK1, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Add linear term to RHS
			#if defined(__RK4)
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// Both Hyperviscosity and Ekman drag
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
			}
			else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// No hyperviscosity but we have Ekman drag
				D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW));
			}
			else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
				// Hyperviscosity only
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW));
			}
			else {
				// No hyper viscosity or no ekman drag -> just normal viscosity
				D_fac = dt * (sys_vars->NU * k_sqr);
			}

			Int_data->RK1[indx] += D_fac * run_data->w_hat[indx];
			#endif


			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + RK4_A21 * (dt * Int_data->RK1[indx]);
			// printf("RHS1[%d, %d]: %1.16lf %1.16lf -- (%d, %d) -- wh[%d, %d]: %1.16lf %1.16lf \n", run_data->k[0][i], run_data->k[1][j], creal(Int_data->RK1[indx] * dt), cimag(Int_data->RK1[indx] * dt), run_data->k[0][i], run_data->k[1][j], run_data->k[0][i], run_data->k[1][j], creal(Int_data->RK_tmp[indx]), cimag(Int_data->RK_tmp[indx]));
		}
		// printf("\n");
	}
	// printf("\n");
	// PrintScalarFourier(Int_data->RK1, N, "RK1");
	// PrintScalarFourier(Int_data->RK_tmp, N, "wh");

	// ----------------------- Stage 2
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK2, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Add linear term to RHS
			#if defined(__RK4)
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// Both Hyperviscosity and Ekman drag
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
			}
			else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// No hyperviscosity but we have Ekman drag
				D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW));
			}
			else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
				// Hyperviscosity only
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW));
			}
			else {
				// No hyper viscosity or no ekman drag -> just normal viscosity
				D_fac = dt * (sys_vars->NU * k_sqr);
			}

			Int_data->RK2[indx] += D_fac * Int_data->RK_tmp[indx];
			#endif


			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + RK4_A32 * (dt * Int_data->RK2[indx]);
			// printf("RHS2[%d, %d]: %1.16lf %1.16lf -- (%d, %d)\n", run_data->k[0][i], run_data->k[1][j], creal(Int_data->RK2[indx] * dt), cimag(Int_data->RK2[indx] * dt), run_data->k[0][i], run_data->k[1][j]);
		}
		// printf("\n");
	}
	// printf("\n");
	// PrintScalarFourier(Int_data->RK2, N, "RK2");
	// PrintScalarFourier(Int_data->RK_tmp, N, "wh");

	// ----------------------- Stage 3
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK3, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Add linear term to RHS
			#if defined(__RK4)
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// Both Hyperviscosity and Ekman drag
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
			}
			else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// No hyperviscosity but we have Ekman drag
				D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW));
			}
			else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
				// Hyperviscosity only
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW));
			}
			else {
				// No hyper viscosity or no ekman drag -> just normal viscosity
				D_fac = dt * (sys_vars->NU * k_sqr);
			}

			Int_data->RK3[indx] += D_fac * Int_data->RK_tmp[indx];
			#endif


			// Update temporary input for nonlinear term
			Int_data->RK_tmp[indx] = run_data->w_hat[indx] + RK4_A43 * (dt * Int_data->RK3[indx]);
			// printf("RHS3[%d, %d]: %1.16lf %1.16lf -- (%d, %d)\n", run_data->k[0][i], run_data->k[1][j], creal(Int_data->RK3[indx] * dt), cimag(Int_data->RK3[indx] * dt), run_data->k[0][i], run_data->k[1][j]);
		}
		// printf("\n");
	}
	// printf("\n");
	// ----------------------- Stage 4
	NonlinearRHSBatch(Int_data->RK_tmp, Int_data->RK4, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	#if defined(__RK4)
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Add linear term to RHS
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// Both Hyperviscosity and Ekman drag
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
			}
			else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
				// No hyperviscosity but we have Ekman drag
				D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW));
			}
			else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
				// Hyperviscosity only
				D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW));
			}
			else {
				// No hyper viscosity or no ekman drag -> just normal viscosity
				D_fac = dt * (sys_vars->NU * k_sqr);
			}

			Int_data->RK4[indx] += D_fac * Int_data->RK_tmp[indx];

			// printf("RHS4[%d, %d]: %1.16lf %1.16lf -- (%d, %d)\n", run_data->k[0][i], run_data->k[1][j], creal(Int_data->RK4[indx] * dt), cimag(Int_data->RK4[indx] * dt), run_data->k[0][i], run_data->k[1][j]);

		}
		// printf("\n");
	}
	#endif
	
	// for (int i = 0; i < local_Ny; ++i) {
	// 	tmp = i * Nx_Fourier;
	// 	for (int j = 0; j < Nx_Fourier; ++j) {
	// 		indx = tmp + j;
	// 		printf("RHS4[%d, %d]: %1.16lf %1.16lf -- (%d, %d)\n", run_data->k[0][i], run_data->k[1][j], creal(Int_data->RK4[indx] * dt), cimag(Int_data->RK4[indx] * dt), run_data->k[0][i], run_data->k[1][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n");
	// printf("\n\n\n\n\n");
	
	// printf("nu: %lf\tdt: %lf\ta: %lf\thyper_flag: %d\thyper_pow: %lf\thypo_flag: %d\thypo_pow: %lf\t\t--HYPER: %d\t\tHYPO: %d\n", sys_vars->NU, dt, sys_vars->EKMN_ALPHA, sys_vars->HYPER_VISC_FLAG, sys_vars->HYPER_VISC_POW, sys_vars->EKMN_DRAG_FLAG, sys_vars->EKMN_DRAG_POW, HYPER_VISC, EKMN_DRAG);
	
	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				#if defined(PHASE_ONLY)
				// Pre-record the amplitudes
				tmp_a_k_norm = cabs(run_data->w_hat[indx]);
				#endif

				#if defined(__EULER) || (defined(__NAVIER) && defined(__RK4))
				// Update Fourier vorticity with the RHS
				run_data->w_hat[indx] = run_data->w_hat[indx] + (dt / 6.0) * (Int_data->RK1[indx] + 2.0 * Int_data->RK2[indx] + 2.0 * Int_data->RK3[indx] + Int_data->RK4[indx]);
				// run_data->w_hat[indx] = run_data->w_hat[indx] + (dt * (RK4_B1 * Int_data->RK1[indx]) + dt * (RK4_B2 * Int_data->RK2[indx]) + dt * (RK4_B3 * Int_data->RK3[indx]) + dt * (RK4_B4 * Int_data->RK4[indx]));
				#elif defined(__NAVIER) && defined(__RK4CN)
				// Compute the pre factors for the RK4CN update step
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
				
				if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// Both Hyperviscosity and Ekman drag
					D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW)); 
				}
				else if((sys_vars->HYPER_VISC_FLAG != HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG == EKMN_DRAG)) {
					// No hyperviscosity but we have Ekman drag
					D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_DRAG_POW));
				}
				else if((sys_vars->HYPER_VISC_FLAG == HYPER_VISC) && (sys_vars->EKMN_DRAG_FLAG != EKMN_DRAG)) {
					// Hyperviscosity only
					D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->HYPER_VISC_POW));
				}
				else {
					// No hyper viscosity or no ekman drag -> just normal viscosity
					D_fac = dt * sys_vars->NU * k_sqr;
				}
				
				// Update Fourier vorticity
				run_data->w_hat[indx] = (1.0 / (1.0 + 0.5 * D_fac)) * ((1.0 - 0.5 * D_fac) * run_data->w_hat[indx] + (RK4_B1 * (Int_data->RK1[indx] * dt) + RK4_B2 * (Int_data->RK2[indx] * dt) + RK4_B3 * (Int_data->RK3[indx] * dt) + RK4_B4 * (Int_data->RK4[indx] * dt)));
				// run_data->w_hat[indx] = run_data->w_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * Int_data->RK1[indx] + RK4_B2 * Int_data->RK2[indx] + RK4_B3 * Int_data->RK3[indx] + RK4_B4 * Int_data->RK4[indx]);
				#endif
				#if defined(PHASE_ONLY)
				run_data->w_hat[indx] *= (tmp_a_k_norm / cabs(run_data->w_hat[indx]));
				#endif
			}
			else {
				run_data->w_hat[indx] = 0.0 + 0.0 * I;
			}
			// printf("d[%d,%d]: %1.6lf - %1.6lf ", run_data->k[0][i], run_data->k[1][j], 1.0/(1.0 + D_fac * 0.5), 1.0 - D_fac * 0.5);
			// printf("wh[%d,%d]: %1.16lf %1.16lfi\n", run_data->k[0][i], run_data->k[1][j], creal(run_data->w_hat[indx]), cimag(run_data->w_hat[indx]));
		}
		// printf("\n");
	}
	// PrintScalarFourier(run_data->w_hat, N, "wnew");
	#if defined(__NONLIN)
	// Record the nonlinear term with the updated Fourier vorticity
	NonlinearRHSBatch(run_data->w_hat, run_data->nonlinterm, Int_data->nonlin, Int_data->nabla_psi, Int_data->nabla_w);
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
			run_data->nonlinterm[run_data->forcing_indx[i]] -= run_data->forcing[i];
		}
	}
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
void NonlinearRHSBatch(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* nonlinear, double* u, double* nabla_w) {

	// Initialize variables
	int tmp, indx;
	const ptrdiff_t local_Ny  = sys_vars->local_Ny;
	const long int Ny         = sys_vars->N[0];
	const long int Nx         = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	double k_sqr;
	double vel1;
	double vel2;
	double norm_fac = 1.0 / (Ny * Nx);

	// Ensure conjugacy in the ky = 0 modes of the intial condition
    ForceConjugacy(w_hat, sys_vars->N, 1);

	// -----------------------------------
	// Compute Fourier Space Velocities
	// -----------------------------------
	// Compute (-\Delta)^-1 \omega - i.e., u_hat = d\psi_dy = I ky/|k|^2 \omegahat_k, v_hat = -d\psi_dx = I kx/|k|^2 \omegahat_k
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * (Nx_Fourier);
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			// Write w_hat to temporary array for transform back to real space
			run_data->w_hat_tmp[indx] = w_hat[indx];

			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
				// denominator
				k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Fill fill fourier velocities array
				dw_hat_dt[SYS_DIM * indx + 0] = I * ((double) run_data->k[0][i]) * k_sqr * w_hat[indx];
				dw_hat_dt[SYS_DIM * indx + 1] = -1.0 * I * ((double) run_data->k[1][j]) * k_sqr * w_hat[indx];
			}
			else {
				dw_hat_dt[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
				dw_hat_dt[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
			}
			// printf("uh[%d, %d]: %1.16lf %1.16lf -- vh[%d, %d]: %1.16lf %1.16lf -- wh[%d, %d]: %1.16lf %1.16lf  -- (%d, %d)\n", run_data->k[0][i], run_data->k[1][j], creal(dw_hat_dt[SYS_DIM * indx + 0]), cimag(dw_hat_dt[SYS_DIM * indx + 0]), run_data->k[0][i], run_data->k[1][j], creal(dw_hat_dt[SYS_DIM * indx + 1]), cimag(dw_hat_dt[SYS_DIM * indx + 1]), run_data->k[0][i], run_data->k[1][j], creal(w_hat[indx]), cimag(w_hat[indx]), run_data->k[0][i], run_data->k[1][j]);
		}
		// printf("\n");
	}
	// printf("\n");

	// PrintVectorFourier(dw_hat_dt, sys_vars->N, "uh", "vh");
	// PrintScalarFourier(run_data->w_hat_tmp, sys_vars->N, "wh");

	// // Ensure conjugacy in the ky = 0 modes of the intial condition
    ForceConjugacy(dw_hat_dt, sys_vars->N, 2);

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier velocites to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), dw_hat_dt, u);
	
	
	// ---------------------------------------------
	// Compute Fourier Space Vorticity Derivatives
	// ---------------------------------------------
	// // Compute \nabla\omega - i.e., d\omegahat_dx = -I kx \omegahat_k, d\omegahat_dy = -I ky \omegahat_k
	// for (int i = 0; i < local_Ny; ++i) {
	// 	tmp = i * (Nx_Fourier);
	// 	for (int j = 0; j < Nx_Fourier; ++j) {
	// 		indx = tmp + j;

	// 		// Fill vorticity derivatives array
	// 		if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
	// 			dw_hat_dt[SYS_DIM * indx + 0] = I * ((double) run_data->k[1][j]) * w_hat[indx];
	// 			dw_hat_dt[SYS_DIM * indx + 1] = I * ((double) run_data->k[0][i]) * w_hat[indx]; 
	// 		}
	// 		else {
	// 			dw_hat_dt[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
	// 			dw_hat_dt[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
	// 		}
	// 		// printf("dwdx[%d, %d]: %1.16lf %1.16lf -- dwdy[%d, %d]: %1.16lf %1.16lf -- (%d, %d)\n", i, j, creal(dw_hat_dt[SYS_DIM * indx + 0]), cimag(dw_hat_dt[SYS_DIM * indx + 0]), i, j, creal(dw_hat_dt[SYS_DIM * indx + 1]), cimag(dw_hat_dt[SYS_DIM * indx + 1]), run_data->k[0][i], run_data->k[1][j]);

	// 	}
	// 	// printf("\n");
	// }
	// // printf("\n");

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier vorticity derivatives to real space
	// fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), dw_hat_dt, nabla_w);
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->w_hat_tmp, run_data->w);
	
	// PrintVectorReal(u, sys_vars->N, "u", "v");
	// PrintScalarReal(run_data->w, sys_vars->N, "w");
	// -----------------------------------
	// Perform Convolution in Real Space
	// -----------------------------------
	// Perform the multiplication in real space
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * (Nx + 2);
		for (int j = 0; j < Nx; ++j) {
			indx = tmp + j; 
			// printf("u[%d, %d]: %1.16lf -- v[%d, %d]: %1.16lf -- w[%d, %d]: %1.16lf\n", i, j, u[SYS_DIM * indx + 0], i, j, u[SYS_DIM * indx + 1], i, j, run_data->w[indx]);

 			// // Perform multiplication of the nonlinear term 
 			// vel1 = u[SYS_DIM * indx + 0];
 			// vel2 = u[SYS_DIM * indx + 1];
 			// nonlinear[indx] = (vel1 * nabla_w[SYS_DIM * indx + 0] + vel2 * nabla_w[SYS_DIM * indx + 1]);
			// Perform multiplication of the nonlinear term 
			u[SYS_DIM * indx + 0] *= run_data->w[indx];
			u[SYS_DIM * indx + 1] *= run_data->w[indx];
 		}
 		// printf("\n");
 	}
 	// printf("\n");

 	// -------------------------------------
 	// Transform Nonlinear Term To Fourier
 	// -------------------------------------
 	// Transform Fourier nonlinear term back to Fourier space
 	// fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), nonlinear, dw_hat_dt);
 	fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_batch_r2c), u, dw_hat_dt);

	// // Ensure conjugacy in the ky = 0 modes of the intial condition
    ForceConjugacy(dw_hat_dt, sys_vars->N, 2);

 	for (int i = 0; i < local_Ny; ++i) {
 		tmp = i * (Nx_Fourier);
 		for (int j = 0; j < Nx_Fourier; ++j) {
 			indx = tmp + j;

 			if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
 				dw_hat_dt[indx] = (-I * run_data->k[1][j] * dw_hat_dt[SYS_DIM * indx + 0] -I * run_data->k[0][i] * dw_hat_dt[SYS_DIM * indx + 1]) * pow(norm_fac, 1.0);
 				// dw_hat_dt[indx] *= pow(norm_fac, 2.0);
 			}
 			else {
 				dw_hat_dt[indx] = 0.0 + 0.0 * I;
 				dw_hat_dt[indx] = 0.0 + 0.0 * I;
 			}
		}
	}

 	// ----------------------------------------
 	// Add Forcing Apply Dealiasing & Conjugacy
 	// ----------------------------------------
 	// Add the forcing
	if (sys_vars->local_forcing_proc) {
		for (int i = 0; i < sys_vars->num_forced_modes; ++i) {
	 		// printf("scale: %lf\t-\tk[%d, %d]: %1.16lf\t%1.16lfi\t\t%1.16lf\t%1.16lfi--\t%1.16lf %1.16lfi\n", sys_vars->force_scale_var, run_data->forcing_k[0][i], run_data->forcing_k[1][i], creal(dw_hat_dt[run_data->forcing_indx[i]]), cimag(dw_hat_dt[run_data->forcing_indx[i]]), creal(run_data->forcing[i]), cimag(run_data->forcing[i]), creal(run_data->forcing[i]) / creal(dw_hat_dt[run_data->forcing_indx[i]]), cimag(run_data->forcing[i]) / cimag(dw_hat_dt[run_data->forcing_indx[i]]));
			dw_hat_dt[run_data->forcing_indx[i]] += run_data->forcing[i]; 
	 		// printf("scale: %lf\t-\tk[%d, %d]: %1.16lf\t%1.16lfi\t\t%1.16lf\t%1.16lfi--\t%1.16lf %1.16lfi\n", sys_vars->force_scale_var, run_data->forcing_k[0][i], run_data->forcing_k[1][i], creal(dw_hat_dt[run_data->forcing_indx[i]]), cimag(dw_hat_dt[run_data->forcing_indx[i]]), creal(run_data->forcing[i]), cimag(run_data->forcing[i]), creal(run_data->forcing[i]) / creal(dw_hat_dt[run_data->forcing_indx[i]]), cimag(run_data->forcing[i]) / cimag(dw_hat_dt[run_data->forcing_indx[i]]));
		}
	}

	// PrintScalarFourier(dw_hat_dt, sys_vars->N, "curl");

	
 	// Apply dealiasing 
 	ApplyDealiasing(dw_hat_dt, 1, sys_vars->N);

	// Ensure conjugacy in the ky = 0 modes of the intial condition
    ForceConjugacy(dw_hat_dt, sys_vars->N, 1);
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
	ptrdiff_t local_Ny        = sys_vars->local_Ny;
	const long int Ny         = N[0];
	const long int Nx         = N[1];
	const long int Nx_Fourier = Nx / 2 + 1;
	double k_sqr;
	double kmax_sqr = pow((int) (Ny / 3.0), 2.0);
	#if defined(__DEALIAS_HOU_LI)
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter 
	// --------------------------------------------
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = array_dim * (tmp + j);

			// Get |k|^2
			k_sqr = (double) run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j];

			#if defined(__DEALIAS_23)
			if (sqrt(k_sqr) <= 0.0 || sqrt(k_sqr) > (int)(Ny / 3.0) - 0.5) {
				for (int l = 0; l < array_dim; ++l) {
					// Set dealised modes to 0
					array[indx + l] = 0.0 + 0.0 * I;	
				}
			}
			else if (!(strcmp(sys_vars->forcing, "KOLM")) && (run_data->k[1][j] == 0 && run_data->k[0][i] > 0)) {
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
			hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Ny / 2), 2.0) + pow(run_data->k[1][j] / (Nx / 2), 2.0))), 36.0));

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
	const long int Ny         = N[0];
	const long int Nx 		  = N[1];
	const long int Nx_Fourier = N[1] / 2 + 1; 

	// Initialize local variables 
	ptrdiff_t local_Ny = sys_vars->local_Ny;

    // ------------------------------------------------
    // Set Seed for RNG
    // ------------------------------------------------
    srand(123456789);

	if(!(strcmp(sys_vars->u0, "TG_VEL"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
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
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				if (run_data->k[0][i] == 0 || run_data->k[1][j] != 0) {
					w_hat[indx] = I * (run_data->k[0][i] * u_hat[SYS_DIM * (indx) + 1] - run_data->k[1][j] * u_hat[SYS_DIM * (indx) + 0]);
				}
				else {
					// Zero mode is always 0 + 0 * I
					w_hat[indx] = 0.0 + 0.0 * I;
				}
			}
		}
	}
	else if(!(strcmp(sys_vars->u0, "TG_VORT"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
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
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
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
	else if(!(strcmp(sys_vars->u0, "ZERO")) || !(strcmp(sys_vars->u0, "ZERO_KOLM"))) {
		// ------------------------------------------------
		// Zero - A Zero vorticity field initial condition
		// ------------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = (tmp + j);

				// Zero field
				w_hat[indx] = 0.0 + 0.0 * I; 
				
				// Initialize the specific modes for the Kolmogorov zero initial condition
				if (!(strcmp(sys_vars->u0, "ZERO_KOLM"))){
					if (run_data->k[1][j] == 1 && (run_data->k[0][i] == 1 || run_data->k[0][i] == -1)) {
						w_hat[indx] = -0.5 + 0.0 * I;
					}
					if (run_data->k[1][j] == 2 && (run_data->k[0][i] == 3 || run_data->k[0][i] == 4)) {
						w_hat[indx] = -4.0 + 1.0 * I;
					}
					if (run_data->k[1][j] == 3 && (run_data->k[0][i] == 4 || run_data->k[0][i] == 5)) {
						w_hat[indx] = -2.0 + 3.0 * I;
					}
					if (run_data->k[1][j] == 4 && (run_data->k[0][i] == 5 || run_data->k[0][i] == 6)) {
						w_hat[indx] = 8.0 - 6.0 * I;
					} 
				}
			}
		}
	}
	else if (!(strcmp(sys_vars->u0, "DECAY_TURB_BB")) || !(strcmp(sys_vars->u0, "DECAY_TURB_NB")) || !(strcmp(sys_vars->u0, "DECAY_TURB_EXP")) || !(strcmp(sys_vars->u0, "DECAY_TURB_EXP_II"))) {
		// --------------------------------------------------------
		// Decaying Turbulence ICs
		// --------------------------------------------------------
		// Initialize variables
		double sqrt_k;
		double inv_k_sqr;
		double u1;
		double spec_1d = 0.0;

		#if defined(DEBUG)
		double* rand_u = (double*)fftw_malloc(sizeof(double) * Ny * Nx_Fourier);
		#endif
		// ---------------------------------------------------------------
		// Initialize Vorticity with Specific Spectrum and Random Phases
		// ---------------------------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
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

					if (!(strcmp(sys_vars->u0, "DECAY_TURB_BB"))) {
						// Computet the Broad band initial spectrum
						spec_1d = sqrt_k / (1.0 + pow(sqrt_k, 4.0) / DT_K0);
					} 
					else if (!(strcmp(sys_vars->u0, "DECAY_TURB_NB"))) {
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
		int dims[2] = {Ny, Nx_Fourier};
		WriteTestDataReal(rand_u, "rand_u", 2, dims, local_Ny);
		WriteTestDataFourier(w_hat, "w_hat", Ny, Nx_Fourier, local_Ny);
		fftw_free(rand_u);
		#endif

		// ---------------------------------------
		// Compute the Initial Energy
		// ---------------------------------------
		double enrg = 0.0;
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;	

				if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
					// Wavenumber prefactor -> 1 / |k|^2
					inv_k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					if ((j == 0) || (j == Nx_Fourier - 1)) {
						enrg += inv_k_sqr * cabs(w_hat[indx] * conj(w_hat[indx]));
					}
					else {
						enrg += 2.0 * inv_k_sqr * cabs(w_hat[indx] * conj(w_hat[indx]));
					}
				}
			}
		}
		// Normalize 
		enrg *= (0.5 / pow(Ny * Nx, 2.0)) * 4.0 * pow(M_PI, 2.0);

		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &enrg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// -------------------------------------------
		// Normalize & Compute the Fourier Vorticity
		// -------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if (run_data->k[0][i] == 0 || run_data->k[1][j] != 0) {
					if (!(strcmp(sys_vars->u0, "DECAY_TURB_BB"))) {
						// Compute the Fouorier vorticity
						w_hat[indx] *= sqrt(DT_E0 / enrg);
					}
					else if (!(strcmp(sys_vars->u0, "DECAY_TURB_NB"))) {
						// Compute the Fouorier vorticity
						w_hat[indx] *= sqrt(DT2_E0 / enrg);
					}
					else if (!(strcmp(sys_vars->u0, "DECAY_TURB_EXP"))) {
						// Compute the Fouorier vorticity
						w_hat[indx] *= sqrt(DTEXP_E0 / enrg);
					}
				}
				else {
					// Zero mode is always 0 + 0 * I
					w_hat[indx] = 0.0 + 0.0 * I;
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
		fftw_complex* rand_u = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Ny * Nx_Fourier);
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

		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
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
		WriteTestDataFourier(rand_u, "rand_u", Ny, Nx_Fourier, local_Ny);
		WriteTestDataFourier(psi_hat, "psi_hat", Ny, Nx_Fourier, local_Ny);
		fftw_free(rand_u);
		#endif
		
		// ---------------------------------------------
		// Ensure Zero Mean Field
		// ---------------------------------------------
		// Variable to compute the mean
		double mean = 0.0;

		// Transform and Normalize while also gathering the mean
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), psi_hat, psi);
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;	

				// Normalize
				psi[indx] *= 1.0; /// (Ny * Nx);

				// Update the sum for the mean
				mean += psi[indx];
			}
		}
		

		// Reduce all local mean sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Enforce the zero mean of the stream function field
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;	

				// Normalize
				psi[indx] -=  (mean / (Ny * Nx));
			}
		}

		// Transform zero mean field back to Fourier space
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), psi, psi_hat);

		
		// ---------------------------------------------
		// Compute the Energy
		// ---------------------------------------------
		double enrg = 0.0;
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Wavenumber prefactor -> |k|^2
				k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				if ((j == 0) || (j == Nx_Fourier - 1)) {
					enrg += k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
				}
				else {
					enrg += 2.0 * k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
				}
			}
		}
		// Normalize the energy
		enrg *= 4.0 * pow(M_PI, 2.0) * (0.5 / pow(Ny * Nx, 2.0));
		
		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &enrg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// ---------------------------------------------
		// Normalize Initial Condition - Compute the Vorticity
		// ---------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Normalize the initial condition
				psi_hat[indx] *= sqrt(0.5 / enrg);

				// Get |k|^2
				k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Compute the vorticity 
				if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
					w_hat[indx] = -k_sqrd * psi_hat[indx];
				}
				else {
					// Zero mode is always 0 + 0 * I
					w_hat[indx] = 0.0 + 0.0 * I;
				}
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

		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
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
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
					// Wavenumber prefactor -> |k|^2
					k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					if ((j == 0) || (j == Nx_Fourier - 1)) {
						enrg += k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
					}
					else {
						enrg += 2.0 * k_sqrd * cabs(psi_hat[indx] * conj(psi_hat[indx]));
					}
				}
			}
		}
		// Normalize the energy
		enrg *= 4.0 * pow(M_PI, 2.0) * (0.5 / pow(Ny * Nx, 2.0));
		
		// Reduce all local energy sums and broadcast back to each process
		MPI_Allreduce(MPI_IN_PLACE, &enrg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// ---------------------------------------------
		// Normalize Initial Condition - Compute the Vorticity
		// ---------------------------------------------
		for (int i = 0; i < local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Normalize the initial condition
				psi_hat[indx] /=  sqrt(enrg);
				psi_hat[indx] *=  sqrt(0.5);

				// Get |k|^2
				k_sqrd = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Compute the vorticity 
				if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
					w_hat[indx] = k_sqrd * psi_hat[indx];
				}
				else {
					// Zero mode is always 0 + 0 * I
					w_hat[indx] = 0.0 + 0.0 * I;
				}
			}
		}

		// Free memory
		fftw_free(psi_hat);
	}
	else if (!(strcmp(sys_vars->u0, "EXTRM_ENS"))) {
	
		// ---------------------------------------
		// Define the Normalization Factor
		// ---------------------------------------
		double norm = 0.0;
		double k_sqr;
		for (int i = 0; i < local_Ny; ++i) {	
			for (int j = 0; j < Nx_Fourier; ++j) {
				// Get |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				if ((j == 0) || (j == Nx_Fourier - 1)) {
					// Compute the sum of appropiate modes to use as a normalization factor
					if ((k_sqr > 0) && (sqrt(k_sqr) > EXTRM_ENS_MIN_K)) {
						norm += pow(sqrt(k_sqr), -EXTRM_ENS_POW);
					}
					else if (sqrt(k_sqr) <= EXTRM_ENS_MIN_K){
						norm += pow(EXTRM_ENS_MIN_K, -EXTRM_ENS_POW) * exp((sqrt(k_sqr) - EXTRM_ENS_MIN_K)/3.5);
					}
				}
				else {
					// Compute the sum of appropiate modes to use as a normalization factor
					if ((k_sqr > 0) && (sqrt(k_sqr) > EXTRM_ENS_MIN_K)) {
						norm += 2.0 * pow(sqrt(k_sqr), -EXTRM_ENS_POW);
					}
					else if (sqrt(k_sqr) <= EXTRM_ENS_MIN_K){
						norm += 2.0 * pow(EXTRM_ENS_MIN_K, -EXTRM_ENS_POW) * exp((sqrt(k_sqr) - EXTRM_ENS_MIN_K)/3.5);
					}	
				}
			}
		}
		// Synchronize normalization factor amongst the processes
		MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		norm = sqrt(EXTRM_ENS_MIN_K / norm);

		// ---------------------------------------
		// Initialize Fourier Space Vorticity
		// ---------------------------------------
		double u1;
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Get |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Initialize the Fourier modes
				if ((k_sqr > 0) && (sqrt(k_sqr) > EXTRM_ENS_MIN_K)) {
					// Get random uniform number of the phases
					u1 = (double) rand() / (double) RAND_MAX;
					w_hat[indx] = pow(sqrt(k_sqr), -EXTRM_ENS_POW / 2.0) * norm * cexp(I * 2.0 * M_PI * u1);
				}
				else if (sqrt(k_sqr) <= EXTRM_ENS_MIN_K){
					// Get random uniform number of the phases
					u1 = (double) rand() / (double) RAND_MAX;
					w_hat[indx] = pow(EXTRM_ENS_MIN_K, -EXTRM_ENS_POW / 2.0) * exp((sqrt(k_sqr) - EXTRM_ENS_MIN_K)/3.5) * norm * cexp(I * 2.0 * M_PI * u1);
				}
				else {
					w_hat[indx] = 0.0 + 0.0 * I;
				}
			}
		}
	}
	else if (!(strcmp(sys_vars->u0, "RING"))) {
	
		// ---------------------------------------
		// Define the Normalization Factor
		// ---------------------------------------
		double norm = 0.0;
		double abs_k;
		double u1;
		for (int i = 0; i < local_Ny; ++i) {	
			for (int j = 0; j < Nx_Fourier; ++j) {
				// Get the absolute wavevector value
				abs_k = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				u1 = (double) rand() / (double) RAND_MAX;

				// Compute the sum of appropiate modes to use as a normalization factor
				if ((abs_k >= RING_MIN_K && abs_k <= RING_MAX_K) && (run_data->k[1][j] != 0 || run_data->k[0][i] > 0)) {
					norm += exp(- pow(abs_k - ((RING_MIN_K + RING_MAX_K) /2.0), 2.0));
				}
			}
		}
		// Synchronize normalization factor amongst the processes
		MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// ---------------------------------------
		// Initialize Fourier Space Vorticity
		// ---------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Get the absolute wavevector value
				abs_k = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				// Initialize the Fourier modes
				if ((abs_k >= RING_MIN_K && abs_k <= RING_MAX_K) && (run_data->k[1][j] != 0 || run_data->k[0][i] > 0)) {
					// Get random uniform number of the phases
					u1 = (double) rand() / (double) RAND_MAX;
					w_hat[indx] = sqrt(72.0 * exp(-pow(abs_k - ((RING_MIN_K + RING_MAX_K) /2.0), 2.0)) / (M_PI * abs_k * norm)) * cexp(2.0 * M_PI * I * u1);
				}
				else {
					w_hat[indx] = 0.0 + 0.0 * I;
				}
			}
		}
	}
	else if (!(strcmp(sys_vars->u0, "UNIF"))) {
	
		// ---------------------------------------
		// Define the Normalization Factor
		// ---------------------------------------
		double norm = 0.0;
		double k_abs;
		double u1;
		for (int i = 0; i < local_Ny; ++i) {	
			for (int j = 0; j < Nx_Fourier; ++j) {
				// Get |k|^2
				k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				// Compute the sum of appropiate modes to use as a normalization factor
				if ((k_abs > UNIF_MIN_K && k_abs < UNIF_MAX_K)) {
					if (j == 0 || j == Nx_Fourier - 1) {
						norm += 1.0;
					}
					else {
						norm += 2.0;	
					}
				}
			}
		}
		// Synchronize normalization factor amongst the processes
		MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		norm = sqrt(1.0 / norm);

		// ---------------------------------------
		// Initialize Fourier Space Vorticity
		// ---------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Get the absolute wavevector value
				k_abs = sqrt((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

				// Initialize the Fourier modes
				if ((k_abs > UNIF_MIN_K && k_abs < UNIF_MAX_K)) {
					// Get random uniform number of the phases
					u1 = (double) rand() / (double) RAND_MAX;
					w_hat[indx] = norm * cexp(2.0 * M_PI * I * u1); 
				}
				else {
					w_hat[indx] = 0.0 + 0.0 * I;
				}
			}
		}
	}
	else if (!(strcmp(sys_vars->u0, "MAX_PALIN"))) {
	
		// ---------------------------------------
		// Allocate Temporary Memory
		// ---------------------------------------
		// Allocate memory for the Real Space stream function
		double* tmp_psi       = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
		double* tmp_psi_local = (double* )fftw_malloc(sizeof(double) * sys_vars->local_Ny * Nx);
		double* tmp_psi_full  = (double* )fftw_malloc(sizeof(double) * Ny * Nx);
		fftw_complex* tmp_psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);


		// ---------------------------------------
		// Read in Initial Condition From File
		// ---------------------------------------
		if (!sys_vars->rank) {
			// Get input file path
			char tmp_path[512];
			char tmp_filename[512];
			strcpy(tmp_path, file_info->input_dir);
			sprintf(tmp_filename, "MaxPalinstrophy/maxdpdt_P10000_N%ld_IG0_psi_f.h5", sys_vars->N[0]);
			strcat(tmp_path, tmp_filename); 
			strcpy(file_info->input_file_name, tmp_path); 

			// Open input file containing initial condition
			file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    		if (file_info->input_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"] for the initial condition ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name, sys_vars->u0);
				exit(1);
			}

			// Read in stream function initial condition
			if ((H5LTread_dataset_double(file_info->input_file_handle, "/psi", tmp_psi_full)) < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read dataset ["CYAN"%s"RESET"] for the initial condition ["CYAN"%s"RESET"]\n-->> Exiting...\n", "/psi", sys_vars->u0);
				exit(1);	
			}
		}

		// ---------------------------------------
		// Distribute Initial Condition
		// ---------------------------------------
		// Scatter the initial stream function data to the local processes
		MPI_Scatter(tmp_psi_full, sys_vars->local_Ny * Nx, MPI_DOUBLE, tmp_psi_local, sys_vars->local_Ny * Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Write the local data to the appropriately padded arrays
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Write the local data to the padded arrays
				tmp_psi[indx] = tmp_psi_local[i * Nx + j];
			}
		}

		// --------------------------------------------
		// Transform to Fourier Space Stream Function
		// --------------------------------------------
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), tmp_psi, tmp_psi_hat);

		// ---------------------------------------
		// Initialize Fourier Space Vorticity
		// ---------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Compute the real space vorticity from stream function
				run_data->w_hat[indx] = - (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) * tmp_psi_hat[indx];
			}
		}
		
		// Free temporary memory
		fftw_free(tmp_psi);
		fftw_free(tmp_psi_hat);
		fftw_free(tmp_psi_full);
		fftw_free(tmp_psi_local);
	}
	else if (!(strcmp(sys_vars->u0, "GAUSS_BLOB"))) {
	
		// ---------------------------------------
		// Define the Gaussian Blob
		// ---------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
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
		// Initialize variables
		double enrg_k, enst_k;
		double sqrt_k;
		double enst_norm;
		double r1;

		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Get the |k|^2
				sqrt_k = sqrt(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

				// Get the energy for this mode
				if (sqrt_k > UNIF_MIN_K && sqrt_k < UNIF_MAX_K) {
					enrg_k = 1.0 / sqrt_k;
					
					// Get the enstrophy for this mode
					enst_k = pow(sqrt_k, 2.0) * enrg_k;

					// Get random uniform number
					r1 = (double)rand() / (double) RAND_MAX;

					// Fill vorticity
					w_hat[indx] = sqrt(enst_k / enrg_k) * cexp(r1 * 2.0 * M_PI * I);
				}
				else {
					// Set Vorticity to 0
					w_hat[indx] = 0.0 + 0.0 * I;
				}

				// Compute the enstrophy norm
				enst_norm += pow(cabs(w_hat[indx]), 2.0);
			}
		}

		// ---------------------------------------
		// Sync Norm Factor
		// ---------------------------------------
		// Synchronize normalization factor amongst the processes
		MPI_Allreduce(MPI_IN_PLACE, &enst_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		enst_norm *= 2.0;

		// ---------------------------------------
		// Normalize
		// ---------------------------------------
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				if ((run_data->k[0][i] != 0) || (run_data->k[1][j] != 0)){
					w_hat[indx] *= sqrt(RAND_ENST0 / enst_norm) ;
				}
				else {
					// Zero mode is always 0 + 0 * I
					w_hat[indx] = 0.0 + 0.0 * I;
				}			
			}
		}		
	}
	else if (!(strcmp(sys_vars->u0, "TESTING"))) {
		// ---------------------------------------
		// Powerlaw Amplitude & Fixed Phase
		// ---------------------------------------
		// Initialize temp variables
		double inv_k_sqr;

		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)){
					// Fill zero modes
					w_hat[indx] = 0.0 + 0.0 * I;
				}
				else if (j == 0 && run_data->k[0][i] < 0) {
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
		for (int i = 0; i < local_Ny; ++i) {	
			tmp = i * (Nx_Fourier);
			for (int j = 0; j < Nx_Fourier; ++j) {
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
    ForceConjugacy(w_hat, N, 1);
    ForceConjugacy(u_hat, N, 2);
 	

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
	const long int Ny = N[0];
	const long int Nx = N[1];
	const long int Nx_Fourier = N[1] / 2 + 1;

	// Initialize local variables 
	ptrdiff_t local_Ny       = sys_vars->local_Ny;
	ptrdiff_t local_Ny_start = sys_vars->local_Ny_start;
	
	// Set the spatial increments
	sys_vars->dx = 2.0 * M_PI / (double )Ny;
	sys_vars->dy = 2.0 * M_PI / (double )Nx;

	// -------------------------------
	// Fill the first dirction 
	// -------------------------------
	int j = 0;
	for (int i = 0; i < Ny; ++i) {
		if((i >= local_Ny_start) && ( i < local_Ny_start + local_Ny)) { // Ensure each process only writes to its local array slice
			x[0][j] = (double) i * 2.0 * M_PI / (double) Ny;
			j++;
		}
	}
	j = 0;
	for (int i = 0; i < local_Ny; ++i) {
		if (local_Ny_start + i <= Ny / 2) {   // Set the first half of array to the positive k
			k[0][j] = local_Ny_start + i;
			j++;
		}
		else if (local_Ny_start + i > Ny / 2) { // Set the second half of array to the negative k
			k[0][j] = local_Ny_start + i - Ny;
			j++;
		}
	}

	// -------------------------------
	// Fill the second direction 
	// -------------------------------
	for (int i = 0; i < Nx; ++i) {
		if (i < Nx_Fourier) {
			k[1][i] = i;
		}
		x[1][i] = (double) i * 2.0 * M_PI / (double) Nx;
	}
}
/**
 * Function to force conjugacy of the initial condition
 * @param w_hat The Fourier space worticity field
 * @param N     The array containing the size of the system in each dimension
 * @param dim   Dimension of the array -> 1: for scalar, 2: for vector.
 */
void ForceConjugacy(fftw_complex* array, const long int* N, const int dim) {

	// Initialize variables
	int tmp;
	int local_Ny       = (int) sys_vars->local_Ny;
	int local_Ny_start = (int) sys_vars->local_Ny_start;
	const long int Ny         = N[0];
	const long int Nx_Fourier = N[1] / 2 + 1;

	// Allocate tmp memory to hold the data to be conjugated
	fftw_complex* conj_data = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Ny * dim);

	// Loop through local process and store data in appropriate location in conj_data
	for (int i = 0; i < local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int d = 0; d < dim; ++d) {
			conj_data[dim * (local_Ny_start + i) + d] = array[dim * tmp + d];
			// printf("c[%d,%d]: %lf %lf i\n", run_data->k[0][i], d, creal(conj_data[dim * (local_Ny_start + i) + d]), cimag(conj_data[dim * (local_Ny_start + i) + d]));
		}
	}

	// Gather the data on all process
	MPI_Allgather(MPI_IN_PLACE, (int)(local_Ny * dim), MPI_C_DOUBLE_COMPLEX, conj_data, (int)(local_Ny * dim), MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);

	// Now ensure the 
	for (int i = 0; i < local_Ny; ++i) {
		if (run_data->k[0][i] > 0) {
			tmp = i * Nx_Fourier;
			for (int d = 0; d < dim; ++d) {
				// Fill the conjugate modes with the conjugate of the postive k modes
				array[dim * tmp + d] = conj(conj_data[dim * (Ny - run_data->k[0][i]) + d]);
			}
		}
	}

	// Free memory
	fftw_free(conj_data);
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
	if (sys_vars->ADAPT_STEP_FLAG == ADAPTIVE_STEP) {
		GetTimestep(&(sys_vars->dt));
	}

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
	if (sys_vars->TRANS_ITERS_FLAG == TRANSIENT_ITERS) {
		// Get the transient iterations
		(* trans_steps)       = (long int)(sys_vars->TRANS_ITERS_FRAC * sys_vars->num_t_steps);
		sys_vars->trans_iters = (* trans_steps);

		// Get the number of steps to perform before printing to file -> allowing for a transient fraction of these to be ignored
		sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? (sys_vars->num_t_steps - sys_vars->trans_iters) / sys_vars->SAVE_EVERY : sys_vars->num_t_steps - sys_vars->trans_iters;	 
		if (!(sys_vars->rank)){
			printf("Total Iters: %ld\t Saving Iters: %ld\t Transient Steps: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps, sys_vars->trans_iters);
		}
	}
	else {
		// Get the transient iterations
		(* trans_steps)       = 0;
		sys_vars->trans_iters = (* trans_steps);

		// Get the number of steps to perform before printing to file
		sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? sys_vars->num_t_steps / sys_vars->SAVE_EVERY + 1 : sys_vars->num_t_steps + 1; // plus one to include initial condition
		if (!(sys_vars->rank)){
			printf("Total Iters: %ld\t Saving Iters: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps);
		}
	}

	// Variable to control how ofter to print to screen -> set it to half the saving to file steps
	sys_vars->print_every = (sys_vars->num_t_steps >= 10 ) ? (int)sys_vars->SAVE_EVERY : 1;
}
/**
 * Function used to compute of either the velocity or vorticity
 * @return  Returns the computed maximum
 */
double GetMaxData(char* dtype) {

	// Initialize variables
	const long int Ny         = sys_vars->N[0];
	const long int Nx 		  = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;
	int tmp;
	int indx;
	fftw_complex I_over_k_sqr;

	// -------------------------------
	// Compute the Data 
	// -------------------------------
	if (strcmp(dtype, "VEL") == 0) {
		// Compute the velocity in Fourier space
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
					// Compute the prefactor				
					I_over_k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// compute the velocity in Fourier space
					run_data->u_hat_tmp[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * I_over_k_sqr * run_data->w_hat[indx];
					run_data->u_hat_tmp[SYS_DIM * indx + 1] = -1.0 * ((double) run_data->k[0][i]) * I_over_k_sqr * run_data->w_hat[indx];
				}
				else {
					run_data->u_hat_tmp[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat_tmp[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
				}
			}
		}

		// Transform back to Fourier space
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat_tmp, run_data->u);
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->u[SYS_DIM * indx + 0] *= 1.0; /// (double )(Ny * Nx);
				run_data->u[SYS_DIM * indx + 1] *= 1.0; /// (double )(Ny * Nx);
			}
		}
	}
	else if (strcmp(dtype, "VORT") == 0) {
		// Write w_hat to temporary array for transform back to real space
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
				indx = tmp + j;

				run_data->w_hat_tmp[indx] = run_data->w_hat[indx];
			}
		}
		// Perform transform back to real space for the vorticity
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->w_hat_tmp, run_data->w);
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->w[indx] *= 1.0; // (double )(Ny * Nx);
			}
		}
	}

	// -------------------------------
	// Find the maximum
	// -------------------------------
	// Define the maximum
	double wmax = 0.0;

	// Loop over array to find the maximum
	for (int i = 0; i < sys_vars->local_Ny; ++i) {
		// Find the max of the Real Space vorticity
		if (strcmp(dtype, "VORT_FOUR") == 0) {
			tmp = i * Nx_Fourier;
			for (int j = 0; j < Nx_Fourier; ++j) {
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
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
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
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
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
			printf("Iter: %d\tt: %1.6lf/%1.3lf\tdt: %1.6g \tMax Vort: %1.5g \tKE: %1.5g\tENS: %1.5g\tPAL: %1.5g\tL2: %1.5g\tLinf: %1.5g\n", iters, t, T, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx], norms[0], norms[1]);
		}
	}
	else {
		// Print Update to screen
		if( !(sys_vars->rank) ) {	
			printf("Iter: %d\tt: %1.6lf/%1.3lf\tdt: %1.6g \tMax Vort: %1.5g \tTKE: %1.8lf\tENS: %1.8lf\tPAL: %1.5g\tE_Diss: %1.5g\tEns_Diss: %1.5g\n", iters, t, T, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx], run_data->enrg_diss[save_data_indx], run_data->enst_diss[save_data_indx]);
		}
	}
	#else
	// Get max vorticity
	max_vort = GetMaxData("VORT");

	// Print to screen
	if( !(sys_vars->rank) ) {	
		printf("Iter: %d/%ld\tt: %1.6lf/%1.3lf\tdt: %1.6g \tMax Vort: %1.5g \tKE: %1.5g\tENS: %1.5g\t PAL: %1.5g\t E_Diss: %1.5g\tEns_Diss: %1.5g\n", iters, sys_vars->num_t_steps, t, T, dt, max_vort, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx], run_data->enrg_diss[save_data_indx], run_data->enst_diss[save_data_indx]);
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
	if(sys_vars->CFL_COND_FLAG == CFL_STEP) {
		// Initialize variables
		int tmp;
		int indx;
		fftw_complex I_over_k_sqr;
		const long int Ny = sys_vars->N[0];
		const long int Nx = sys_vars->N[1];
		
		// -------------------------------
		// Compute the Velocity
		// -------------------------------
		// Compute the velocity in Fourier space
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
			tmp = i * sys_vars->N[1] / 2 + 1;
			for (int j = 0; j < sys_vars->N[1] / 2 + 1; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] != 0) || (run_data->k[1][j]  != 0)) {
					// Compute the prefactor				
					I_over_k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// compute the velocity in Fourier space
					run_data->u_hat_tmp[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * I_over_k_sqr * run_data->w_hat[indx];
					run_data->u_hat_tmp[SYS_DIM * indx + 1] = -1.0 * ((double) run_data->k[0][i]) * I_over_k_sqr * run_data->w_hat[indx];
				}
				else {
					// Get the zero mode
					run_data->u_hat_tmp[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat_tmp[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
				}
			}
		}

		// Transform back to Fourier space
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat_tmp	, run_data->u);
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
			tmp = i * (Nx + 2);
			for (int j = 0; j < Nx; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->u[SYS_DIM * indx + 0] *= 1.0; /// (double )(Ny * Nx);
				run_data->u[SYS_DIM * indx + 1] *= 1.0; /// (double )(Ny * Nx);
			}
		}

		// -------------------------------
		// Find the Times Scales 
		// -------------------------------
		// Find the convective scales
		double scales_convect = 0.0;
		for (int i = 0; i < sys_vars->local_Ny; ++i) {
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
	}
	else {
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
	}

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
	const long int Ny         = N[0];
	const long int Nx         = N[1];
	const long int Nx_Fourier = N[1] / 2.0 + 1;
	double norm_const = 1.0 / (double)(Ny * Nx);
	double linf_norm  = 0.0;
	double l2_norm    = 0.0;
	double tg_exact;
	double abs_err;

	// --------------------------------
	// Get the Real Space Vorticity
	// --------------------------------
	// Write w_hat to temporary array for transform back to real space
	for (int i = 0; i < sys_vars->local_Ny; ++i) {
		tmp = i * Nx_Fourier;
		for (int j = 0; j < Nx_Fourier; ++j) {
			indx = tmp + j;

			run_data->w_hat_tmp[indx] = run_data->w_hat[indx];
		}
	}
	// Transform back to Fourier space -> Don't normalize now - normalize in loop below
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->w_hat_tmp, run_data->w);
	
	// --------------------------------
	// Compute The Error
	// --------------------------------
	for (int i = 0; i < sys_vars->local_Ny; ++i) {
		tmp = i * (Nx + 2);
		for (int j = 0; j < Nx; ++j) {
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
	const long int Nx = N[1];
	
	// --------------------------------
	// Compute The Taylor Green Soln
	// --------------------------------
	for (int i = 0; i < sys_vars->local_Ny; ++i) {
		tmp = i * (Nx + 2);
		for (int j = 0; j < Nx; ++j) {
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
 * Wrapper function used to allocate memory all the nessecary local and global system and integration arrays
 * @param NBatch  Array holding the dimensions of the Fourier space arrays
 * @param Int_data Pointer to struct containing the integration arrays
 */
void AllocateMemory(const long int* NBatch, Int_data_struct* Int_data) {

	// Initialize variables
	const long int Ny         = sys_vars->N[0];
	const long int Nx         = sys_vars->N[1];
	const long int Nx_Fourier = sys_vars->N[1] / 2 + 1;	

	// -------------------------------
	// Get Local Array Sizes - FFTW 
	// -------------------------------
	//  Find the size of memory for the FFTW transforms - use these to allocate appropriate memory
	sys_vars->alloc_local       = fftw_mpi_local_size_2d(Ny, Nx_Fourier, MPI_COMM_WORLD, &(sys_vars->local_Ny), &(sys_vars->local_Ny_start));
	sys_vars->alloc_local_batch = fftw_mpi_local_size_many((int)SYS_DIM, NBatch, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &(sys_vars->local_Ny), &(sys_vars->local_Ny_start));
	if (sys_vars->local_Ny == 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- FFTW was unable to allocate local memory for each process -->> Code will run but will be slow\n");
	}
	
	// -------------------------------
	// Allocate Space Variables 
	// -------------------------------
	// Allocate the wavenumber arrays
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->local_Ny);  // ky
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * Nx_Fourier);     		// kx
	if (run_data->k[0] == NULL || run_data->k[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "wavenumber list");
		exit(1);
	}

	// Allocate the collocation points
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->local_Ny);  // y direction 
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * Nx);     			  // x direction
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
	// This array is for performing Real to Complex transforms of w_hat because the input array gets overwritten
	run_data->w_hat_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->w_hat_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Temporary Fourier Space Vorticity");
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
	run_data->u_hat_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->u_hat_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Temporary Fourier Space Velocities");
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
	run_data->tg_soln = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
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
	Int_data->nabla_psi = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (Int_data->nabla_psi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "nabla_psi");
		exit(1);
	}
	Int_data->nabla_w   = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (Int_data->nabla_w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "nabla_w");
		exit(1);
	}
	Int_data->nonlin   = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
	if (Int_data->nonlin == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "nonlin");
		exit(1);
	}
	Int_data->RK1       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK1 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK1");
		exit(1);
	}
	Int_data->RK2       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK2 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK2");
		exit(1);
	}
	Int_data->RK3       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK3 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK3");
		exit(1);
	}
	Int_data->RK4       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK4 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK4");
		exit(1);
	}
	Int_data->RK_tmp    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (Int_data->RK_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK_tmp");
		exit(1);
	}
	#if defined(__RK5) || defined(__DPRK5)
	Int_data->RK5       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK5 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK5");
		exit(1);
	}
	Int_data->RK6       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK6 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK6");
		exit(1);
	}
	#endif
	#if defined(__DPRK5)
	Int_data->RK7       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->RK7 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK7");
		exit(1);
	}
	Int_data->w_hat_last = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (Int_data->w_hat_last == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "w_hat_last");
		exit(1);
	}
	#endif
	#if defined(__AB4)
	// Initialize the number of derivative steps
	Int_data->AB_pre_steps = 3;

	// Allocate Adams Bashforth arrays
	Int_data->AB_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (Int_data->AB_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "AB_tmp");
		exit(1);
	}
	for (int i = 0; i < 3; ++i) {
		Int_data->AB_tmp_nonlin[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
		if (Int_data->AB_tmp_nonlin[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "AB_tmp_nonlin");
			exit(1);
		}
	}
	#endif

	// -------------------------------
	// Initialize All Data 
	// -------------------------------
	int tmp_real, tmp_four;
	int indx_real, indx_four;
	for (int i = 0; i < sys_vars->local_Ny; ++i) {
		tmp_real = i * (Nx + 2);
		tmp_four = i * Nx_Fourier;
		
		for (int j = 0; j < Nx; ++j){
			indx_real = tmp_real + j;
			indx_four = tmp_four + j;
			
			Int_data->nonlin[indx_real]                  = 0.0;
			run_data->u[SYS_DIM * indx_real + 0]         = 0.0;
			run_data->u[SYS_DIM * indx_real + 1]         = 0.0;
			Int_data->nabla_w[SYS_DIM * indx_real + 0]   = 0.0;
			Int_data->nabla_w[SYS_DIM * indx_real + 1]   = 0.0;
			Int_data->nabla_psi[SYS_DIM * indx_real + 0] = 0.0;
			Int_data->nabla_psi[SYS_DIM * indx_real + 1] = 0.0;
			#if defined(TESTING)
			run_data->tg_soln[indx_real]                = 0.0;
			#endif
			run_data->w[indx_real]                      = 0.0;
			if (j < Nx_Fourier) {
				#if defined(PHASE_ONLY)
				run_data->a_k[indx_four] 				= 0.0;
				run_data->phi_k[indx_four]			    = 0.0;
				run_data->tmp_a_k[indx_four] 			= 0.0;
				#endif
				run_data->w_hat[indx_four]               	  = 0.0 + 0.0 * I;
				run_data->w_hat_tmp[indx_four]             	  = 0.0 + 0.0 * I;
				Int_data->RK_tmp[indx_four]    			 	  = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 0] 	  = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 1] 	  = 0.0 + 0.0 * I;
				run_data->u_hat_tmp[SYS_DIM * indx_four + 0]  = 0.0 + 0.0 * I;
				run_data->u_hat_tmp[SYS_DIM * indx_four + 1]  = 0.0 + 0.0 * I;
				Int_data->RK1[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				Int_data->RK1[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				Int_data->RK2[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				Int_data->RK2[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				Int_data->RK3[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				Int_data->RK3[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				Int_data->RK4[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				Int_data->RK4[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				#if defined(__NONLIN)
				run_data->nonlinterm[SYS_DIM * indx_four + 0] = 0.0 + 0.0 * I;
				run_data->nonlinterm[SYS_DIM * indx_four + 1] = 0.0 + 0.0 * I;
				#endif
				#if defined(__RK5)
				Int_data->RK5[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				Int_data->RK5[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				Int_data->RK6[SYS_DIM * indx_four + 0]    	  = 0.0 + 0.0 * I;
				Int_data->RK6[SYS_DIM * indx_four + 1]    	  = 0.0 + 0.0 * I;
				#endif
				#if defined(__DPRK5)
				Int_data->RK7[SYS_DIM * indx_four + 0]    	 = 0.0 + 0.0 * I;
				Int_data->RK7[SYS_DIM * indx_four + 1]    	 = 0.0 + 0.0 * I;
				Int_data->w_hat_last[indx_four]			 	 = 0.0 + 0.0 * I;
				#endif
				#if defined(__DPRK5)
				Int_data->AB_tmp[SYS_DIM * indx_four + 0] = 0.0 + 0.0 * I;
				Int_data->AB_tmp[SYS_DIM * indx_four + 1] = 0.0 + 0.0 * I;
				for (int a = 0; a < 3; ++a) {
					Int_data->AB_tmp_nonlin[a][SYS_DIM * indx_four + 0] = 0.0 + 0.0 * I;
					Int_data->AB_tmp_nonlin[a][SYS_DIM * indx_four + 1] = 0.0 + 0.0 * I;
				}
				#endif
			}
			if (i == 0) {
				if (j < Nx_Fourier) {
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
	const long int Ny = N[0];
	const long int Nx = N[1];

	// -----------------------------------
	// Initialize Plans for Vorticity 
	// -----------------------------------
	// Set up FFTW plans for normal transform - vorticity field
	sys_vars->fftw_2d_dft_r2c = fftw_mpi_plan_dft_r2c_2d(Ny, Nx, run_data->w, run_data->w_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	sys_vars->fftw_2d_dft_c2r = fftw_mpi_plan_dft_c2r_2d(Ny, Nx, run_data->w_hat, run_data->w, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
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
 * Wrapper function that frees any memory dynamcially allocated in the programme
 * @param Int_data Pointer to a struct contaiing the integraiont arrays
 */
void FreeMemory(Int_data_struct* Int_data) {

	// ------------------------
	// Free memory 
	// ------------------------
	// Free space variables
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
		if (sys_vars->local_forcing_proc) {
			fftw_free(run_data->forcing_k[i]);
		}
	}

	// Free system variables
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	fftw_free(run_data->u_hat_tmp);
	fftw_free(run_data->w);
	fftw_free(run_data->w_hat);
	fftw_free(run_data->w_hat_tmp);
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
	fftw_free(run_data->mean_flow_x);
	fftw_free(run_data->mean_flow_y);
	#endif
	#if defined(__PHASE_SYNC)
	fftw_free(run_data->phase_order_k);
	fftw_free(run_data->normed_phase_order_k);
	#endif
	#if defined(__ENST_FLUX)
	fftw_free(run_data->enst_flux_sbst);
	fftw_free(run_data->enst_diss_sbst);
	fftw_free(run_data->d_enst_dt_sbst);
	#endif
	#if defined(__ENRG_FLUX)
	fftw_free(run_data->enrg_flux_sbst);
	fftw_free(run_data->enrg_diss_sbst);
	fftw_free(run_data->d_enrg_dt_sbst);
	#endif
	#if defined(__ENRG_SPECT)
	fftw_free(run_data->enrg_spect);
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	fftw_free(run_data->enrg_flux_spect);
	fftw_free(run_data->enrg_diss_spect);
	fftw_free(run_data->d_enrg_dt_spect);
	#endif
	#if defined(__ENST_SPECT)
	fftw_free(run_data->enst_spect);
	#endif
	#if defined(__ENST_FLUX_SPECT)
	fftw_free(run_data->enst_flux_spect);
	fftw_free(run_data->enst_diss_spect);
	fftw_free(run_data->d_enst_dt_spect);
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
	fftw_free(Int_data->RK1);
	fftw_free(Int_data->RK2);
	fftw_free(Int_data->RK3);
	fftw_free(Int_data->RK4);
	#if defined(__RK5) || defined(__DPRK5)
	fftw_free(Int_data->RK5);
	fftw_free(Int_data->RK6);
	#endif 
	#if defined(__DPRK5)
	fftw_free(Int_data->RK7);
	fftw_free(Int_data->w_hat_last);
	#endif
	#if defined(__AB4)
	fftw_free(Int_data->AB_tmp);
	for (int i = 0; i < 3; ++i) {
		fftw_free(Int_data->AB_tmp_nonlin[i]);
	}
	#endif
	fftw_free(Int_data->RK_tmp);
	fftw_free(Int_data->nonlin);
	fftw_free(Int_data->nabla_w);
	fftw_free(Int_data->nabla_psi);

	#if defined(__STATS)
	FreeStatsObjects();
	#endif

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