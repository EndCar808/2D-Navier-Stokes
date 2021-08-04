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
#ifdef __RK4
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
	sys_vars->u0   = "TAYLOR_GREEN";
	sys_vars->N[0] = 4;
	sys_vars->N[1] = 4;
	herr_t status;
	const long int N[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1]};
	const long int NBatch[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1] / 2 + 1};
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Initialize the Runge-Kutta struct
	struct RK_data_struct* RK_data;	// Initialize pointer to a RK_data_struct
	struct RK_data_struct RK_data_tmp; // Initialize a RK_data_struct
	RK_data = &RK_data_tmp;		// Point the ptr to this new RK_data_struct

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
	PrintVelocityReal(N);
	PrintVorticityReal(N);
	PrintVorticityFourier(N);
	NonlinearRHSBatch(run_data->w_hat, RK_data->RK1, RK_data->nabla_psi, RK_data->nabla_w);
	PrintScalarFourier(RK_data->RK1, N, "RHS");
	// -------------------------------
	// Integration Variables
	// -------------------------------
	// Set the spatial increments
	sys_vars->dx = 2.0 * M_PI / (double )Nx;
	sys_vars->dy = 2.0 * M_PI / (double )Ny;

	// Get the timestep using a CFL like condition
	double umax          = GetMaxData("VEL");
	sys_vars->w_max_init = GetMaxData("VORT");
	sys_vars->dt         = (sys_vars->dx) / umax;

	// Compute integration time variables
	sys_vars->t0 = 0.0;
	sys_vars->T  = 0.008;
	double t0    = sys_vars->t0;
	double t     = t0;
	double dt    = sys_vars->dt;
	double T     = sys_vars->T;

	// Number of iterations
	sys_vars->num_t_steps     = (T - t0) / dt;
	sys_vars->num_print_steps = sys_vars->num_t_steps / SAVE_EVERY + 1; // plus one to include initial condition
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps);
	}

	// Variable to control how ofter to print to screen -> 10% of num time steps
	int print_update = (sys_vars->num_t_steps >= 10 ) ? (int)((double)sys_vars->num_t_steps * 0.1) : 1;


	// -------------------------------
	// Create & Open Output File
	// -------------------------------
	// Create and open the output file - also write initial conditions to file
	CreateOutputFileWriteICs(N, dt);

	// Inialize system measurables
	InitializeSystemMeasurables();


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
	// 	if (iters % SAVE_EVERY == 0) {
	// 		// Record System Measurables
	// 		RecordSystemMeasures(t, save_data_indx);

	// 		// Write the appropriate datasets to file
	// 		WriteDataToFile(t, dt, save_data_indx);
			
	// 		// Update saving data index
	// 		save_data_indx++;
	// 	}

	// 	// -------------------------------
	// 	// Print Update To Screen
	// 	// -------------------------------
	// 	#ifdef __PRINT_SCREEN
	// 	if( !(sys_vars->rank) ) {
	// 		if (iters % print_update == 0) {
	// 			printf("Iter: %d/%ld\tt: %1.6lf/%1.3lf\tdt: %g\tE: %g\tZ: %g\tP: %g\n", iters, sys_vars->num_t_steps, t, T, dt, run_data->tot_energy[save_data_indx], run_data->tot_enstr[save_data_indx], run_data->tot_palin[save_data_indx]);
	// 		}
	// 	}
	// 	#endif

	// 	// -------------------------------
	// 	// Update & System Check
	// 	// -------------------------------
	// 	// Update timestep & iteration counter
	// 	#if defined(__ADAPTIVE_STEP) 
	// 	GetTimestep(&t);
	// 	#elif !defined(__DPRK5) && !defined(__ADAPTIVE_STEP)
	// 	t = iters * dt;
	// 	#endif
	// 	iters++;

	// 	// Check System: Determine if system has blown up or integration limits reached
	// 	SystemCheck(t, iters);
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
	// ----------------------- Stage 6
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
			D_fac = dt * (NU * pow(k_sqr, VIS_POW) + EKMN_ALPHA * pow(k_sqr, EKMN_POW)); 

			// Complete the update step
			run_data->w_hat[indx] = run_data->w_hat[indx] * ((2 - D_fac) / (2 + D_fac)) + (2 * dt / (2 + D_fac)) * ((RK5_B1 * RK_data->RK1[indx]) + (RK5_B3 * RK_data->RK3[indx]) + (RK5_B4 * RK_data->RK4[indx]) + (RK5_B5 * RK_data->RK5[indx]) + (RK5_B6 * RK_data->RK6[indx]));
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
			D_fac = dt * (NU * pow(k_sqr, VIS_POW) + EKMN_ALPHA * pow(k_sqr, EKMN_POW)); 

			// Update temporary input for nonlinear term
			run_data->w_hat[indx] = run_data->w_hat[indx] * ((2 - D_fac) / (2 + D_fac)) + (2 * dt / (2 + D_fac)) * ((RK4_B1 * RK_data->RK1[indx]) + (RK4_B2 * RK_data->RK2[indx]) + (RK4_B3 * RK_data->RK3[indx]) + (RK4_B4 * RK_data->RK4[indx]));
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
	ptrdiff_t local_Nx        = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	const double norm_fac     = 1.0 / (Nx * Ny);
	fftw_complex k_sqr;

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

			// denominator
			k_sqr = I / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + (double)1E-50);		

			// Fill fill fourier velocities array
			dw_hat_dt[SYS_DIM * (indx) + 0] = k_sqr * ((double) run_data->k[1][j]) * w_hat[indx];
			dw_hat_dt[SYS_DIM * (indx) + 1] = -k_sqr * ((double) run_data->k[0][i]) * w_hat[indx];
		}
	}

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier velocites to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), dw_hat_dt, u);

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
		}
	}

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier vorticity derivatives to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), dw_hat_dt, nabla_w);

	// -----------------------------------
	// Perform Convolution in Real Space
	// -----------------------------------
	// Perform the multiplication in real space
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j; 
 			
 			// Perform multiplication of the nonlinear term 
 			nonlinterm[indx] = -1.0 * ((u[SYS_DIM * indx + 0] * nabla_w[SYS_DIM * indx + 0]) + (u[SYS_DIM * indx + 1] * nabla_w[SYS_DIM * indx + 1]));
 		}
 	}

 	// -------------------------------------
 	// Transform Nonlinear Term To Fourier
 	// -------------------------------------
 	// Transform Fourier nonlinear term back to Fourier space
 	fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_r2c), nonlinterm, dw_hat_dt);


 	// -------------------------------------
 	// Apply Dealiasing to Nonlinear Term
 	// -------------------------------------
 	// Apply dealiasing and DFT normalization to the new nonlinear term
 	ApplyDealiasing(dw_hat_dt, sys_vars->N, norm_fac);



 	// Free memory
 	free(nonlinterm);
}
/**
 * Function to apply the selected dealiasing filter and the DFT normalization factor
 * @param w_hat    The nonlinear term that is to be dealiased
 * @param N        Array containing the dimensions of the system
 * @param norm_fac The normalization factor for the DFT: 1 / (Nx*Ny)
 */
void ApplyDealiasing(fftw_complex* w_hat, const long int* N, const double norm_fac) {

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
			indx = tmp + j;

			#ifdef __DEALIAS_23
			if (sqrt(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) < (int) ceil((double)Nx / 3.0)) {  // (abs(run_data->k[0][i]) < (int) ceil((double)Nx / 3.0)) && (abs(run_data->k[1][j]) < (int) ceil((double)Ny / 3.0))
				// Apply DFT normaliztin to undealiased modes
				w_hat[indx] *= norm_fac;
			}
			else {
				// Set dealised modes to 0
				w_hat[indx] = 0.0 + 0.0 * I;
			}
			#elif __DEALIAS_HOU_LI
			// Compute Hou-Li filter
			hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Nx / 2), 2.0) + pow(run_data->k[1][j] / (Ny / 2), 2.0))), 36.0));

			// Apply filter and DFT normaliztion
			w_hat[indx] *= (norm_fac * hou_li_filter);
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
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1; 

	// Initialize local variables 
	ptrdiff_t local_Nx = sys_vars->local_Nx;
	

	if(!(strcmp(sys_vars->u0, "TAYLOR_GREEN"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = (tmp + j);

				// Fill the velocities
				u[SYS_DIM * indx + 0] = sin(run_data->x[0][i]) * cos(run_data->x[1][j]);
				u[SYS_DIM * indx + 1] = -cos(run_data->x[0][i]) * sin(run_data->x[1][j]);		
			}
		}

		// Transform velocities to Fourier space
		fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_batch_r2c), u, u_hat);

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
	else if (!(strcmp(sys_vars->u0, "TESTING"))) {
		// Initialize temp variables
		double k_sqr;

		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if (run_data->k[0][i] == 0 || run_data->k[1][j] == 0){
					// Fill zero modes
					w_hat[indx] = 0.0 + 0.0 * I;
				}
				else {
					// Amplitudes
					k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + (double)1E-50);

					// Fill vorticity
					w_hat[indx] = k_sqr * cexp(I * M_PI / 4.0);
				}
			}
		}
	}
	else {
		// Use random initial conditions
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				w_hat[indx] = rand() * 2.0 * M_PI + rand() * 2.0 * M_PI * I;
			}
		}
	}
	
	// -------------------------------------------------
	// Initialize the Dealiasing
	// -------------------------------------------------
	ApplyDealiasing(w_hat, N, 1.0);
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
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	int tmp;
	int indx;
	double dt;
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

				// Compute the factor
				k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + (double) 1e-50);

				// compute the velocity in Fourier space
				run_data->u_hat[SYS_DIM * indx + 0] = ((double) run_data->k[1][j]) * k_sqr * run_data->w_hat[indx];
				run_data->u_hat[SYS_DIM * indx + 1] = -((double) run_data->k[0][i]) * k_sqr * run_data->w_hat[indx];
			}
		}

		// Transform back to Fourier space
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat, run_data->u);
	}
	else if (strcmp(dtype, "VORT") == 0) {
		// Perform transform back to real space for the vorticity
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->w_hat, run_data->w);
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
		fprintf(stderr, "\n[SOVLER FAILURE] --- System has reached maximum Vorticity limt at Iter: %d!\n-->> Exiting!!!n", iters);
		exit(1);
	}
	else if (dt <= MIN_STEP_SIZE) {
		fprintf(stderr, "\n[SOVLER FAILURE] --- Timestep has become too small to continue at Iter: %d!\n-->> Exiting!!!\n", iters);
		exit(1);		
	}
	else if (iters >= MAX_ITERS) {
		fprintf(stderr, "\n[SOVLER FAILURE] --- The maximum number of iterations has been reached at Iter: %d!\n-->> Exiting!!!\n", iters);
		exit(1);		
	}
}
/**
 * Function to update the timestep if adaptive timestepping is enabled
 * @param dt The current timestep
 */
void GetTimestep(double* dt) {

	// Initialize variables
	double dt_new;
	double w_max;

	// -------------------------------
	// Get Current Max Vorticity 
	// -------------------------------
	w_max = GetMaxData("VORT");
	
	// -------------------------------
	// Compute New Timestep
	// -------------------------------
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
}
/**
 * Function used to compute the energy spectrum of the current iteration. The energy spectrum is defined as all(sum) of the energy contained in concentric annuli in
 * wavenumber space. 	
 * @param  spectrum_size Variable used to store the size of the spectrum
 * @return               Pointer to the computed spectrum
 */
double* EnergySpectrum(int spectrum_size) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	double u_hat;
	double v_hat;
	fftw_complex k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// ------------------------------------
	// Allocate Memory
	// ------------------------------------
	// Size of the spectrum
	spectrum_size = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	
	// Allocate memory
	double* spectrum = (double* )fftw_malloc(sizeof(double) * spectrum_size);
	
	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute the prefactor
			k_sqr = I / (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + 1e-50);

			// Compute Fourier velocities
			u_hat = k_sqr * ((double) run_data->k[1][j]) * run_data->w_hat[indx];
			v_hat = -k_sqr * ((double) run_data->k[0][i]) * run_data->w_hat[indx];

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

			// Update the energy sum for the current mode
			spectrum[spec_indx] += cabs(u_hat * conj(u_hat)) + cabs(v_hat * conj(v_hat));
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
double* EnstrophySpectrum(int spectrum_size) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// ------------------------------------
	// Allocate Memory
	// ------------------------------------
	// Size of the spectrum
	spectrum_size = (int) sqrt((double) ((sys_vars->N[0] / 2) * (sys_vars->N[0] / 2) + (sys_vars->N[1] / 2) * (sys_vars->N[1] / 2))) + 1; // plus 1 for the 0 mode
	
	// Allocate memory
	double* spectrum = (double* )fftw_malloc(sizeof(double) * spectrum_size);
	
	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
			spec_indx = (int) sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));

			// Update the sum of the enstrophy in the current mode
			spectrum[spec_indx] += cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
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
	fftw_complex k_sqr;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// -------------------------------
	// Compute Fourier Space Velocity 
	// -------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// The prefactor
			k_sqr = I / (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + 1e-50);

			// Compute the Foureir velocities
			run_data->u_hat[SYS_DIM * indx + 0] = k_sqr * ((double) run_data->k[1][j]) * run_data->w_hat[indx];
			run_data->u_hat[SYS_DIM * indx + 1] = -k_sqr * ((double) run_data->k[0][i]) * run_data->w_hat[indx];
		}
	}

	// -------------------------------
	// Compute The Total Energy 
	// -------------------------------
	// Initialize the energy 
	double tot_energy = 0.0;

	// Loop over Fourier velocities
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update the sum for the total energy
			tot_energy += 0.5 * (cabs(run_data->u_hat[SYS_DIM * indx + 0] * conj(run_data->u_hat[SYS_DIM * indx + 0])) + cabs(run_data->u_hat[SYS_DIM * indx + 1] * conj(run_data->u_hat[SYS_DIM * indx + 1])));
		}
	}
	
	// Return result
	return tot_energy;	
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
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;


	// -------------------------------
	// Compute The Total Energy 
	// -------------------------------
	// Initialize total enstrophy
	double tot_enstr = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update the sum for the total enstrophy
			tot_enstr += 0.5 * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
		}
	}

	// Return result
	return tot_enstr;
}
/**
 * Function to compute the total palinstrophy of the system at the current timestep
 * @return  The total palinstrophy
 */
double TotalPalinstrophy(void) {
	
	// Initialize variables
	int tmp;
	int indx;
	double w_hat_dx;
	double w_hat_dy;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;


	// -------------------------------
	// Compute The Total Energy 
	// -------------------------------
	// Initialize total enstrophy
	double tot_palin = 0.0;

	// Loop over Fourier vorticity
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute the sum for the total palinstrophy
			w_hat_dx = I * ((double ) run_data->k[0][i]) * run_data->w_hat[indx];
			w_hat_dy = I * ((double ) run_data->k[1][j]) * run_data->w_hat[indx];
			tot_palin += 0.5 * (cabs(w_hat_dx * conj(w_hat_dx)) + cabs(w_hat_dy * conj(w_hat_dy)));
		}
	}

	// Return result
	return tot_palin;
}	
/**
 * Function to record the system measures for the current timestep 
 * @param t          The current time in the simulation
 * @param print_indx The current index of the measurables arrays
 */
void RecordSystemMeasures(double t, int print_indx) {

	// -------------------------------
	// Record the System Measures 
	// -------------------------------
	// The integration time
	#ifdef __TIME
	if (!(sys_vars->rank)) {
		run_data->time[print_indx] = t;
	}
	#endif

	// Total Energy, enstrophy and palinstrophy
	run_data->tot_enstr[print_indx] = TotalEnstrophy();
	run_data->tot_energy[print_indx] = TotalEnergy();
	run_data->tot_palin[print_indx] = TotalPalinstrophy();
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
		printf("[WARNING] --- FFTW was unable to allocate local memory for each process -->> Code will run but will be slown\n");
	}
	
	// -------------------------------
	// Allocate Space Variables 
	// -------------------------------
	// Allocate the wavenumber arrays
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->local_Nx);  // kx
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * Ny_Fourier);     		// ky
	if (run_data->k[0] == NULL || run_data->k[1] == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for wavenumber list \n-->> Exiting!!!\n");
		exit(1);
	}

	// Allocate the collocation points
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->local_Nx);  // x direction 
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * Ny);     			  // y direction
	if (run_data->x[0] == NULL || run_data->x[1] == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for collocation points \n-->> Exiting!!!\n");
		exit(1);
	}
	// -------------------------------
	// Allocate System Variables 
	// -------------------------------
	// Allocate the Real and Fourier space vorticity
	run_data->w     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Real Space Vorticity \n-->> Exiting!!!\n");
		exit(1);
	}
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Fourier Space Vorticity \n-->> Exiting!!!\n");
		exit(1);
	}

	// Allocate the Real and Fourier space velocities
	run_data->u     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Real Space Velocities \n-->> Exiting!!!\n");
		exit(1);
	}
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Fourier Space Velocities \n-->> Exiting!!!\n");
		exit(1);
	}

	// -------------------------------
	// Allocate Integration Variables 
	// -------------------------------
	// Runge-Kutta Integration arrays
	RK_data->nabla_psi = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->nabla_psi == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "nabla_psi");
		exit(1);
	}
	RK_data->nabla_w   = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->nabla_w == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "nabla_w");
		exit(1);
	}
	RK_data->RK1       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK1 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK1");
		exit(1);
	}
	RK_data->RK2       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK2 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK2");
		exit(1);
	}
	RK_data->RK3       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK3 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK3");
		exit(1);
	}
	RK_data->RK4       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK4 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK4");
		exit(1);
	}
	RK_data->RK_tmp    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK_tmp == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK_tmp");
		exit(1);
	}
	#ifdef __RK5
	RK_data->RK5       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK5 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK5");
		exit(1);
	}
	RK_data->RK6       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK6 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK6");
		exit(1);
	}
	#endif
	#ifdef __DPRK5
	RK_data->RK7       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK7 == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "RK7");
		exit(1);
	}
	RK_data->w_hat_last = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->w_hat_last == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for Integration Array [%s] \n-->> Exiting!!!\n", "w_hat_last");
		exit(1);
	}
	#endif
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
		fprintf(stderr, "\n[ERROR] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}

	// -------------------------------------
	// Initialize batch Plans for Velocity 
	// -------------------------------------
	// Set up FFTW plans for batch transform - velocity fields
	sys_vars->fftw_2d_dft_batch_r2c = fftw_mpi_plan_many_dft_r2c((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u, run_data->u_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	sys_vars->fftw_2d_dft_batch_c2r = fftw_mpi_plan_many_dft_c2r((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u_hat, run_data->u, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	if (sys_vars->fftw_2d_dft_batch_r2c == NULL || sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
}
/**
 * Function to initialize the system measurables and compute the measurables of the initial conditions
 */
void InitializeSystemMeasurables(void) {

	// Set the size of the arrays to twice the number of printing steps to account for extra steps due to adaptive stepping
	#ifdef __ADAPTIVE_STEP
	int print_steps = 2 * sys_vars->num_print_steps;
	#else
	int print_steps = sys_vars->num_print_steps;
	#endif
		
	// ------------------------
	// Allocate Memory
	// ------------------------
	// Total Energy in the system
	run_data->tot_energy = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_energy == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for the Total Energy \n-->> Exiting!!!\n");
		exit(1);
	}	

	// Total Enstrophy
	run_data->tot_enstr = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_enstr == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for the Total Enstrophy \n-->> Exiting!!!\n");
		exit(1);
	}	

	// Total Palinstrophy
	run_data->tot_palin = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_palin == NULL) {
		fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for the Total Palinstrophy \n-->> Exiting!!!\n");
		exit(1);
	}	

	// Time
	#ifdef __TIME
	if (!(sys_vars->rank)){
		run_data->time = (double* )fftw_malloc(sizeof(double) * print_steps);
		if (run_data->time == NULL) {
			fprintf(stderr, "\n[ERROR] --- Unable to allocate memory for the Time \n-->> Exiting!!!\n");
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

	// Time
	#ifdef __TIME
	if (!(sys_vars->rank)) {
		run_data->time[0] = sys_vars->t0;
	}
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
	fftw_free(run_data->tot_enstr);
	fftw_free(run_data->tot_palin);
	fftw_free(run_data->tot_energy);
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