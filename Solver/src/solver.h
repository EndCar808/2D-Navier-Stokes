/**
* @file solver.h
* @author Enda Carroll
* @date Jun 2021
* @brief Header file containing the function prototypes for the solver.c file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "data_types.h"




// ---------------------------------------------------------------------
//  Function Prototpyes
// ---------------------------------------------------------------------
// Main function for the pseudospectral solver
void SpectralSolve(void);
// Integration functions
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
void RK5DPStep(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
void NonlinearRHSBatch(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u, double* w);
// Initialize the system functions
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitialConditions(fftw_complex* w_hat, double* u, fftw_complex* u_hat, const long int* N);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------