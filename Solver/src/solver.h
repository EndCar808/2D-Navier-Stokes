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
#ifdef __RK4
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
#elif defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
#endif
#ifdef __DPRK5
double DPMax(double a, double b);
double DPMin(double a, double b);
#endif
void NonlinearRHSBatch(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u, double* w);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
// Initialize the system functions
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitialConditions(fftw_complex* w_hat, double* u, fftw_complex* u_hat, const long int* N);
double GetMaxData(char* dtype);
void InitializeSystemMeasurables(void);
// Timestep
void GetTimestep(double* dt);
// Check System
void SystemCheck(double dt, int iters);
// System Measurables 
double TotalEnergy(void);
double TotalEnstrophy(void);
double TotalPalinstrophy(void);
double* EnergySpectrum(int spectrum_size);
double* EnstrophySpectrum(int spectrum_size);
void RecordSystemMeasures(double t, int print_indx);
// Testing
void TestTaylorGreenVortex(const double t, const long int* N, double* norms);
void TaylorGreenSoln(const double t, const long int* N);
// Memory Functions
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data);
void InitializeFFTWPlans(const long int* N);
void FreeMemory(RK_data_struct* RK_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------