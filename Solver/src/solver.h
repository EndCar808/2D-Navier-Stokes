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




// ---------------------------------------------------------------------
//  Function Prototpyes
// ---------------------------------------------------------------------
// Main function for the pseudospectral solver
void SpectralSolve(void);
// Integration functions
#if defined(__RK4) || defined(__RK4CN)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Ny, Int_data_struct* Int_data);
#endif
#if defined(__AB4CN)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Ny, Int_data_struct* Int_data);
void AB4Step(const double dt, const long int* N, const int iters, const ptrdiff_t local_Ny, Int_data_struct* Int_data);
#endif
#if defined(__RK5CN) || defined(__DPRK5CN)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Ny, Int_data_struct* Int_data);
#endif
#ifdef __DPRK5CN
double DPMax(double a, double b);
double DPMin(double a, double b);
#endif
void NonlinearTermAndForcing(fftw_complex* input_array, fftw_complex* output_array, double* nabla_psi) ;
void NonlinearRHSBatch(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
void ForceConjugacy(fftw_complex* array, const long int* N, const int dim);
void GetDissipativeTerm(double* diss, double* k_sqr);
// Initialize the system functions
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps);
void InitialConditions(fftw_complex* w_hat, double* u, fftw_complex* u_hat, const long int* N);
void InitializeFFTWPlans(const long int* N);
double GetMaxData(char* dtype);
// Timestep
void GetTimestep(double* dt);
// Check System
void SystemCheck(double dt, int iters);
// Print Update
void PrintUpdateToTerminal(int iters, double t, double dt, double T, int save_data_indx);
// Testing
void TestTaylorGreenVortex(const double t, const long int* N, double* norms);
void TaylorGreenSoln(const double t, const long int* N);
// Memory Functions
void AllocateMemory(const long int* NBatch, Int_data_struct* Int_data);
void FreeMemory(Int_data_struct* Int_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------