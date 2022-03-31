/**
* @file post_proc.h  
* @author Enda Carroll
* @date Sept 2021
* @brief File containing function prototpyes for post processing functions
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
void RealSpaceStats(int s);
void FullFieldData();
void SectorPhaseOrderBruteForceFast(int s);
void SectorPhaseOrder(int s);
void SectorPhaseOrderPar(int s);
void EnstrophyFluxSpectrum(int snap);
void NonlinearRHS(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* nonlinterm, double* u, double* nabla_w);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
void AllocateMemory(const long int* N);
void InitializeFFTWPlans(const long int* N);
void FreeMemoryAndCleanUp(void);
void EnstrophySpectrum(void);
void EnergySpectrum(void);
void EnstrophySpectrumAlt(void);
void EnergySpectrumAlt(void);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------