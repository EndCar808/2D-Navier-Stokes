/**
* @file utils.h
* @author Enda Carroll
* @date Sept 2021
* @brief Header file for the utils.c file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
void AllocateFullFieldMemory(const long int* N);
void FullFieldData(void);
void FluxSpectra(int snap);
void NonlinearRHS(fftw_complex* w_hat, fftw_complex* dw_hat_dt, double* u);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
void ForceConjugacy(fftw_complex* array, const long int* N, const int dim);
void EnstrophySpectrum(void);
void EnergySpectrum(void);
void EnstrophySpectrumAlt(void);
void EnergySpectrumAlt(void);
void FreeFullFieldObjects(void);
// ---------------------------------------------------------------------	
//  End of File
// ---------------------------------------------------------------------	