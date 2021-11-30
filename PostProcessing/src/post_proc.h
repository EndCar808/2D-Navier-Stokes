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