/**
* @file utils.h
* @author Enda Carroll
* @date Jun 2021
* @brief Header file for the sys_msr.c file
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
void InitializeSystemMeasurables(RK_data_struct* RK_data);
double TotalForcing(void);
double TotalDivergence(void);
double TotalEnergy(void);
double TotalEnstrophy(void);
double TotalPalinstrophy(void);
double EnergyDissipationRate(void);
double EnstrophyDissipationRate(void);
void EnergySpectrum(void);
void EnstrophySpectrum(void);
void EnergyFluxSpectrum(RK_data_struct* RK_data);
void EnstrophyFluxSpectrum(RK_data_struct* RK_data);
void EnstrophyFlux(double* enst_flux, double* enst_diss, RK_data_struct* RK_data);
void EnergyFlux(double* enrg_flux, double* enrg_diss, RK_data_struct* RK_data);
void RecordSystemMeasures(double t, int print_indx,  RK_data_struct* RK_data);
void ComputeSystemMeasurables(double t, int iter, RK_data_struct* RK_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------