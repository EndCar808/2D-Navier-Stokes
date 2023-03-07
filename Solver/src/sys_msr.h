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
void InitializeSystemMeasurables(Int_data_struct* Int_data);
double TotalForcing(void);
double TotalDivergence(void);
double TotalEnergy(void);
double TotalEnstrophy(void);
double TotalPalinstrophy(void);
double EnergyDissipationRate(void);
double EnstrophyDissipationRate(void);
void EnergySpectrum(void);
void EnstrophySpectrum(void);
void EnergyFluxSpectrum(Int_data_struct* Int_data);
void EnstrophyFluxSpectrum(Int_data_struct* Int_data);
void EnstrophyFlux(double* d_e_dt, double* enst_flux, double* enst_diss, Int_data_struct* Int_data);
void EnergyFlux(double* d_e_dt, double* enrg_flux, double* enrg_diss, Int_data_struct* Int_data);
void RecordSystemMeasures(double t, int print_indx,  Int_data_struct* Int_data);
void ComputeSystemMeasurables(double t, int iter, int save_iter, int tot_iter, Int_data_struct* Int_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------