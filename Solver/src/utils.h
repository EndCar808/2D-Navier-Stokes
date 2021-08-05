/**
* @file utils.h
* @author Enda Carroll
* @date Jun 2021
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
void PrintVorticityReal(const long int* N);
void PrintVorticityFourier(const long int* N);
void PrintVelocityReal(const long int* N);
void PrintVelocityFourier(const long int* N);
void PrintScalarFourier(fftw_complex* data, const long int* N, char* arr_name);
void PrintVectorReal(double* data, const long int* N, char* arr_name1, char* arr_name2);
void PrintVectorFourier(fftw_complex* data, const long int* N, char* arr_name1, char* arr_name2);
void PrintSpaceVariables(const long int* N);
void PrintScalarReal(double* data, const long int* N, char* arr_name);
double DPMax(double a, double b);
double DPMin(double a, double b);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------