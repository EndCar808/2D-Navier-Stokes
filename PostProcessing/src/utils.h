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
int GetCMLArgs(int argc, char** argv);
double MyMod(double x, double y);
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void PrintTime(double start, double end);
double sgn(double x);
// ---------------------------------------------------------------------	
//  End of File
// ---------------------------------------------------------------------	