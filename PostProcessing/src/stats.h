/**
* @file stats.h  
* @author Enda Carroll
* @date Sept 2021
* @brief File containing function prototpyes for stats file
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
void AllocateStatsMemory(const long int* N);
void RealSpaceStats(int s);
void WriteStatsToFile(void);
void FreeStatsObjects(void);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------