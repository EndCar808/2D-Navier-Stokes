/**
* @file phase_sync.h  
* @author Enda Carroll
* @date Sept 2021
* @brief File containing function prototpyes for phase sync file
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
void PhaseSyncSector(int s, int pre_compute_flag);
void PhaseSync(int s);
void AllocatePhaseSyncMemory(const long int* N);
void FreePhaseSyncObjects(void);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------