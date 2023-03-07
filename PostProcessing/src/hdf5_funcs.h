/**
* @file hdf5_funcs.h  
* @author Enda Carroll
* @date Sept 2021
* @brief File containing function prototpyes for hdf5_funcs file
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
void OpenInputAndInitialize(void);
hid_t CreateComplexDatatype(void);
void OpenOutputFile(void);
void ReadInData(int snap_indx);
void WriteDataToFile(double t, long int snap);
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, long int snap);
void FinalWriteAndClose(char* filename);
void CheckPointOutputFileName(int indx);
void CreateCheckPointCopy(int snap, int chk_pt_indx);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------