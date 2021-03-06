/**
* @file hdf5_funcs.h  
* @author Enda Carroll
* @date Jun 2021
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
void CreateOutputFilesWriteICs(const long int* N, double dt);
void GetOutputDirPath(void);
hid_t CreateComplexDatatype(void);
void FinalWriteAndCloseOutputFile(const long int* N, int iters, int save_data_indx);
void WriteDataFourier(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, fftw_complex* data);
void WriteDataReal(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, double* data);
void WriteDataSerial(double t, int iters, hid_t group_id, int dims, char* dset_name, double* data);
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters);
void OpenTestingFile(void);
void WriteDataToFile(double t, double dt, long int iters);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------