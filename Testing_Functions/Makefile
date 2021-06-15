transform:
	mpicc  fftw_mpi_transform.c -o test_transform -lfftw3_mpi -lfftw3 -lm

rhs:
	mpicc  2d_vort_rhs.c -o test_rhs -lfftw3_mpi -lfftw3 -lm

dealias:
	mpicc  dealiasing.c -o test_dealias -lfftw3_mpi -lfftw3 -lm
