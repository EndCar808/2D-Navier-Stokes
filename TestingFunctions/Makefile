transform:
	mpicc  fftw_mpi_transform.c -o test_transform -lfftw3_mpi -lfftw3 -lm

rhs:
	mpicc  2d_vort_rhs.c -o test_rhs -lfftw3_mpi -lfftw3 -lm

dealias:
	mpicc  dealiasing.c -o test_dealias -lfftw3_mpi -lfftw3 -lm

batch_transform:
	mpicc  test_batch_mpi_transform.c -o test_batch_transform -lfftw3_mpi -lfftw3 -lm

par_str:
	gcc parallel_str_func.c -o par_str_func -I/usr/include/hdf5/serial -I/opt/software/fftw-3.3.10/include/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -L/opt/software/fftw-3.3.10/lib/ -lfftw3_omp -lfftw3 -lhdf5 -lhdf5_hl -fopenmp -lpthread -lm
