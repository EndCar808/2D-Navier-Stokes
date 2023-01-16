/**
* @file main.c 
* @author Enda Carroll
* @date Sept 2021
* @brief Main file for post processing solver data from the 2D Navier stokes psuedospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <omp.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "data_types.h"
// #include "utils.h"
// #include "post_proc.h"
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
#define NUM_POW 6
#define NUM_INCR 2
#define SYS_DIM 3

typedef struct data_struct {
	double* str_func_data_par[NUM_POW];
	double* str_func_data_ser[NUM_POW];
	double* str_func_data_rad_par[NUM_POW];
	double* str_func_data_rad_ser[NUM_POW];
} data_struct; 
extern data_struct *data_st;

data_struct* data_st;
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	data_struct data_structure;
	
	// Point the global pointers to these structs
	data_st   = &data_structure;
	
	// Initialize variables
	int Nx = atoi(argv[1]);
	int Ny = Nx;
	int Max_Incr = Nx / 2;
	int num_threads = atoi(argv[2]);
	double increment;
	int tmp, indx, r_tmp, r_indx, x_indx, y_indx;
	double start, par_time, ser_time;
	double rad_increment;
	double norm_fac = 1.0 / (Nx * Ny);
	omp_set_num_threads(num_threads);
	printf("\n\nN: %d\t Max r: %d\tThreads: %d\n\n", Nx, Max_Incr, num_threads);

	// Allocate test memory
	double* data_u = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	double* data_w = (double* )fftw_malloc(sizeof(double) * Nx * Ny);
	double* str_func_par[NUM_POW];
	double* str_func_ser[NUM_POW];
	double* str_func_rad_par[NUM_POW];
	double* str_func_rad_ser[NUM_POW];

	// Initialize test memory
	for (int i = 0; i < Ny; ++i) {
		tmp = i * Nx;
		for (int j = 0; j < Nx; ++j) {
			indx = tmp + j;
			for (int l = 0; l < SYS_DIM; ++l) {
				data_u[SYS_DIM * (indx) + 0] = i * j + (i) / (j + 1);
				data_u[SYS_DIM * (indx) + 1] = i * j + (i) / (j + 1);
				data_w[indx] = i * j + (i) / (j + 1);
			}
		}
	}
	for (int p = 1; p <= NUM_POW; ++p) {
		str_func_ser[p - 1]                   = (double* )fftw_malloc(sizeof(double) * Max_Incr);
		str_func_par[p - 1]                   = (double* )fftw_malloc(sizeof(double) * Max_Incr);
		data_st->str_func_data_ser[p - 1]     = (double* )fftw_malloc(sizeof(double) * Max_Incr);
		data_st->str_func_data_par[p - 1]     = (double* )fftw_malloc(sizeof(double) * Max_Incr);
		str_func_rad_ser[p - 1]               = (double* )fftw_malloc(sizeof(double) * Max_Incr * Max_Incr);
		str_func_rad_par[p - 1]               = (double* )fftw_malloc(sizeof(double) * Max_Incr * Max_Incr);
		data_st->str_func_data_rad_par[p - 1] = (double* )fftw_malloc(sizeof(double) * Max_Incr * Max_Incr);
		data_st->str_func_data_rad_ser[p - 1] = (double* )fftw_malloc(sizeof(double) * Max_Incr * Max_Incr);
		for (int r = 0; r < Max_Incr; ++r){
			str_func_par[p - 1][r]               = 0.0;
			str_func_ser[p - 1][r]               = 0.0;
			data_st->str_func_data_par[p - 1][r] = 0.0;
			data_st->str_func_data_ser[p - 1][r] = 0.0;
			for (int b = 0; b < Max_Incr; ++b) {
				r_tmp = r * Max_Incr;
				r_indx = r_tmp + b;
				str_func_rad_ser[p - 1][r_indx]               = 0.0;
				str_func_rad_par[p - 1][r_indx]               = 0.0;
				data_st->str_func_data_rad_ser[p - 1][r_indx] = 0.0;
				data_st->str_func_data_rad_par[p - 1][r_indx] = 0.0;
				}
		}
	}
	
	// Get the grainsize of the tasks
	// int grain_size = NUM_POW * Max_Incr / num_threads;
	int grain_size = Nx * Ny / num_threads;

	// Get the grainsize of the tasks
	// int rad_grain_size = NUM_POW * Max_Incr * Max_Incr / num_threads;
	int rad_grain_size = Max_Incr * Max_Incr / num_threads;

	// Get parllel time
	start = omp_get_wtime();
	for (int snape = 0; snape < 100; ++snape)
	{
	#pragma omp parallel num_threads(num_threads) shared(data_u, str_func_par, str_func_rad_par, data_st) private(tmp, indx, r_tmp, r_indx, x_indx, y_indx)
	{
		#pragma omp single 
		{			
				// Loop over powers
				// #pragma omp taskloop reduction (+:increment) collapse(2) grainsize(grain_size)
				for (int p = 1; p <= NUM_POW; ++p) {
					// Loop over increments
					#pragma omp taskloop reduction (+:increment) grainsize(Max_Incr / num_threads)
					for (int r = 1; r <= Max_Incr; ++r) {
						// Initialize increment
						increment = 0.0;

						// Loop over space
						// #pragma omp taskloop reduction (+:increment) num_tasks( * Ny * Nx / num_threads)
						// #pragma omp taskloop reduction (+:increment) collapse(2) grainsize(grain_size)
						for (int i = 0; i < Ny; ++i) {
							for (int j = 0; j < Nx; ++j) {
								tmp = i * Nx;
								indx = tmp + j;

								// Compute increments
								increment += pow(fabs(data_u[SYS_DIM * (i * Nx + ((j + r) % Nx)) + 0]  - data_u[SYS_DIM * indx + 0]), p);
								increment += pow(fabs(data_u[SYS_DIM * ((((j + r) % Ny) * Nx + j)) + 1] - data_u[SYS_DIM * indx + 1]), p);
							}
						}
						// Update structure function
						// str_func_par[p - 1][r - 1]               = increment * norm_fac; 
						data_st->str_func_data_par[p - 1][r - 1] = increment * norm_fac; 
					}
					

					#pragma omp taskloop reduction (+:rad_increment) collapse(2) grainsize(rad_grain_size)
					for (int r_y = 1; r_y <= Max_Incr; ++r_y) {
						for (int r_x = 1; r_x <= Max_Incr; ++r_x) {
							r_tmp = (r_y - 1) * Max_Incr;
							r_indx = r_tmp + (r_x - 1);
							
							rad_increment = 0.0;

							// #pragma omp taskloop reduction (+:increment) collapse(3) grainsize((Nx * Ny *  / num_threads))
							for (int i = 0; i < Ny; ++i) {
								for (int j = 0; j < Nx; ++j) {
									tmp = i * Nx;
									indx = tmp + j;

									// Compute increments
									y_indx = (i + r_y) % Ny;
									x_indx = (j + r_x) % Nx;
									rad_increment += pow(fabs(data_w[y_indx * Nx + x_indx]  - data_w[indx]), p);
								}
							}
							// str_func_rad_par[p - 1][r_indx]               = rad_increment * norm_fac;
							data_st->str_func_data_rad_par[p - 1][r_indx] = rad_increment * norm_fac;
						}
					}
				}
			} 
		}
	}

	// Get parallel time
	par_time = omp_get_wtime() - start;
	
	
	// Serial 
	// Get parllel time
	start = omp_get_wtime();
	for (int snape = 0; snape < 100; ++snape)
	{
		for (int p = 1; p <= NUM_POW; ++p) {
			// Loop over increments
			for (int r = 1; r <= Max_Incr; ++r) {
				// Initialize increment
				increment = 0.0;

				// Loop over space
				for (int i = 0; i < Ny; ++i) {
					for (int j = 0; j < Nx; ++j) {
						tmp = i * Nx;
						indx = tmp + j;

						// Compute increments
						increment += pow(fabs(data_u[SYS_DIM * (i * Nx + ((j + r) % Nx)) + 0]  - data_u[SYS_DIM * indx + 0]), p);
						increment += pow(fabs(data_u[SYS_DIM * ((((j + r) % Ny) * Nx + j)) + 1] - data_u[SYS_DIM * indx + 1]), p);
					}
				}
				// Update structure function
				// str_func_ser[p - 1][r - 1]               = increment * norm_fac; 
				data_st->str_func_data_ser[p - 1][r - 1] = increment * norm_fac; 
			}
		
			for (int r_y = 1; r_y <= Max_Incr; ++r_y) {
				for (int r_x = 1; r_x <= Max_Incr; ++r_x) {
					r_tmp = (r_y - 1) * Max_Incr;
					r_indx = r_tmp + (r_x - 1);
					
					rad_increment = 0.0;

					for (int i = 0; i < Ny; ++i) {
						for (int j = 0; j < Nx; ++j) {
							tmp = i * Nx;
							indx = tmp + j;

							// Compute increments
							y_indx = (i + r_y) % Ny;
							x_indx = (j + r_x) % Nx;
							rad_increment += pow(fabs(data_w[y_indx * Nx + x_indx]  - data_w[indx]), p);
						}
					}
					// str_func_rad_ser[p - 1][r_indx]            = rad_increment * norm_fac;
					data_st->str_func_data_rad_ser[p - 1][r_indx] = rad_increment * norm_fac;
				}
			}
		}
	}
	ser_time = omp_get_wtime() - start;



	double ser_num, par_num;
	double ser_rad_num, par_rad_num;
	double st_ser_num, st_par_num;
	double st_ser_rad_num, st_par_rad_num;
	for (int p = 1; p <= NUM_POW; ++p) {
		ser_num = 0.0;
		par_num = 0.0;
		ser_rad_num = 0.0;
		par_rad_num = 0.0;
		st_ser_num = 0.0;
		st_par_num = 0.0;
		st_ser_rad_num = 0.0;
		st_par_rad_num = 0.0;
		for (int r = 0; r < Max_Incr; ++r) {
			ser_num += str_func_ser[p - 1][r];
			par_num += str_func_par[p - 1][r];
			st_ser_num += data_st->str_func_data_ser[p - 1][r];
			st_par_num += data_st->str_func_data_par[p - 1][r];
			for (int b = 0; b < Max_Incr; ++b) {
				tmp = r * Max_Incr;
				r_indx = tmp + b;
				ser_rad_num    += str_func_rad_ser[p - 1][r_indx];
				par_rad_num    += str_func_rad_par[p - 1][r_indx];
				st_ser_rad_num += data_st->str_func_data_rad_ser[p - 1][r_indx];
				st_par_rad_num += data_st->str_func_data_rad_par[p - 1][r_indx];
				// printf("str_func_rad[%d][%d]: %lf\t%lf\t%lf\n", p, r_indx, str_func_rad_ser[p - 1][r_indx], str_func_rad_par[p - 1][r_indx], str_func_rad_ser[p - 1][r_indx] - str_func_rad_par[p - 1][r_indx]);
			}
			// printf("str_func[%d][%d]: %lf\t%lf\t%lf\n", p, r, str_func_ser[p - 1][r], str_func_par[p - 1][r], str_func_ser[p - 1][r] - str_func_par[p - 1][r]);
		}
		printf("str_func[%d]: %16.16lf - %16.16lf | %16.16lf - %16.16lf\t\t | %lf \t %lf | %lf \t %lf\n", p, ser_num / Max_Incr, par_num / Max_Incr, ser_num / (Max_Incr * Max_Incr), par_num / (Max_Incr * Max_Incr), ser_num - par_num, ser_rad_num - par_rad_num, st_ser_num - st_par_num, st_ser_rad_num - st_par_rad_num);
	}


	printf("\n\nTimes: %lfs (ser) %lfs (par)\t\tSpeed Up: %lf\n", ser_time, par_time, ser_time / par_time);
	// printf("Rad Times: %lfs (ser) %lfs (par)\t\tRad Speed Up: %lf\n", rad_ser_time, rad_par_time, rad_ser_time / rad_par_time);


	for (int p = 0; p < NUM_POW; ++p) {
		fftw_free(str_func_ser[p]);
		fftw_free(str_func_par[p]);
		fftw_free(str_func_rad_ser[p]);
		fftw_free(str_func_rad_par[p]);
		fftw_free(data_st->str_func_data_ser[p]);
		fftw_free(data_st->str_func_data_par[p]);
		fftw_free(data_st->str_func_data_rad_ser[p]);
		fftw_free(data_st->str_func_data_rad_par[p]);
	}
	fftw_free(data_u);
	fftw_free(data_w);

	return 0;
}