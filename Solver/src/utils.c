/**
* @file utils.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the utilities functions for the pseudospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h> 
#include <math.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to read in arguements given at the command line upon execution of the solver
 * @param  argc The number of arguments given
 * @param  argv Array containg the arguments specified
 * @return      Returns 0 if the parsing of arguments has been successful
 */
int GetCMLArgs(int argc, char** argv) {

	// Initialize Variables
	int c;
	int dim_flag = 0;

	// -------------------------------
	// Initialize Default Values
	// -------------------------------
	// Output file directory
	strncpy(file_info->output_dir, "../Data/Tmp", 512);  // Set default output directory to the Tmp folder
	strncpy(file_info->output_tag, "NO_TAG", 64);
	// System dimensions
	sys_vars->N[0] = 64;
	sys_vars->N[1] = 64;
	// Integration time 
	sys_vars->t0 	   	= 0.0;
	sys_vars->dt 	   	= 1e-4;
	sys_vars->T  	   	= 1.0;
	sys_vars->CFL_CONST = sqrt(3);
	// Initial conditions
	strncpy(sys_vars->u0, "TG_VORT", 64);
	// System viscosity
	sys_vars->NU = 1.0;
	// Write to file every 
	sys_vars->SAVE_EVERY = 100;

	// -------------------------------
	// Parse CML Arguments
	// -------------------------------
	while ((c = getopt(argc, argv, "o:h:n:s:e:t:v:i:c:p:f:z:")) != -1) {
		switch(c) {
			case 'o':
				// Read in location of output directory
				strncpy(file_info->output_dir, optarg, 512);	
				break;
			case 'n':
				// Read in the dimensions of the system
				sys_vars->N[dim_flag] = atoi(optarg);

				// Check dimensions satisfy requirements
				if (sys_vars->N[dim_flag] <= 2) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The minimum dimension size of [%ld] must be greater than 2\n-->> Exiting!\n\n", sys_vars->N[dim_flag]);		
					exit(1);
				}
				else if (sys_vars->N[dim_flag] < sys_vars->rank) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The system dimension of [%ld] cannot be less than the number of MPI processes of [%d]\n-->> Exiting!\n\n", sys_vars->N[dim_flag], sys_vars->rank);		
					exit(1);
				}
				else if (sys_vars->N[dim_flag] % 2 != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The system dimension of [%ld] must be a multiple of 2\n-->> Exiting!\n", sys_vars->N[dim_flag]);		
					exit(1);
				}
				dim_flag++;
				break;
			case 's':
				// Read in intial time
				sys_vars->t0 = atof(optarg);	
				break;
			case 'e':
				// Read in final time
				sys_vars->T = atof(optarg);	
				if (sys_vars->T < sys_vars->t0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided end time: [%lf] must be greater than the initial time: [%lf]\n-->> Exiting!\n\n", sys_vars->T, sys_vars->t0);		
					exit(1);
				}
				break;
			case 'h':
				// Read in initial timestep
				sys_vars->dt = atof(optarg);
				if (sys_vars->dt <= 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided timestep: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->dt);		
					exit(1);
				}
				break;
			case 'c':
				// Read in value of the CFL -> this can be used to control the timestep
				sys_vars->CFL_CONST = atof(optarg);
				if (sys_vars->CFL_CONST <= 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided CFL Constant: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->CFL_CONST);		
					exit(1);
				}
				break;
			case 'v':
				// Read in the viscosity
				sys_vars->NU = atof(optarg);
				if (sys_vars->CFL_CONST <= 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided viscosity: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->NU);		
					exit(1);
				}
				break;
			case 'd':
				// Read in the Ekman drag coefficient
				sys_vars->EKMN_ALPHA = atof(optarg);
				break;
			case 'i':
				// Read in the initial conditions
				if (!(strcmp(optarg,"TG_VEL"))) {
					// The Taylor Green vortex - starting with the velocity
					strncpy(sys_vars->u0, "TG_VEL", 64);
					break;
				}
				else if (!(strcmp(optarg,"TG_VORT"))) {
					// The Taylor Green vortex - starting with the vorticity
					strncpy(sys_vars->u0, "TG_VORT", 64);
					break;
				}
				else if (!(strcmp(optarg,"DOUBLE_SHEAR_LAYER"))) {
					// Double Shear Layer - for testing the Euler solver
					strncpy(sys_vars->u0, "DOUBLE_SHEAR_LAYER", 64);
					break;
				}
				else if (!(strcmp(optarg,"TESTING"))) {
					// Specific ICs for testing - powerlaw amps and phases = pi/4
					strncpy(sys_vars->u0, "TESTING", 64);
					break;
				}
				else if (!(strcmp(optarg,"DECAY_TURB"))) {
					// Decay Turbulence -> McWilliams initial conditions
					strncpy(sys_vars->u0, "DECAY_TURB", 64);
					break;
				}
				else if (!(strcmp(optarg,"RANDOM"))) {
					// Random initial conditions
					strncpy(sys_vars->u0, "RANDOM", 64);
					break;
				}
				else {
					// No initial conditions specified -> this will default to random initial conditions
					strncpy(sys_vars->u0, "NONE", 64);
					break;
				}
				break;
			case 't':
				// Read in output directory tag
				strncpy(file_info->output_tag, optarg, 64);	
				break;
			case 'p':
				// Read in how often to print to file
				sys_vars->SAVE_EVERY = atoi(optarg);
				break;
			case 'f':
				// TODO: Read in forcing indicator
				break;
			case 'z':
				// TODO: Read in inputs from a given (.ini) file
				break;
			default:
				fprintf(stderr, "\n["RED"ERROR"RESET"] Incorrect command line flag encountered\n");		
				fprintf(stderr, "Use"YELLOW" -o"RESET" to specify the output directory\n");
				fprintf(stderr, "Use"YELLOW" -n"RESET" to specify the size of each dimension in the system\n");
				fprintf(stderr, "Use"YELLOW" -s"RESET" to specify the start time of the simulation\n");
				fprintf(stderr, "Use"YELLOW" -e"RESET" to specify the end time of the simulation\n");
				fprintf(stderr, "Use"YELLOW" -h"RESET" to specify the timestep\n");
				fprintf(stderr, "Use"YELLOW" -c"RESET" to specify the CFL constant for the adaptive stepping\n");
				fprintf(stderr, "Use"YELLOW" -v"RESET" to specify the system viscosity\n");
				fprintf(stderr, "Use"YELLOW" -i"RESET" to specify the initial condition\n");
				fprintf(stderr, "Use"YELLOW" -t"RESET" to specify the tag name to be used in the output file directory\n");
				fprintf(stderr, "Use"YELLOW" -f"RESET" to specify the forcing type\n");
				fprintf(stderr, "Use"YELLOW" -p"RESET" to specify how often to print to file\n");
				fprintf(stderr, "Use"YELLOW" -z"RESET" to specify an input file to read parameters from\n");
				fprintf(stderr, "\nExample usage:\n"CYAN"\tmpirun -n 4 ./bin/main -o \"../Data/Tmp\" -n 64 -n 64 -s 0.0 -e 1.0 -h 0.0001 -v 1.0 -i \"TG_VORT\" -t \"TEMP_RUN\" \n"RESET);
				fprintf(stderr, "-->> Now Exiting!\n\n");
				exit(1);
		}
	}

	return 0;
}
/**
 * Utility function that gathers all the local real space vorticity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVorticityReal(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* w0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2));
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny + 2; ++j) {
			w0[i * (Ny + 2) + j] = 0.0;
		}
	}

	// Gather all the local arrays into w0
	MPI_Gather(run_data->w, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, w0, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("w[%ld]: %+5.16lf\t", i * (Ny) + j, w0[i * (Ny + 2) + j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w0);
}
/**
 * Utility function that gathers all the local Fourier space vorticity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVorticityFourier(const long int* N) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* w_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * (Ny_Fourier));
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			w_hat0[i * (Ny_Fourier) + j] = 0.0 + 0.0 * I;
		}
	}
	// Gather all the local arrays into w0
	MPI_Gather(run_data->w_hat, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, w_hat0, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("wh[%ld]: %+5.16lf %+5.16lfI\t", i * (Ny_Fourier) + j, creal(w_hat0[i * Ny_Fourier + j]), cimag(w_hat0[i * Ny_Fourier + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w_hat0);
}
/**
 * Utility function that gathers all the local real space velocity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVelocityReal(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* u0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2) * SYS_DIM);
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny + 2; ++j) {
			u0[SYS_DIM * (i * (Ny + 2) + j) + 0] = 0.0;
			u0[SYS_DIM * (i * (Ny + 2) + j) + 1] = 0.0;
		}
	}

	// Gather all the local arrays into u0 and v0
	MPI_Gather(run_data->u, sys_vars->local_Nx * (Ny + 2) * SYS_DIM, MPI_DOUBLE, u0, sys_vars->local_Nx * (Ny + 2)* SYS_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("u[%ld]: %+5.16lf\t", i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 0]);
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("v[%ld]: %+5.16lf\t", i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 1]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u0);
}
/**
 * Utility function that gathers all the local Fourier space velocity arrays on the master process and prints to screen 
 * @param N Array containing the dimensions of the system
 */
void PrintVelocityFourier(const long int* N) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* u_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 0] = 0.0 + 0.0 * I;
			u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 1] = 0.0 + 0.0 * I;
		}
	}

	// Gather all the local arrays into u_hat0 and v_hat0
	MPI_Gather(run_data->u_hat, sys_vars->local_Nx * Ny_Fourier * SYS_DIM, MPI_C_DOUBLE_COMPLEX, u_hat0, sys_vars->local_Nx * Ny_Fourier* SYS_DIM, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	
	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("uh[%ld]: %+5.10lf %+5.10lfI\t", i * (Ny_Fourier) + j, creal(u_hat0[SYS_DIM * (i * Ny_Fourier + j) + 0]), cimag(u_hat0[SYS_DIM * (i * Ny_Fourier + j) + 0]));
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("vh[%ld]: %+5.10lf %+5.10lf I\t", i * (Ny_Fourier) + j, creal(u_hat0[SYS_DIM * (i * Ny_Fourier + j) + 1]), cimag(u_hat0[SYS_DIM * (i * Ny_Fourier + j) + 1]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u_hat0);
}
/**
 * Function to print scalar array in Fourier space
 * @param data The data to be printed
 * @param N    The size of the dimensions
 */
void PrintScalarFourier(fftw_complex* data, const long int* N, char* arr_name) {

	// Initialize variables
	const long int Nx 		  = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* w_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * (Ny_Fourier));
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			w_hat0[(i * (Ny_Fourier) + j)] = 0.0 + 0.0 * I;
		}
	}

	// Gather all the local arrays into w0
	MPI_Gather(data, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, w_hat0, sys_vars->local_Nx * Ny_Fourier, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("%s[%ld]: %1.16lf %1.16lf I ", arr_name, i * (Ny_Fourier) + j, creal(w_hat0[i * Ny_Fourier + j]), cimag(w_hat0[i * Ny_Fourier + j]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w_hat0);
}
/**
 * Function to print scalar array in Real space
 * @param data The data to be printed
 * @param N    The size of the dimensions
 */
void PrintScalarReal(double* data, const long int* N, char* arr_name) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* w_hat0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2));
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny + 2; ++j) {
			w_hat0[(i * (Ny + 2) + j)] = 0.0;
		}
	}

	// Gather all the local arrays into w0
	MPI_Gather(data, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, w_hat0, sys_vars->local_Nx * (Ny + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("%s[%ld]: %g ", arr_name, i * (Ny) + j, w_hat0[i * (Ny + 2) + j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(w_hat0);
}
/**
 * Function for printing a vector array in Real space
 * @param data data to be printed
 * @param N    The size of the dimensions
 */
void PrintVectorReal(double* data, const long int* N, char* arr_name1, char* arr_name2) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// Allocate memory to recieve lcoal vorticity arrays
	double* u0 = (double* )fftw_malloc(sizeof(double) * Nx * (Ny + 2) * SYS_DIM);
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny + 2; ++j) {
			u0[SYS_DIM * (i * (Ny + 2) + j) + 0] = 0.0;
			u0[SYS_DIM * (i * (Ny + 2) + j) + 1] = 0.0;
		}
	}

	// Gather all the local arrays into u0 and v0
	MPI_Gather(data, sys_vars->local_Nx * (Ny + 2) * SYS_DIM, MPI_DOUBLE, u0, sys_vars->local_Nx * (Ny + 2)* SYS_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("%s[%ld]: %+5.16lf", arr_name1, i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 0]);
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				printf("%s[%ld]: %+5.16lf", arr_name2, i * (Ny) + j, u0[SYS_DIM * (i * (Ny + 2) + j) + 1]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u0);
}
void PrintVectorFourier(fftw_complex* data, const long int* N, char* arr_name1, char* arr_name2) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Allocate memory to recieve lcoal vorticity arrays
	fftw_complex* u_hat0 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * (Ny_Fourier) * SYS_DIM);
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 0] = 0.0 + 0.0 * I;
			u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 1] = 0.0 + 0.0 * I;
		}
	}

	// Gather all the local arrays into u_hat0 and v_hat0
	MPI_Gather(data, sys_vars->local_Nx * (Ny_Fourier) * SYS_DIM, MPI_C_DOUBLE_COMPLEX, u_hat0, sys_vars->local_Nx * (Ny_Fourier)* SYS_DIM, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	// On the master rank print the result
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("%s[%ld]: %5.16lf %5.16lf I", arr_name1, i * (Ny_Fourier) + j, creal(u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 0]), cimag(u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 0]));
			}
			printf("\n");
		}
		printf("\n\n");

		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny_Fourier; ++j) {
				printf("%s[%ld]: %5.16lf %5.16lf I", arr_name2, i * (Ny_Fourier) + j, creal(u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 1]), cimag(u_hat0[SYS_DIM * (i * (Ny_Fourier) + j) + 0]));
			}
			printf("\n");
		}
		printf("\n\n");
	}

	// Free temporary memory
	fftw_free(u_hat0);
}
/**
 * Function to print the Real and Fourier space variables
 */
void PrintSpaceVariables(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny_Fourier = N[1] / 2 + 1; 

	// Allocate global array memory
	double* x0 = (double* )fftw_malloc(sizeof(double) * Nx);
	int* k0    = (int* )fftw_malloc(sizeof(int) * Nx);

	// Gather the data from each process onto master rank for printing
	MPI_Gather(run_data->x[0], sys_vars->local_Nx, MPI_DOUBLE, x0, sys_vars->local_Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(run_data->k[0], sys_vars->local_Nx, MPI_INT, k0, sys_vars->local_Nx, MPI_INT, 0, MPI_COMM_WORLD);

	// Print results on the master process
	if ( !(sys_vars->rank) ) {
		for (int i = 0; i < Nx; ++i) {
			if (i < Ny_Fourier) {
				printf("x[%d]: %5.16lf\ty[%d]: %5.16lf\tkx[%d]: %d\tky[%d]: %d\n", i, x0[i], i, run_data->x[1][i], i, k0[i], i, run_data->k[1][i]);
			}
			else {
				printf("x[%d]: %5.16lf\ty[%d]: %5.16lf\tkx[%d]: %d\n", i, x0[i], i, run_data->x[1][i], i, k0[i]);
			}
		}
		printf("\n\n");
	}

	// Free up memory
	fftw_free(x0);
	fftw_free(k0);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------