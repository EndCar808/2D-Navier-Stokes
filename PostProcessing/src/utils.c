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
	int output_dir_flag = 0;
	int input_dir_flag  = 0;
	int sector_flag     = 0;

	// -------------------------------
	// Initialize Default Values
	// -------------------------------
	// Output & Input file directory
	strncpy(file_info->output_dir, "NONE", 512);  // Set default output directory to the Tmp folder
	strncpy(file_info->input_dir, "NONE", 512);  // Set default output directory to the Tmp folder
	strncpy(file_info->output_tag, "NO_TAG", 64);
	file_info->input_file_only = 0; // used to indicate if input file was file only i.e., not output folder
	file_info->output_file_only = 0; // used to indicate if output file should be file only i.e., not output folder
	// Number of wavevector space sectors
	sys_vars->num_sect = 40;
	// Fraction of maximum wavevector
	sys_vars->kmax_frac = 1.0;
	// Set the default amount of threads to use
	sys_vars->num_threads = 1;
	// Viscosity
	sys_vars->NU = 1e-4;
	// Default number of k1 sectors
	sys_vars->num_k1_sectors = NUM_K1_SECTORS;
	// -------------------------------
	// Parse CML Arguments
	// -------------------------------
	while ((c = getopt(argc, argv, "a:o:h:n:s:e:t:v:i:c:p:f:z:k:t:")) != -1) {
		switch(c) {
			case 'o':
				if (output_dir_flag == 0) {
					// Read in location of output directory
					strncpy(file_info->output_dir, optarg, 512);
					output_dir_flag++;
				}
				else if (output_dir_flag == 1) {
					// Output file only indicated
					file_info->output_file_only = 1;
				}
				break;
			case 'i':
				if (input_dir_flag == 0) {
					// Read in location of input directory
					strncpy(file_info->input_dir, optarg, 512);
					input_dir_flag++;
				}
				else if (input_dir_flag == 1) {
					// Input file only indicated
					file_info->input_file_only = 1;
				}
				break;
			case 'a':
				if (sector_flag == 0) {
					// Get the number of sectors to use
					sys_vars->num_sect = atoi(optarg); 
					sector_flag++;
					if (sys_vars->num_sect <= 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of sector angles must be strictly positive, number provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_sect", sys_vars->num_sect);
						exit(1);
					}
					break;	
				}
				else if (sector_flag == 1) {
					// If full search is turned on set the number of k1 sectors to the number of sectors
					sys_vars->num_k1_sectors = sys_vars->num_sect;
					if (sys_vars->num_sect <= 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of sector angles must be strictly positive, number provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_sect", sys_vars->num_sect);
						exit(1);
					}
				}
				else {
					break;
				}				
			case 'p':
				// Get the number of omp threads to use
				sys_vars->num_threads = atoi(optarg); 
				if (sys_vars->num_threads <= 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of OMP threads must be greater than or equal to 1, umber provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_sect", sys_vars->num_threads);
					exit(1);
				}
				break;
			case 'k':
				// Get the fraction of kmax wavevectors to consider in the phase sync
				sys_vars->kmax_frac = atof(optarg); 
				if (sys_vars->kmax_frac <= 0 || sys_vars->kmax_frac > 1.0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], the fraction of maximum wavevector must be in (0, 1], fraction provided ["CYAN"%lf"RESET"]\n--->> Now Exiting!\n", "sys_vars->kmax_frac", sys_vars->kmax_frac);
					exit(1);
				}
				break;
			case 'v':
				// Get the fraction of kmax wavevectors to consider in the phase sync
				sys_vars->NU = atof(optarg); 
				if (sys_vars->NU < 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], the kinematic viscosity must positive, viscosity provided ["CYAN"%lf"RESET"]\n--->> Now Exiting!\n", "sys_vars->NU", sys_vars->NU);
					exit(1);
				}
				break;
			case 't':
				// Get the tag for the output file
				strncpy(file_info->output_tag, optarg, 64);
				break;
			default:
				fprintf(stderr, "\n["RED"ERROR"RESET"] Incorrect command line flag encountered\n");		
				fprintf(stderr, "Use"YELLOW" -o"RESET" to specify the output directory\n");
				fprintf(stderr, "Use"YELLOW" -i"RESET" to specify the input directory\n");
				fprintf(stderr, "Use"YELLOW" -a"RESET" to specify the number of sectors in wavevector space to use\n");
				fprintf(stderr, "Use"YELLOW" -p"RESET" to specify the number of OMP threads to use\n");
				fprintf(stderr, "Use"YELLOW" -k"RESET" to specify the frac of kmax to use as the set C\n");
				fprintf(stderr, "Use"YELLOW" -t"RESET" to specify the tag for the output file\n");
				fprintf(stderr, "\nExample usage:\n"CYAN"\tmpirun -n 4 ./bin/main -o \"../Data/Tmp\" -n 64 -n 64 -s 0.0 -e 1.0 -h 0.0001 -v 1.0 -i \"TG_VORT\" -t \"TEMP_RUN\" \n"RESET);
				fprintf(stderr, "\n-->> Now Exiting!\n\n");
				exit(1);
		}
	}


	return 0;
}
double MyMod(double x, double y) {

	double rem;
	rem = fmod(x, y);

	if (rem < 0) rem += y;

	return rem;
}
/**
 * Function to initialize the Real space collocation points arrays and Fourier wavenumber arrays
 * 
 * @param x Array containing the collocation points in real space
 * @param k Array to contain the wavenumbers on both directions
 * @param N Array containging the dimensions of the system
 */
void InitializeSpaceVariables(double** x, int** k, const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;


	// -------------------------------
	// Fill the first dirction 
	// -------------------------------
	for (int i = 0; i < Nx; ++i) {
		x[0][i] = (double) i * 2.0 * M_PI / (double) Nx;
		if(i <= Nx / 2) {
			k[0][i] = i;
		}
		else {
			k[0][i] = -Nx + i;
		}
	}

	// -------------------------------
	// Fill the second dirction 
	// -------------------------------
	for (int i = 0; i < Ny; ++i) {
		if (i < Ny_Fourier) {
			k[1][i] = i;
		}
		x[1][i] = (double) i * 2.0 * M_PI / (double) Ny;
	}
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
