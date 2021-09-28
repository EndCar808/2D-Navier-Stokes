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
	int dim_flag   = 0;
	int force_flag = 0;
	int output_dir_flag = 0;
	int input_dir_flag  = 0;

	// -------------------------------
	// Initialize Default Values
	// -------------------------------
	// Output file directory
	strncpy(file_info->output_dir, "../Data/Tmp", 512);  // Set default output directory to the Tmp folder
	strncpy(file_info->output_tag, "NO_TAG", 64);
	file_info->output_file_only = 0; // used to indicate if output file should be file only i.e., not output folder

	// -------------------------------
	// Parse CML Arguments
	// -------------------------------
	while ((c = getopt(argc, argv, "o:h:n:s:e:t:v:i:c:p:f:z:")) != -1) {
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
			default:
				fprintf(stderr, "\n["RED"ERROR"RESET"] Incorrect command line flag encountered\n");		
				fprintf(stderr, "Use"YELLOW" -o"RESET" to specify the output directory\n");
				fprintf(stderr, "Use"YELLOW" -i"RESET" to specify the input directory\n");
				fprintf(stderr, "\nExample usage:\n"CYAN"\tmpirun -n 4 ./bin/main -o \"../Data/Tmp\" -n 64 -n 64 -s 0.0 -e 1.0 -h 0.0001 -v 1.0 -i \"TG_VORT\" -t \"TEMP_RUN\" \n"RESET);
				fprintf(stderr, "\n-->> Now Exiting!\n\n");
				exit(1);
		}
	}


	return 0;
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
