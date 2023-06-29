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
#include <time.h>
#include <sys/time.h>
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
	int visc_flag 	    = 0;
	int drag_flag       = 0;
	int force_flag      = 0;
	int threads_flag    = 0;
	int k_flag 			= 0;

	// -------------------------------
	// Initialize Default Values
	// -------------------------------
	// Output & Input file directory
	strncpy(file_info->output_dir, "NONE", 1024);  // Set default output directory to the Tmp folder
	strncpy(file_info->input_dir, "NONE", 1024);  // Set default output directory to the Tmp folder
	strncpy(file_info->output_tag, "No-Tag", 64);
	file_info->input_file_only = 0; // used to indicate if input file was file only i.e., not output folder
	file_info->output_file_only = 0; // used to indicate if output file should be file only i.e., not output folder
	// Number of wavevector space sectors
	sys_vars->num_k3_sectors = 24;
	sys_vars->num_k1_sectors = 24;
	sys_vars->REDUCED_K1_SEARCH_FLAG = 0;
	// Fraction of maximum wavevector
	sys_vars->kmax_frac = 1.0;
	sys_vars->kmin_sqr  = 0.0;
	// Set the default amount of threads to use
	sys_vars->num_threads = 1;
	sys_vars->num_fftw_threads = 1;
	// Viscosity
	sys_vars->NU = 1e-4;
	// Default number of k1 sectors
	sys_vars->num_k1_sectors = NUM_K1_SECTORS;
	// Forcing
	strncpy(sys_vars->forcing, "NONE", 64);	
	sys_vars->FORCING_FLAG    = 0;
	sys_vars->force_k         = 0;
	sys_vars->force_scale_var = 1.0;
	// Viscosity
	sys_vars->NU              = 0.1;
	sys_vars->HYPER_VISC_FLAG = 0;
	sys_vars->HYPER_VISC_POW  = VISC_POW;
	// Hypo drag
	sys_vars->EKMN_ALPHA      = 0.1;
	sys_vars->EKMN_DRAG_FLAG = 0;
	sys_vars->EKMN_DRAG_POW  = EKMN_POW;
	
	// -------------------------------
	// Parse CML Arguments
	// -------------------------------
	while ((c = getopt(argc, argv, "a:o:h:n:s:e:t:v:i:c:p:f:z:k:t:d:")) != -1) {
		switch(c) {
			case 'o':
				if (output_dir_flag == 0) {
					// Read in location of output directory
					strncpy(file_info->output_dir, optarg, 1024);
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
					strncpy(file_info->input_dir, optarg, 1024);
					input_dir_flag++;
				}
				else if (input_dir_flag == 1) {
					// Input file only indicated
					file_info->input_file_only = 1;
				}
				break;
			case 'a':
				if (sector_flag == 0) {
					// Get the number of k3 and k1 sectors to use
					sys_vars->num_k3_sectors = atoi(optarg); 
					sys_vars->num_k1_sectors = atoi(optarg);
					sector_flag++;
					if (sys_vars->num_k3_sectors <= 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of sector angles must be strictly positive, number provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_k3_sectors", sys_vars->num_k3_sectors);
						exit(1);
					}
					break;	
				}
				else if (sector_flag == 1) {
					// If full search is turned on set the number of k1 sectors to the number of sectors
					sys_vars->num_k1_sectors = atoi(optarg);
					sector_flag++;
					if (sys_vars->num_k3_sectors <= 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of sector angles must be strictly positive, number provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_k3_sectors", sys_vars->num_k3_sectors);
						exit(1);
					}
					break;
				}
				else if (sector_flag == 2) {
					// If reduced k1 sectors search is selected set to NUM_K1_SECTORS and turn on flag
					sys_vars->num_k1_sectors = NUM_K1_SECTORS;
					sys_vars->REDUCED_K1_SEARCH_FLAG = 1;
					sector_flag++;
				}
				else {
					break;
				}				
			case 'p':
				// Get the number of threads to use
				if (threads_flag == 0) {
					sys_vars->num_threads = atoi(optarg); 
					if (sys_vars->num_threads <= 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of OMP threads must be greater than or equal to 1, umber provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_threads", sys_vars->num_threads);
						exit(1);
					}
					threads_flag = 1;
					break;
				}
				else if (threads_flag == 1) {
					sys_vars->num_fftw_threads = atoi(optarg); 
					if (sys_vars->num_fftw_threads <= 0 || sys_vars->num_fftw_threads > 16) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of FFTW threads must be greater than or equal to 1 and less than 16, number provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_fftw_threads", sys_vars->num_fftw_threads);
						exit(1);
					}
					threads_flag = 2;
					break;	
				}
				break;
			case 'k':
				if (k_flag == 0) {
					// Get the fraction of kmax wavevectors to consider in the phase sync
					sys_vars->kmax_frac = atof(optarg); 
					if (sys_vars->kmax_frac <= 0 || sys_vars->kmax_frac > 1.0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], the fraction of maximum wavevector must be in (0, 1], fraction provided ["CYAN"%lf"RESET"]\n--->> Now Exiting!\n", "sys_vars->kmax_frac", sys_vars->kmax_frac);
						exit(1);
					}
					k_flag = 1;
					break;
				}
				if (k_flag == 1) {
					sys_vars->kmin_sqr = atof(optarg); 
					// Get the fraction of kmax wavevectors to consider in the phase sync
					if (sys_vars->kmin_sqr < 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], the minimum wavevector squared must be greater or equal to 0, value provided ["CYAN"%lf"RESET"]\n--->> Now Exiting!\n", "sys_vars->kmax_frac", sys_vars->kmin_sqr);
						exit(1);
					}
					k_flag = 2;
					break;
				}
				break;
			case 'v':
				if (visc_flag == 0) {
					// Read in the viscosity
					sys_vars->NU = atof(optarg); 
					if (sys_vars->NU < 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], the kinematic viscosity must positive, viscosity provided ["CYAN"%lf"RESET"]\n--->> Now Exiting!\n", "sys_vars->NU", sys_vars->NU);
						exit(1);
					}
					visc_flag = 1;
					break;
				}
				else if (visc_flag == 1) {
					// Read in hyperviscosity flag
					sys_vars->HYPER_VISC_FLAG = atoi(optarg);
					if ((sys_vars->HYPER_VISC_FLAG == 0) || (sys_vars->HYPER_VISC_FLAG == 1)) {
					}
					else {
						fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: Incorrect Hyperviscosity flag: [%d] Must be either 0 or 1 -- Set to 0 (no Hyperviscosity) by default\n-->> Exiting!\n\n", sys_vars->HYPER_VISC_FLAG);		
						exit(1);
					}
					visc_flag = 2;
					break;	
				}
				else if (visc_flag == 2) {
					// Read in the hyperviscosity power
					sys_vars->HYPER_VISC_POW = atof(optarg);
					if (sys_vars->HYPER_VISC_POW < 0.0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided hyperviscosity power: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->HYPER_VISC_POW);		
						exit(1);
					}
					// visc_flag = 3;
					break;
				}
				break;
			case 'd':
				if (drag_flag == 0) {
					// Read in the drag
					sys_vars->EKMN_ALPHA = atof(optarg);
					if (sys_vars->EKMN_ALPHA < 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided Ekman Drag: [%lf] must be positive\n-->> Exiting!\n\n", sys_vars->EKMN_ALPHA);		
						exit(1);
					}
					drag_flag = 1;
					break;
				}
				else if (drag_flag == 1) {
					// Read in Ekman drag flag
					sys_vars->EKMN_DRAG_FLAG = atoi(optarg);
					if ((sys_vars->EKMN_DRAG_FLAG == 0) || (sys_vars->EKMN_DRAG_FLAG == 1)) {
					}
					else {
						fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: Incorrect Ekman Drag flag: [%d] Must be either 0 or 1 -- Set to 0 (no drag) by default\n-->> Exiting!\n\n", sys_vars->EKMN_DRAG_FLAG);		
						exit(1);
					}
					drag_flag = 2;
					break;	
				}
				else if (drag_flag == 2) {
					// Read in the hypodiffusivity power
					sys_vars->EKMN_DRAG_POW = atof(optarg);
					if (sys_vars->EKMN_DRAG_POW > 0.0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided Hypodiffusibity power: [%lf] must be strictly negative\n-->> Exiting!\n\n", sys_vars->EKMN_DRAG_POW);		
						exit(1);
					}
					// drag_flag = 3;
					break;
				}
				break;
			case 'f':
				// Read in the forcing type
				if (!(strcmp(optarg,"ZERO")) && (force_flag == 0)) {
					// Killing certain modes
					strncpy(sys_vars->forcing, "ZERO", 64);
					sys_vars->FORCING_FLAG = 1;
					force_flag = 1;
					break;
				}
				else if (!(strcmp(optarg,"KOLM"))  && (force_flag == 0)) {
					// Kolmogorov forcing
					strncpy(sys_vars->forcing, "KOLM", 64);
					sys_vars->FORCING_FLAG = 1;
					force_flag = 1;
					break;
				}
				else if (!(strcmp(optarg,"NONE"))  && (force_flag == 0)) {
					// No forcing
					strncpy(sys_vars->forcing, "NONE", 64);
					sys_vars->FORCING_FLAG = 0;
					force_flag = 1;
					break;
				}
				else if (!(strcmp(optarg,"STOC"))  && (force_flag == 0)) {
					// Stochastic forcing
					strncpy(sys_vars->forcing, "STOC", 64);
					sys_vars->FORCING_FLAG = 1;
					force_flag = 1;
					break;
				}
				else if ((force_flag == 1)) {
					// Get the forcing wavenumber
					sys_vars->force_k = atoi(optarg);
					force_flag = 2;
				}
				else if ((force_flag == 2)) {
					// Get the force scaling variable
					sys_vars->force_scale_var = atof(optarg);
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
/**
 * Converts time in seconds into hours, minutes, seconds and prints to screen
 * @param start The wall time at the start of timing
 * @param end   The wall time at the end of timing
 */
void PrintTime(time_t start, time_t end) {

	// Get time spent in seconds
	double time_spent = (double)(end - start);

	// Get the hours, minutes and seconds
	int hh = (int) time_spent / 3600;
	int mm = ((int )time_spent - hh * 3600) / 60;
	int ss = time_spent - (hh * 3600) - (mm * 60);

	// Print hours minutes and second to screen
	printf("Time taken: ["CYAN"%5.10lf"RESET"] --> "CYAN"%d"RESET" hrs : "CYAN"%d"RESET" mins : "CYAN"%d"RESET" secs\n\n", time_spent, hh, mm, ss);

}
/**
 * Function to carry out the signum function 
 * @param  x double Input value
 * @return   Output of performing the signum function on the input
 */
double sgn(double x) {

	if (x < 0.0) return -1.0;
	if (x > 0.0) return 1.0;
	return 0.0;
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
