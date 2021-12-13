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
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"
#include "post_proc.h"
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// Define the global points that will be pointed to the global structs
runtime_data_struct*      run_data;
system_vars_struct*       sys_vars;
HDF_file_info_struct*    file_info;
postprocess_data_struct* proc_data;
stats_data_struct*      stats_data;
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Initialize variables
	int tmp, tmp1, tmp2;
	int indx;
	int gsl_status;
	double k_sqr, k_sqr_fac, phase, amp;
	double w_max, w_min;
	double u_max, u_min;
	#if defined(__SEC_PHASE_SYNC)
	double r;
	double angle;
	int k_x, k1_x, k2_x, k2_y;
	double k_angle, k1_sqr, k1_angle, k2_sqr, k2_angle;
	int num_phases, num_triads[NUM_TRIAD_TYPES + 1];
	int tmp_k, tmp_k1, tmp_k2;
	double flux_pre_fac;
	double flux_wght;
	#endif

	
	// --------------------------------
	//  Create Global Stucts
	// --------------------------------
	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	HDF_file_info_struct   HDF_file_info;
	postprocess_data_struct postproc_data;
	stats_data_struct statistics_data;

	// Point the global pointers to these structs
	run_data   = &runtime_data;
	sys_vars   = &system_vars;
	file_info  = &HDF_file_info;
	proc_data  = &postproc_data;
	stats_data = &statistics_data;

	// --------------------------------
	//  Get Command Line Arguements
	// --------------------------------
	if ((GetCMLArgs(argc, argv)) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line aguments, check utils.c file for details\n");
		exit(1);
	}

	// --------------------------------
	//  Open Input File and Get Data
	// --------------------------------
	OpenInputAndInitialize(); 
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// --------------------------------
	//  Open Output File
	// --------------------------------
	OpenOutputFile();
	// --------------------------------
	//  Allocate Processing Memmory
	// --------------------------------
	AllocateMemory(sys_vars->N);

	InitializeFFTWPlans(sys_vars->N);	


	//////////////////////////////
	// Begin Snapshot Processing
	//////////////////////////////
	printf("\nStarting Snapshot Processing:\n");
	for (int s = 0; s < sys_vars->num_snaps; ++s) {  // sys_vars->num_snaps
		
		// Print update to screen
		printf("Snapshot: %d\n", s);
	
		// --------------------------------
		//  Read in Data
		// --------------------------------
		ReadInData(s);

		// --------------------------------
		//  Real Space Stats
		// --------------------------------
		#ifdef __REAL_STATS
		// Get min and max data for histogram limits
		w_max = 0.0;
		w_min = 1e8;
		gsl_stats_minmax(&w_min, &w_max, run_data->w, 1, Nx * Ny);

		u_max = 0.0; 
		u_min = 1e8;
		gsl_stats_minmax(&u_min, &u_max, run_data->u, 1, Nx * Ny * SYS_DIM);
		if (fabs(u_min) > u_max) {
			u_max = fabs(u_min);
		}

		// Set histogram ranges
		gsl_status = gsl_histogram_set_ranges_uniform(stats_data->w_pdf, w_min - 0.5, w_max + 0.5);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_pdf, 0.0 - 0.1, u_max + 0.5);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
			exit(1);
		}

		// Update histograms
		for (int i = 0; i < Nx; ++i) {
			tmp = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Add current values to appropriate bins
				gsl_status = gsl_histogram_increment(stats_data->w_pdf, run_data->w[indx]);
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Vorticity", s);
					exit(1);
				}
				gsl_status = gsl_histogram_increment(stats_data->u_pdf, fabs(run_data->u[SYS_DIM * indx + 0]));
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
					exit(1);
				}
				gsl_status = gsl_histogram_increment(stats_data->u_pdf, fabs(run_data->u[SYS_DIM * indx + 1]));
				if (gsl_status != 0) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Real Velocity", s);
					exit(1);
				}
			}
		}
		#endif

		// --------------------------------
		//  Spectra Data
		// --------------------------------
		#ifdef __SPECTRA
		// Compute the enstrophy spectrum
		EnstrophySpectrum();
        EnergySpectrum();
        EnstrophySpectrumAlt();
        EnergySpectrumAlt();
		#endif

		// --------------------------------
		//  Full Field Data
		// --------------------------------
		#if defined(__FULL_FIELD) 
		for (int i = 0; i < Nx; ++i) {
			if (abs(run_data->k[0][i]) < sys_vars->kmax) {
				tmp  = i * Ny_Fourier;	
				tmp1 = (sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
				tmp2 = (sys_vars->kmax - 1 - run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
				for (int j = 0; j < Ny_Fourier; ++j) {
					indx = tmp + j;
					if (abs(run_data->k[1][j]) < sys_vars->kmax) {

						// Compute |k|^2 and 1 / |k|^2
						k_sqr = (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]); 
						if (run_data->k[0][i] != 0 || run_data->k[1][j] != 0) {
							k_sqr_fac = 1.0 / k_sqr;
						}
						else {
							k_sqr_fac = 0.0;	
						}

						// Pre-compute data
						phase = fmod(carg(run_data->w_hat[indx]) + 2.0 * M_PI, 2.0 * M_PI);
						amp   = cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));

						// fill the full field phases and spectra
		 				if (sqrt(k_sqr) < sys_vars->kmax) {
		 					// No conjugate for ky = 0
		 					if (run_data->k[1][j] == 0) {
		 						// proc_data->k_full[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = 
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = phase;
		 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = cabs(run_data->w_hat[indx]);
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp * k_sqr_fac;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
		 					}
		 					else {
		 						// Fill data and its conjugate
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = phase;
		 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = fmod(-phase + 2.0 * M_PI, 2.0 * M_PI);
		 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] 	 = cabs(run_data->w_hat[indx]);
		 						proc_data->amps[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = cabs(run_data->w_hat[indx]);
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp * k_sqr_fac;
		 						proc_data->enrg[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = amp * k_sqr_fac;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = amp;
		 						proc_data->enst[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = amp;
		 					}
		 				}
		 				else {
		 					// All dealiased modes set to zero
							if (run_data->k[1][j] == 0) {
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = -50.0;
		 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 					}
		 					else {	
		 						proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]] = -50.0;
		 						proc_data->phases[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]] = -50.0;
		 						proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = -50.0;
		 						proc_data->amps[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = -50.0;
		 						proc_data->enrg[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 						proc_data->enrg[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = 0.0;
		 						proc_data->enst[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]   = 0.0;
		 						proc_data->enst[tmp2 + sys_vars->kmax - 1 - run_data->k[1][j]]   = 0.0;
		 					}
		 				}
					}	
				}						
			}
		}
		#endif

		// --------------------------------
		//  Phase Sync
		// --------------------------------
		#if defined(__SEC_PHASE_SYNC) 
		///------------------------ Individual Phases
		// #pragma omp parallel for shared(run_data->k, proc_data->phases, proc_data->theta, proc_data->phase_order, proc_data->phase_sect_pdf, proc_data->phase_sect_pdf_t, proc_data->triad_phase_order, proc_data->ord_triad_phase_order) private(num_phases, num_ordered_phases, r, angle, phase)
		for (int a = 0; a < sys_vars->num_sect; ++a) {
			// Initialize counter
			num_phases = 0;
			for (int i = 0; i < Nx; ++i) {
				if (abs(run_data->k[0][i]) < sys_vars->kmax && abs(run_data->k[0][i]) > 0) {
					tmp  = i * Ny_Fourier;	
					tmp1 = (sys_vars->kmax - 1 + run_data->k[0][i]) * (2 * sys_vars->kmax - 1);
					for (int j = 0; j < Ny_Fourier; ++j) {
						indx = tmp + j;
						if ((abs(run_data->k[1][j]) < sys_vars->kmax) && (run_data->k[1][j] != 0)) {

							// Compute the polar coords
			 				r     = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			 				angle = (double)run_data->k[0][i] / (double)run_data->k[1][j];

			 				// printf("r: %lf - kmax: %.0lf t: %lf \t an: %lf\t t+1: %lf\t\t kx: %d, ky: %d, a: %lf\n", r, sys_vars->kmax_sqr, proc_data->theta[a], angle, proc_data->theta[a + 1], run_data->k[0][i], run_data->k[1][j], (double)run_data->k[0][i] / (double)run_data->k[1][j]);
							
							// Update the phase order parameter for the current sector
			 				if ((r < sys_vars->kmax_sqr) && (angle >= proc_data->theta[a] && angle < proc_data->theta[a + 1])) {
								// Pre-compute phase data
								phase = proc_data->phases[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]];

								// Update the phase order sum
								proc_data->phase_order[a] += cexp(I * phase);
			 					
			 					// Update counter
			 					num_phases++;

								// Update individual phases pdfs for this sector
								gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf[a], phase - M_PI);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF", a, s);
									exit(1);
								}
								gsl_status = gsl_histogram_increment(proc_data->phase_sect_pdf_t[a], phase - M_PI);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s);
									exit(1);
								}
								gsl_status = gsl_histogram_accumulate(proc_data->phase_sect_wghtd_pdf_t[a], phase - M_PI, proc_data->amps[tmp1 + sys_vars->kmax - 1 + run_data->k[1][j]]);
								if (gsl_status != 0) {
									fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time", a, s);
									exit(1);
								}
			 				}
			 			}
			 		}
			 	}
			}
			// printf("Num: %d\t phase_order: %lf %lf I\n", num_phases, creal(proc_data->phase_order[a]), cimag(proc_data->phase_order[a]));

			// Normalize the phase order parameter
			proc_data->phase_order[a] /= num_phases;

			// Record the phase sync and average phase
			proc_data->phase_R[a]   = cabs(proc_data->phase_order[a]);
			proc_data->phase_Phi[a] = carg(proc_data->phase_order[a]);


			///------------------- Triad Phases
			// Initialize counters
			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
				num_triads[i] = 0;
			}
			for (int tmp_k_x = 0; tmp_k_x <= 2 * sys_vars->kmax - 1; ++tmp_k_x) {
				// Get k_x
				k_x = tmp_k_x - sys_vars->kmax + 1;

				if (abs(k_x) > 0) {
					for (int k_y = 1; k_y <= sys_vars->kmax; ++k_y) {
						// Get polar coords for the k wavevector
						k_sqr   = (double) (k_x * k_x + k_y * k_y);
						k_angle =  (double) k_x / (double) k_y; 

						// Check if the k wavevector is in the current sector
						if (k_sqr < sys_vars->kmax_sqr && (k_angle >= proc_data->theta[a] && k_angle < proc_data->theta[a + 1])) {
							for (int tmp_k1_x = 0; tmp_k1_x <= 2 * sys_vars->kmax - 1; ++tmp_k1_x) {
								// Get k1_x
								k1_x = tmp_k1_x - sys_vars->kmax + 1;

								if (abs(k1_x) > 0) {
									for (int k1_y = 1; k1_y <= sys_vars->kmax; ++k1_y) {
										// Get polar coords for k1
										k1_sqr   = (double) (k1_x * k1_x + k1_y * k1_y);
										k1_angle = (double) k1_x / (double) k1_y;

										// Check if k1 wavevector is in the current sector
										if(k1_sqr < sys_vars->kmax_sqr && (k1_angle >= proc_data->theta[a] && k1_angle < proc_data->theta[a + 1])) {
											// Find the k2 wavevector
											k2_x = k_x - k1_x;
											k2_y = k_y - k1_y;
											
											if (abs(k2_x) > 0 && abs(k2_y) > 0) {
												// Get polar coords for k2
												k2_sqr   = (double) (k2_x * k2_x + k2_y * k2_y);
												k2_angle = (double) k2_x / (double) k2_y; 

												// Check if k2 is in the current sector
												if (k2_sqr < sys_vars->kmax_sqr && (k2_angle >= proc_data->theta[a] && k2_angle < proc_data->theta[a + 1])) {
													// Get correct phase index
													tmp_k  = (sys_vars->kmax - 1 + k_x) * (2 * sys_vars->kmax - 1);
													tmp_k1 = (sys_vars->kmax - 1 + k1_x) * (2 * sys_vars->kmax - 1);
													tmp_k2 = (sys_vars->kmax - 1 + k2_x) * (2 * sys_vars->kmax - 1);

													// Get the triad phase
													phase = proc_data->phases[tmp_k1 + sys_vars->kmax - 1 + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax - 1 + k2_y] - proc_data->phases[tmp_k + sys_vars->kmax - 1 + k_y];
													phase = fmod(phase + 2.0 * M_PI, 2.0 * M_PI) - M_PI;

													// Get the wavevector prefactor
													flux_pre_fac = (double)(k1_x * k2_y - k2_x * k1_y) / (1.0 / k1_sqr - 1.0 / k2_sqr);

													// Get the weighting (modulus) of this term to the contribution to the flux
													flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax - 1 + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax - 1 + k2_y] * proc_data->amps[tmp_k + sys_vars->kmax - 1 + k_y]);

													// printf("Pre: %lf\t Wght: %lf\n", flux_pre_fac, flux_wght);

													// Check for each type of triad contribution to the flux
													if (GSL_SIGN(flux_pre_fac) > 0) {
														// This is type 1 -> when the k1 is orientated below k2 and magnitude of k2 < magnitude of k1
														proc_data->triad_phase_order[1][a] += cexp(I * phase);
														num_triads[1]++;

														// Update the PDFs
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[1][a], phase);
														if (gsl_status != 0) {
															printf("Err: %d\n", gsl_status);
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[1][a], phase);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[1][a], phase, flux_wght);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time", a, s);
															exit(1);
														}
													}
													else if (GSL_SIGN(flux_pre_fac) < 0){
														// This is type 2 -> when the k1 is orientated below k2 and magnitude of k2 > magnitude of k1
														proc_data->triad_phase_order[2][a] += cexp(I * phase);
														num_triads[2]++;

														// Update the PDFs
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[2][a], phase);
														if (gsl_status != 0) {
															printf("Err: %d\n", gsl_status);
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[2][a], phase);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[2][a], phase, flux_wght);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time", a, s);
															exit(1);
														}
													}
													else if (k_sqr < k1_sqr && k_sqr < k2_sqr){
														// This is type 3 -> when k1 and k2 are larger in magnitude to k (k3)
														proc_data->triad_phase_order[3][a] += cexp(I * phase);
														num_triads[3]++;

														// Update the PDFs
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[3][a], phase);
														if (gsl_status != 0) {
															printf("Err: %d\n", gsl_status);
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[3][a], phase);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[3][a], phase, flux_wght);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time", a, s);
															exit(1);
														}
													}
													else if (k_sqr > k1_sqr && k_sqr > k2_sqr){
														// This is the type 4 -> when k1 and k2 are smaller in magnitude to k (k3)
														proc_data->triad_phase_order[4][a] += cexp(I * phase);
														num_triads[4]++;

														// Update the PDFs
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[4][a], phase);
														if (gsl_status != 0) {
															printf("Err: %d\n", gsl_status);
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[4][a], phase);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[4][a], phase, flux_wght);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time", a, s);
															exit(1);
														}
													}

													// Update the phase order parameter with all types
													if (k_sqr < k1_sqr && k_sqr < k2_sqr){
														// If type 3 there is a negative contribution to the flux
														proc_data->triad_phase_order[0][a] += -1.0 * cexp(I * phase) * GSL_SIGN(flux_pre_fac);

														// Update the PDFs
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], phase);
														if (gsl_status != 0) {
															printf("Err: %d\n", gsl_status);
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], phase);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], phase, flux_wght);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time", a, s);
															exit(1);
														}
													}
													else {
														// Otherwise the sign of the wavevector prefactor determines the sign of the contribution
														proc_data->triad_phase_order[0][a] += cexp(I * phase) * GSL_SIGN(flux_pre_fac);

														// Update the PDFs
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], phase);
														if (gsl_status != 0) {
															printf("Err: %d\n", gsl_status);
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], phase);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time", a, s);
															exit(1);
														}
														gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], phase, flux_wght);
														if (gsl_status != 0) {
															fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time", a, s);
															exit(1);
														}
													}

													// Update the triad counter
													num_triads[0]++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
				
			}

			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
				// Normalize the phase order parameters
				proc_data->triad_phase_order[i][a] /= num_triads[i];
				
				// Record the phase syncs and average phases
				proc_data->triad_R[i][a]   = cabs(proc_data->triad_phase_order[i][a]);
				proc_data->triad_Phi[i][a] = carg(proc_data->triad_phase_order[i][a]); 
				// printf("a: %d type: %d Num: %d\t triad_phase_order: %lf %lf I\t\t R: %lf, Phi %lf\n", a, i, num_triads[i], creal(proc_data->triad_phase_order[i][a]), cimag(proc_data->triad_phase_order[i][a]), proc_data->triad_R[i][a], proc_data->triad_Phi[i][a]);
			}
			// printf("\n");

			// printf("a: %d Num: %d\t triad_phase_order: %lf %lf I\t Num: %d\t triad_phase_order: %lf %lf I \t Num: %d\t triad_phase_order: %lf %lf I\n", a, num_triads[0], creal(proc_data->triad_phase_order[0][a]), cimag(proc_data->triad_phase_order[0][a]), num_triads[1], creal(proc_data->triad_phase_order[1][a]), cimag(proc_data->triad_phase_order[1][a]), num_triads[2], creal(proc_data->triad_phase_order[2][a]), cimag(proc_data->triad_phase_order[2][a]));
			// printf("a: %d | Num: %d R0: %lf Phi0: %lf |\t Num: %d R1: %lf Phi1: %lf |\t Num: %d R2: %lf Phi2: %lf\n", a, num_triads[0], proc_data->triad_R[0][a], proc_data->triad_Phi[0][a], num_triads[1], proc_data->triad_R[1][a], proc_data->triad_Phi[1][a], num_triads[2], proc_data->triad_R[2][a], proc_data->triad_Phi[2][a]);

			//------------------- Reset phase order arrays for next iteration
			proc_data->phase_order[a] = 0.0 + 0.0 * I;
			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
				proc_data->triad_phase_order[i][a] = 0.0 + 0.0 * I;
			}			
		}
		#endif
		// --------------------------------
		//  Write Data to File
		// --------------------------------
		WriteDataToFile(run_data->time[s], s);
	}
	///////////////////////////////
	// End Snapshot Processing
	///////////////////////////////

	// ---------------------------------
	//  Final Write of Data and Close 
	// ---------------------------------
	// Write any remaining datasets to output file
	FinalWriteAndClose();

	// --------------------------------
	//  Clean Up
	// --------------------------------
	// Free allocated memory
	FreeMemoryAndCleanUp();


	return 0;
}
/**
 * Wrapper function used to allocate the nescessary data objects
 * @param N Array containing the dimensions of the system
 */
void AllocateMemory(const long int* N) {

	// Initialize variables
	int gsl_status;
	int tmp1, tmp2, tmp3;
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Compute maximum wavenumber
	sys_vars->kmax = (int) (Nx / 3);	

	// --------------------------------
	//  Allocate Field Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}

	// Allocate the Fourier stream funciton
	run_data->psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier);
	if (run_data->psi_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Stream Function");
		exit(1);
	}

	#ifdef __REAL_STATS
	// Allocate current Fourier vorticity
	run_data->w = (double* )fftw_malloc(sizeof(double) * Nx * Ny);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity");
		exit(1);
	}

	// Allocate current Fourier vorticity
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny_Fourier * SYS_DIM);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}

	// Allocate current Fourier vorticity
	run_data->u = (double* )fftw_malloc(sizeof(double) * Nx * Ny * SYS_DIM);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}
	#endif
	
	#ifdef __FULL_FIELD
	// Allocate memory for the full field phases
	proc_data->phases = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->phases == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field amplitudes
	proc_data->amps = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->amps == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Phases");
		exit(1);
	}

	// Allocate memory for the full field enstrophy spectrum
	proc_data->enst = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->enst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Enstrophy");
		exit(1);
	}	

	// Allocate memory for the full field enrgy spectrum
	proc_data->enrg = (double* )fftw_malloc(sizeof(double) * (2 * sys_vars->kmax - 1) * (2 * sys_vars->kmax - 1));
	if (proc_data->enrg == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Full Field Energy");
		exit(1);
	}		
	#endif

	#ifdef __SPECTRA
	// Get the size of the spectra
	sys_vars->n_spec = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0)) + 1;

	// Allocate memory for the enstrophy spectrum
	proc_data->enst_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Enstrophy Spectrum");
		exit(1);
	}
    
    // Allocate memory for the enstrophy spectrum
	proc_data->enrg_spec = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enrg_spec == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Energy Spectrum");
		exit(1);
	}

	// Allocate memory for the alternative enstrophy spectrum
	proc_data->enst_alt = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enst_alt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Enstrophy Spectrum");
		exit(1);
	}
    
    // Allocate memory for the alternative enstrophy spectrum
	proc_data->enrg_alt = (double* )fftw_malloc(sizeof(double) * sys_vars->n_spec);
	if (proc_data->enrg_alt == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "1D Energy Spectrum");
		exit(1);
	}

	// Initialize arrays
	for (int i = 0; i < sys_vars->n_spec; ++i) {
		proc_data->enst_spec[i] = 0.0;
        proc_data->enrg_spec[i] = 0.0;
        proc_data->enst_alt[i]  = 0.0;
        proc_data->enrg_alt[i]  = 0.0;
	}
	#endif

	// --------------------------------	
	//  Allocate Phase Sync Data
	// --------------------------------
	#ifdef __SEC_PHASE_SYNC
	// Get kmax ^2
	sys_vars->kmax_sqr = pow(sys_vars->kmax, 2.0);

	///--------------- Sector Angles
	// Allocate the array of sector angles
	proc_data->theta = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect + 1));
	if (proc_data->theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Sector Angles");
		exit(1);
	}

	///--------------- Individual Phases
	// Allocate memory for the phase order parameter for the individual phases
	proc_data->phase_order = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
	if (proc_data->phase_order == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Phase Order Parameter");
		exit(1);
	}
	// Allocate the array of phase sync per sector for the individual phases
	proc_data->phase_R = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
	if (proc_data->phase_R == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Parameter");
		exit(1);
	}
	// Allocate the array of average phase per sector for the individual phases
	proc_data->phase_Phi = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
	if (proc_data->phase_Phi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Phase");
		exit(1);
	}

	///------------- Triad Phases
	// Allocate memory for each  of the triad types
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		// Allocate memory for the phase order parameter for the triad phases
		proc_data->triad_phase_order[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
		if (proc_data->triad_phase_order[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
		if (proc_data->triad_R[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
		if (proc_data->triad_Phi[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
			exit(1);
		}
	}
	
	///--------------- Phase Order Stats Objects
	// Allocate memory for the arrays stats objects
	proc_data->phase_sect_pdf         = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	proc_data->phase_sect_pdf_t       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	proc_data->phase_sect_wghtd_pdf_t = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		proc_data->triad_sect_pdf[i]         = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
		proc_data->triad_sect_pdf_t[i]       = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
		proc_data->triad_sect_wghtd_pdf_t[i] = (gsl_histogram** )fftw_malloc(sizeof(gsl_histogram) * sys_vars->num_sect);
	}
	

	// Allocate stats objects for each sector and set ranges
	for (int i = 0; i < sys_vars->num_sect; ++i) {
		// Allocate pdfs for the individual phases
		proc_data->phase_sect_pdf[i] = gsl_histogram_alloc(N_BINS_SEC);
		if (proc_data->phase_sect_pdf[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF");
			exit(1);
		}	
		proc_data->phase_sect_pdf_t[i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
		if (proc_data->phase_sect_pdf_t[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phase PDF In Time");
			exit(1);
		}	
		proc_data->phase_sect_wghtd_pdf_t[i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
		if (proc_data->phase_sect_wghtd_pdf_t[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phase Weighted PDF In Time");
			exit(1);
		}	
		// Set bin ranges for the individual phases
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_pdf[i], -M_PI - 0.05,  M_PI + 0.05);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases PDF");
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_pdf_t[i], -M_PI - 0.05,  M_PI + 0.05);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases PDF In Time");
			exit(1);
		}
		gsl_status = gsl_histogram_set_ranges_uniform(proc_data->phase_sect_wghtd_pdf_t[i], -M_PI - 0.05,  M_PI + 0.05);
		if (gsl_status != 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Phases Weighted PDF In Time");
			exit(1);
		}

		// Allocate and set ranges for each of the triad types
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			// Allocate pdfs for the triad phases
			proc_data->triad_sect_pdf[j][i] = gsl_histogram_alloc(N_BINS_SEC);
			if (proc_data->triad_sect_pdf[j][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF");
				exit(1);
			}	
			proc_data->triad_sect_pdf_t[j][i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
			if (proc_data->triad_sect_pdf_t[j][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF In Time");
				exit(1);
			}
			proc_data->triad_sect_wghtd_pdf_t[j][i] = gsl_histogram_alloc(N_BINS_SEC_INTIME);
			if (proc_data->triad_sect_wghtd_pdf_t[j][i] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF In Time");
				exit(1);
			}	
			// Set bin ranges for the triad phases
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_pdf[j][i], -M_PI - 0.05,  M_PI + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_pdf_t[j][i], -M_PI - 0.05,  M_PI + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF In Time");
				exit(1);
			}
			gsl_status = gsl_histogram_set_ranges_uniform(proc_data->triad_sect_wghtd_pdf_t[j][i], -M_PI - 0.05,  M_PI + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phases PDF In Time");
				exit(1);
			}
		}
	}
	
	// Initialize arrays
	double dtheta = M_PI / (double )sys_vars->num_sect;
	for (int i = 0; i < sys_vars->num_sect; ++i) {
		if (i > 0) {
			// Precompute the tan(theta) here for efficiency - only fill internal angles
			proc_data->theta[i] = tan(-M_PI / 2.0 + i * dtheta); 
		}
			proc_data->phase_R[i]     = 0.0;
			proc_data->phase_Phi[i]   = 0.0;
			proc_data->phase_order[i] = 0.0 + 0.0 * I;
			for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
				proc_data->triad_R[j][i]     	   = 0.0;
				proc_data->triad_Phi[j][i]         = 0.0;
				proc_data->triad_phase_order[j][i] = 0.0 + 0.0 * I;
			}
	}
	// Now fill angles at +-pi/2
	proc_data->theta[0]  				 = proc_data->theta[1] - abs(proc_data->theta[2] - proc_data->theta[1]);              //tan(-M_PI / 2.0 + 2e-1 * dtheta);
	proc_data->theta[sys_vars->num_sect] = proc_data->theta[sys_vars->num_sect - 1] + abs(proc_data->theta[2] - proc_data->theta[1]);  //tan(M_PI / 2.0 - 2e-1 * dtheta);
	#endif

	// --------------------------------	
	//  Allocate Stats Data
	// --------------------------------
	#ifdef __REAL_STATS
	// Allocate vorticity histograms
	stats_data->w_pdf = gsl_histogram_alloc(N_BINS);
	if (stats_data->w_pdf == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity Histogram");
		exit(1);
	}	

	// Allocate velocity histograms
	stats_data->u_pdf = gsl_histogram_alloc(N_BINS);
	if (stats_data->u_pdf == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity In Time Histogram");
		exit(1);
	}	
	#endif

	// --------------------------------
	//  Initialize Arrays
	// --------------------------------
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		tmp2 = i * (Ny_Fourier);
		tmp3 = i * (2 * sys_vars->kmax - 1);
		for (int j = 0; j < Ny; ++j) {
			if (j < Ny_Fourier) {
				run_data->w_hat[tmp2 + j] 				  = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 0] = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * (tmp2 + j) + 1] = 0.0 + 0.0 * I;
			}
			if ((i < 2 * sys_vars->kmax - 1) && (j < 2 * sys_vars->kmax - 1)) {
				proc_data->phases[tmp3 + j] = 0.0;
				proc_data->enst[tmp3 + j]   = 0.0;
				proc_data->enrg[tmp3 + j]   = 0.0;
			}
				run_data->w[tmp1 + j] = 0.0;
				run_data->u[SYS_DIM * (tmp1 + j) + 0] = 0.0;
				run_data->u[SYS_DIM * (tmp1 + j) + 1] = 0.0;
		}
	}
}
/**
 * Wrapper function for initializing FFTW plans
 * @param N Array containing the size of the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const int N_batch[SYS_DIM] = {Nx, Ny};

	#ifdef __REAL_STATS
	// Initialize Fourier Transforms
	sys_vars->fftw_2d_dft_c2r = fftw_plan_dft_c2r_2d(Nx, Ny, run_data->w_hat, run_data->w, FFTW_ESTIMATE);
	if (sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	
	// Initialize Batch Fourier Transforms
	sys_vars->fftw_2d_dft_batch_c2r = fftw_plan_many_dft_c2r(SYS_DIM, N_batch, SYS_DIM, run_data->u_hat, NULL, SYS_DIM, 1, run_data->u, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
	#endif
}
/**
* Function to compute 1D energy spectrum from the Fourier vorticity
*/
void EnergySpectrumAlt(void) {
    
    // Initialize variables
    int tmp, indx;
	int spec_indx;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0); 
    double k_sqr;
    
    // --------------------------------
	//  Initialize Spectrum
	// --------------------------------	
    for(int i = 0; i < sys_vars->n_spec; ++i) {
        proc_data->enrg_alt[i] = 0.0;
    }

    // --------------------------------
	//  Get Psi
	// --------------------------------	
	for(int i = 0; i < Nx; ++i) {
	    tmp = i * Ny_Fourier;
	    for(int j = 0; j < Ny_Fourier; ++j) {
	        indx = tmp + j;

	        if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
	        	run_data->psi_hat[indx] = 0.0 + 0.0 * I;
	        }
	        else {
	       		k_sqr = 1.0 / ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

		        // Get Fourier stream function
		        run_data->psi_hat[indx] = run_data->w_hat[indx] * k_sqr;
	        }
	    }
	}

    // --------------------------------
	//  Compute spectrum
	// --------------------------------	
    for(int i = 0; i < Nx; ++i) {
        tmp = i * Ny_Fourier;
        for(int j = 0; j < Ny_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int ) round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
       		
       		// Compute the |k|^2
            k_sqr = ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
        
            if ((j == 0) || (j == Ny_Fourier - 1)) {
                proc_data->enrg_alt[spec_indx] += const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
            }
            else {
                proc_data->enrg_alt[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
            }
        }
    }
}
/**
* Function to compute 1D enstrophy spectrum from the Fourier vorticity
*/
void EnstrophySpectrumAlt(void) {
    
    // Initialize variables
    int tmp, indx;
	int spec_indx;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0); 
    double k_sqr;
    
    // --------------------------------
	//  Initialize Spectrum
	// --------------------------------	
    for(int i = 0; i < sys_vars->n_spec; ++i) {
        proc_data->enst_alt[i] = 0.0;
    }


    // --------------------------------
	//  Get Psi
	// --------------------------------	
	for(int i = 0; i < Nx; ++i) {
	    tmp = i * Ny_Fourier;
	    for(int j = 0; j < Ny_Fourier; ++j) {
	        indx = tmp + j;

	        if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
	        	run_data->psi_hat[indx] = 0.0 + 0.0 * I;
	        }
	        else {
	       		k_sqr = 1.0 / ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

		        // Get Fourier stream function
		        run_data->psi_hat[indx] = run_data->w_hat[indx] * k_sqr;
	        }
	    }
	}

    // --------------------------------
	//  Compute spectrum
	// --------------------------------	
    for(int i = 0; i < Nx; ++i) {
        tmp = i * Ny_Fourier;
        for(int j = 0; j < Ny_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int ) round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
            
            // Compute |k|^2
            k_sqr = ((double) (pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));
        
            if ((j == 0) || (j == Ny_Fourier - 1)) {
                proc_data->enst_alt[spec_indx] += const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * pow(k_sqr, 2.0);
            }
            else {
                proc_data->enst_alt[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * pow(k_sqr, 2.0);
            }
        }
    }
}
/**
* Function to compute 1Denergy spectrum from the Fourier vorticity
*/
void EnergySpectrum(void) {
    
    // Initialize variables
    int tmp, indx;
	int spec_indx;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0); 
    double k_sqr;
    
    // --------------------------------
	//  Initialize Spectrum
	// --------------------------------	
    for(int i = 0; i < sys_vars->n_spec; ++i) {
        proc_data->enrg_spec[i] = 0.0;
    }
    

    // --------------------------------
	//  Compute spectrum
	// --------------------------------	
    for(int i = 0; i < Nx; ++i) {
        tmp = i * Ny_Fourier;
        for(int j = 0; j < Ny_Fourier; ++j) {
            indx = tmp + j;
            
            // Compute the spectral index
            spec_indx = (int) round(sqrt((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j])));
            
            if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				proc_data->enrg_spec[spec_indx] += 0.0;
			}
			else {
                // Compute the normalization factor 1.0 / |k|^2
                k_sqr = 1.0 / ((double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]));
            
                if ((j == 0) || (j == Ny_Fourier - 1)) {
                    proc_data->enrg_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                }
                else {
                    proc_data->enrg_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx])) * k_sqr;
                }
            }
        }
    }
}
/**
 * Function to compute the 1D enstrophy spectrum from the Fourier vorticity
 */
void EnstrophySpectrum(void) {

	// Initialize variables
	int tmp, indx;
	int spec_indx;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
	double const_fac = 4.0 * pow(M_PI, 2.0); 

	// --------------------------------
	//  Initialize Spectrum
	// --------------------------------	
    for(int i = 0; i < sys_vars->n_spec; ++i) {
        proc_data->enst_spec[i] = 0.0;
    }


	// --------------------------------
	//  Compute spectrum
	// --------------------------------	
	for (int i = 0; i < Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute spectrum index/bin
			spec_indx = (int )round(sqrt(pow(run_data->k[0][i], 2.0) + pow(run_data->k[1][j], 2.0)));

			if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)) {
				proc_data->enst_spec[spec_indx] += 0.0;
			}
			else {
				if ((j == 0) || (j == Ny_Fourier - 1)) {
					proc_data->enst_spec[spec_indx] += const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
				else {
					proc_data->enst_spec[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->w_hat[indx] * conj(run_data->w_hat[indx]));
				}
			}
		}
	}
}
/**
 * Wrapper function to free memory and close any objects before exiting
 */
void FreeMemoryAndCleanUp(void) {


	// --------------------------------
	//  Free memory
	// --------------------------------
	fftw_free(run_data->w_hat);
	fftw_free(run_data->time);
	fftw_free(run_data->psi_hat);
	#ifdef __REAL_STATS
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	#endif
	#ifdef __FULL_FIELD
	fftw_free(proc_data->phases);
	fftw_free(proc_data->amps);
	fftw_free(proc_data->enrg);
	fftw_free(proc_data->enst);
	#endif
	#ifdef __SPECTRA
	fftw_free(proc_data->enst_spec);
    fftw_free(proc_data->enrg_spec);
    fftw_free(proc_data->enst_alt);
    fftw_free(proc_data->enrg_alt);
	#endif
	#ifdef __SEC_PHASE_SYNC
	fftw_free(proc_data->theta);
	fftw_free(proc_data->phase_order);
	fftw_free(proc_data->phase_R);
	fftw_free(proc_data->phase_Phi);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		fftw_free(proc_data->triad_phase_order[i]);
		fftw_free(proc_data->triad_R[i]);
		fftw_free(proc_data->triad_Phi[i]);
	}
	#endif
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	#ifdef __REAL_STATS
	// Free histogram structs
	gsl_histogram_free(stats_data->w_pdf);
	gsl_histogram_free(stats_data->u_pdf);
	#endif
	#ifdef __SEC_PHASE_SYNC
	for (int i = 0; i < sys_vars->num_sect; ++i) {
		gsl_histogram_free(proc_data->phase_sect_pdf[i]);
		gsl_histogram_free(proc_data->phase_sect_pdf_t[i]);
		gsl_histogram_free(proc_data->phase_sect_wghtd_pdf_t[i]);
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			gsl_histogram_free(proc_data->triad_sect_pdf[j][i]);
			gsl_histogram_free(proc_data->triad_sect_pdf_t[j][i]);
			gsl_histogram_free(proc_data->triad_sect_wghtd_pdf_t[j][i]);
		}	
	}	
	#endif

	// --------------------------------
	//  Free FFTW Plans
	// --------------------------------
	#ifdef __REAL_STATS
	// Destroy FFTW plans
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);
	#endif

	// --------------------------------
	//  Close HDF5 Objects
	// --------------------------------
	// Close HDF5 identifiers
	herr_t status = H5Tclose(file_info->COMPLEX_DTYPE);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close compound datatype for complex data\n-->> Exiting...\n");
		exit(1);		
	}
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------