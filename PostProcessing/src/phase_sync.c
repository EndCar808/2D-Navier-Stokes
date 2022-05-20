/**
* @file stats.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the phase sync functions for the pseudospectral solver data
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
#include "utils.h"
#include "data_types.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to compute the phase synchroniztion data sector by sector in wavenumber space for the current snaphsot
 * @param s The index of the current snapshot
 */
void PhaseSyncSector(int s) {

	// Initialize variables
	int k1_x, k1_y, k2_x, k2_y, k3_x, k3_y;
	int tmp_k1, tmp_k2, tmp_k3;
	double k1_sqr, k2_sqr, k3_sqr;
	double flux_pre_fac;
	double flux_wght;
	double triad_phase;
	double gen_triad_phase;
	double S_k3, S_k3_lwr, S_k3_upr; 
	double S_k1, S_k1_lwr, S_k1_upr; 
	double k1_angle, k2_angle, k3_angle;
	double k1_angle_neg, k2_angle_neg, k3_angle_neg;
	int gsl_status;


	// Loop through the sectors for k3
	for (int a = 0; a < sys_vars->num_sect; ++a) {	

		// Get the sector for k3
		S_k3 = proc_data->theta[a];
		S_k3_upr = S_k3 + proc_data->dtheta/2.0;
		S_k3_lwr = S_k3 - proc_data->dtheta/2.0;

		// Initialize counters number of triads and enstrophy flux for each triad type
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			proc_data->num_triads[i][a]                    = 0;
			proc_data->enst_flux[i][a]                     = 0.0;
			proc_data->num_triads_1d[i][a]                 = 0;
			proc_data->enst_flux_1d[i][a]                  = 0.0;
			proc_data->phase_order_C_theta_triads[i][a]    = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads_1d[i][a] = 0.0 + 0.0 * I;
		}

		// Loop through the sectors for k1
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {

			// Initialize counters for number of triads and enstrophy flux across sectors for each triad type
			for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
				proc_data->num_triads_across_sec[i][a][l] = 0;
				proc_data->enst_flux_across_sec[i][a][l]  = 0.0;
			}

			// Get the angles for the second -> sector where k1 lies
			if (sys_vars->num_k1_sectors == NUM_K1_SECTORS) {
				S_k1 = MyMod(proc_data->theta[a] + (proc_data->k1_sector_angles[l] * proc_data->dtheta/2.0) + M_PI, 2.0 * M_PI) - M_PI;
			}
			else {
				S_k1 = proc_data->theta[(a + l) % sys_vars->num_sect];
			}
			S_k1_lwr = S_k1 - proc_data->dtheta / 2.0;
			S_k1_upr = S_k1 + proc_data->dtheta / 2.0;
			
			// Loop through wavevectors
			if (proc_data->num_wave_vecs[a][l] != 0) {
				for (int n = 0; n < proc_data->num_wave_vecs[a][l]; ++n) {
					
					// Get k1 and k2 and k3
					k1_x = (int) (proc_data->phase_sync_wave_vecs[a][l][K1_X][n]);
					k1_y = (int) (proc_data->phase_sync_wave_vecs[a][l][K1_Y][n]);
					k2_x = (int) (proc_data->phase_sync_wave_vecs[a][l][K2_X][n]);
					k2_y = (int) (proc_data->phase_sync_wave_vecs[a][l][K2_Y][n]);
					k3_x = (int) (proc_data->phase_sync_wave_vecs[a][l][K3_X][n]);
					k3_y = (int) (proc_data->phase_sync_wave_vecs[a][l][K3_Y][n]);

					// Get the mod square of the wavevectors
					k1_sqr = proc_data->phase_sync_wave_vecs[a][l][K1_SQR][n];
					k2_sqr = proc_data->phase_sync_wave_vecs[a][l][K2_SQR][n];
					k3_sqr = proc_data->phase_sync_wave_vecs[a][l][K3_SQR][n];

					// Get the angles of the wavevectors
					k1_angle     = proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE][n];
					k2_angle     = proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE][n];
					k3_angle     = proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE][n];
					k1_angle_neg = proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE_NEG][n];
					k2_angle_neg = proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE_NEG][n];
					k3_angle_neg = proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE_NEG][n];

					// Get correct phase index -> recall that to access kx > 0, use -kx
					tmp_k1 = (sys_vars->kmax - 1 - k1_x) * (2 * sys_vars->kmax - 1);	
					tmp_k2 = (sys_vars->kmax - 1 - k2_x) * (2 * sys_vars->kmax - 1);
					tmp_k3 = (sys_vars->kmax - 1 - k3_x) * (2 * sys_vars->kmax - 1);
					
					// Compute the flux pre factor
					flux_pre_fac = (double) (k1_x * k2_y - k2_x * k1_y) * (1.0 / k1_sqr - 1.0 / k2_sqr);

					// Get the flux weight term
					flux_wght = flux_pre_fac * (proc_data->amps[tmp_k1 + sys_vars->kmax - 1 + k1_y] * proc_data->amps[tmp_k2 + sys_vars->kmax - 1 + k2_y] * proc_data->amps[tmp_k3 + sys_vars->kmax - 1 + k3_y]);

					// Get the triad phase
					triad_phase = proc_data->phases[tmp_k1 + sys_vars->kmax - 1 + k1_y] + proc_data->phases[tmp_k2 + sys_vars->kmax - 1 + k2_y] - proc_data->phases[tmp_k3 + sys_vars->kmax - 1 + k3_y];

					// Update the Kuramoto order parameters for each triad type: Split into First (postive) term -> which has three scenarios; second (negative) term; zero contribution terms; and ignored terms
					if ((k3_sqr > sys_vars->kmax_C_sqr && ((k3_angle >= S_k3_lwr && k3_angle < S_k3_upr) || (k3_angle_neg >= S_k3_lwr && k3_angle_neg < S_k3_upr)))
						&& (k1_sqr < sys_vars->kmax_C_sqr && ((k1_angle >= S_k3_lwr && k1_angle < S_k3_upr) || (k1_angle_neg >= S_k3_lwr && k1_angle_neg < S_k3_upr)))
						&& !(k2_sqr > sys_vars->kmax_C_sqr && ((k2_angle >= S_k3_lwr && k2_angle < S_k3_upr) || (k2_angle_neg >= S_k3_lwr && k2_angle_neg < S_k3_upr))) ) {

						// Define the generalized triad phase for the first term in the flux
						gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
						
						//------------------------------------------ TRIAD TYPE 0
						// Update the combined triad phase order parameter with the appropriate contribution
						proc_data->triad_phase_order[0][a]               += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_1d[0][a]            += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_across_sec[0][a][l] += cexp(I * gen_triad_phase);

						// Update the triad counter for the combined triad type
						proc_data->num_triads[0][a]++;
						proc_data->num_triads_1d[0][a]++;
						proc_data->num_triads_across_sec[0][a][l]++;

						// Update the flux contribution for type 0
						proc_data->enst_flux[0][a]               += flux_wght * cos(triad_phase);
						proc_data->enst_flux_1d[0][a]            += flux_wght * cos(triad_phase);
						proc_data->enst_flux_across_sec[0][a][l] += flux_wght * cos(triad_phase);

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[0][a]    += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
								proc_data->phase_order_C_theta_triads_1d[0][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
							}
						}

						// ------ Update the PDFs of the combined triads
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(flux_wght));
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}

						if (flux_pre_fac < 0) {
							//------------------------------------------ TRIAD TYPE 1
							proc_data->triad_phase_order[1][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_1d[1][a]            += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[1][a][l] += cexp(I * gen_triad_phase);
							proc_data->num_triads[1][a]++;		
							proc_data->num_triads_1d[1][a]++;		
							proc_data->num_triads_across_sec[1][a][l]++;		

							// Update the flux contribution for tpye 1
							proc_data->enst_flux[1][a]               += flux_wght * cos(triad_phase);
							proc_data->enst_flux_1d[1][a]            += flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[1][a][l] += flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[1][a]    += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
									proc_data->phase_order_C_theta_triads_1d[1][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
								}
							}

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[1][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[1][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[1][a], gen_triad_phase, fabs(flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
						else if (flux_pre_fac > 0) {
							
							//------------------------------------------ TRIAD TYPE 2
							proc_data->triad_phase_order[2][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_1d[2][a]            += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[2][a][l] += cexp(I * gen_triad_phase);

							// Update the number of triads
							proc_data->num_triads[2][a]++;		
							proc_data->num_triads_1d[2][a]++;		
							proc_data->num_triads_across_sec[2][a][l]++;		

							// Update the flux contribution for tpye 2
							proc_data->enst_flux[2][a]               += flux_wght * cos(triad_phase);
							proc_data->enst_flux_1d[2][a]            += flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[2][a][l] += flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[2][a]    += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
									proc_data->phase_order_C_theta_triads_1d[2][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
								}
							}

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[2][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Type 2 PDF", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[2][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[2][a], gen_triad_phase, fabs(flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
					}
					else if ( (k3_sqr > sys_vars->kmax_C_sqr && ((k3_angle >= S_k3_lwr && k3_angle < S_k3_upr) || (k3_angle_neg >= S_k3_lwr && k3_angle_neg < S_k3_upr))) 
						&& (((k1_angle >= S_k1_lwr && k1_angle < S_k1_upr) || (k1_angle_neg >= S_k1_lwr && k1_angle_neg < S_k1_upr)) && !((k1_angle >= S_k3_lwr && k1_angle < S_k3_upr) || (k1_angle_neg >= S_k3_lwr && k1_angle_neg < S_k3_upr))) 
						&& !(k2_sqr > sys_vars->kmax_C_sqr && ((k2_angle >= S_k3_lwr && k2_angle < S_k3_upr) || (k2_angle_neg >= S_k3_lwr && k2_angle_neg < S_k3_upr))) ) {
						
						// Define the generalized triad phase for the first term in the flux
						gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;
						
						//------------------------------------------ TRIAD TYPE 0
						// Update the combined triad phase order parameter with the appropriate contribution
						proc_data->triad_phase_order[0][a]               += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_across_sec[0][a][l] += cexp(I * gen_triad_phase);

						// Update the triad counter for the combined triad type
						proc_data->num_triads[0][a]++;
						proc_data->num_triads_across_sec[0][a][l]++;

						// Update the flux contribution for type 0
						proc_data->enst_flux[0][a]               += flux_wght * cos(triad_phase);
						proc_data->enst_flux_across_sec[0][a][l] += flux_wght * cos(triad_phase);

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[0][a]    += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
							}
						}

						// ------ Update the PDFs of the combined triads
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(flux_wght));
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[1][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
							}
						}

						if (flux_pre_fac < 0) {
							//------------------------------------------ TRIAD TYPE 1
							proc_data->triad_phase_order[1][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[1][a][l] += cexp(I * gen_triad_phase);
							proc_data->num_triads[1][a]++;		
							proc_data->num_triads_across_sec[1][a][l]++;		

							// Update the flux contribution for tpye 1
							proc_data->enst_flux[1][a]               += flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[1][a][l] += flux_wght * cos(triad_phase);

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[1][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[1][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[1][a], gen_triad_phase, fabs(flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 1 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
						else if (flux_pre_fac > 0) {
							
							//------------------------------------------ TRIAD TYPE 2
							proc_data->triad_phase_order[2][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[2][a][l] += cexp(I * gen_triad_phase);

							// Update the number of triads
							proc_data->num_triads[2][a]++;		
							proc_data->num_triads_across_sec[2][a][l]++;		

							// Update the flux contribution for tpye 2
							proc_data->enst_flux[2][a]               += flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[2][a][l] += flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[2][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
								}
							}

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[2][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Type 2 PDF", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[2][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[2][a], gen_triad_phase, fabs(flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 2 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
					}
					else if ( (k3_sqr < sys_vars->kmax_C_sqr && ((k3_angle >= S_k3_lwr && k3_angle < S_k3_upr) || (k3_angle_neg >= S_k3_lwr && k3_angle_neg < S_k3_upr))) 
						&& (k1_sqr > sys_vars->kmax_C_sqr && ((k1_angle >= S_k3_lwr && k1_angle < S_k3_upr) || (k1_angle_neg >= S_k3_lwr && k1_angle_neg < S_k3_upr))) 
						&& (k2_sqr > sys_vars->kmax_C_sqr && ((k2_angle >= S_k3_lwr && k2_angle < S_k3_upr) || (k2_angle_neg >= S_k3_lwr && k2_angle_neg < S_k3_upr))) ) {
						// Define the generalized triad phase for the first term in the flux
						gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
						
						//------------------------------------------ TRIAD TYPE 0
						// Update the combined triad phase order parameter with the appropriate contribution
						proc_data->triad_phase_order[0][a]               += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_1d[0][a]            += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_across_sec[0][a][l] += cexp(I * gen_triad_phase);

						// Update the triad counter for the combined triad type
						proc_data->num_triads[0][a]++;
						proc_data->num_triads_1d[0][a]++;
						proc_data->num_triads_across_sec[0][a][l]++;


						// Update the flux contribution for type 0
						proc_data->enst_flux[0][a]               += -flux_wght * cos(triad_phase);
						proc_data->enst_flux_1d[0][a]            += -flux_wght * cos(triad_phase);
						proc_data->enst_flux_across_sec[0][a][l] += -flux_wght * cos(triad_phase);

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(-flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[0][a]    += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
								proc_data->phase_order_C_theta_triads_1d[1][a] += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
							}
						}

						// ------ Update the PDFs of the combined triads
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(-flux_wght));
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}

						if (flux_pre_fac < 0) {
							//------------------------------------------ TRIAD TYPE 3
							proc_data->triad_phase_order[3][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_1d[3][a]            += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[3][a][l] += cexp(I * gen_triad_phase);
							proc_data->num_triads[3][a]++;		
							proc_data->num_triads_1d[3][a]++;		
							proc_data->num_triads_across_sec[3][a][l]++;		

							// Update the flux contribution for tpye 3
							proc_data->enst_flux[3][a]               += -flux_wght * cos(triad_phase);
							proc_data->enst_flux_1d[3][a]            += -flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[3][a][l] += -flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(-flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[3][a]    += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
									proc_data->phase_order_C_theta_triads_1d[3][a] += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
								}
							}


							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[3][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[3][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[3][a], gen_triad_phase, fabs(-flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
						else if (flux_pre_fac > 0) {
							
							//------------------------------------------ TRIAD TYPE 4
							proc_data->triad_phase_order[4][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_1d[4][a]            += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[4][a][l] += cexp(I * gen_triad_phase);

							// Update the number of triads
							proc_data->num_triads[4][a]++;		
							proc_data->num_triads_1d[4][a]++;		
							proc_data->num_triads_across_sec[4][a][l]++;		

							// Update the flux contribution for tpye 4
							proc_data->enst_flux[4][a]               += -flux_wght * cos(triad_phase);
							proc_data->enst_flux_1d[4][a]            += -flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[4][a][l] += -flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(-flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[4][a]    += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
									proc_data->phase_order_C_theta_triads_1d[4][a] += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
								}
							}

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[4][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Type 4 PDF", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[4][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[4][a], gen_triad_phase, fabs(-flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
					}
					else if ( ((k3_angle >= S_k1_lwr && k3_angle < S_k1_upr) || (k3_angle_neg >= S_k1_lwr && k3_angle_neg < S_k1_upr)) && !(k3_sqr > sys_vars->kmax_C_sqr && ((k3_angle >= S_k3_lwr && k3_angle < S_k3_upr) || (k3_angle_neg >= S_k3_lwr && k3_angle_neg < S_k3_upr))) 
						&& (k1_sqr > sys_vars->kmax_C_sqr && ((k1_angle >= S_k3_lwr && k1_angle < S_k3_upr) || (k1_angle_neg >= S_k3_lwr && k1_angle_neg < S_k3_upr))) 
						&& (k2_sqr > sys_vars->kmax_C_sqr && ((k2_angle >= S_k3_lwr && k2_angle < S_k3_upr) || (k2_angle_neg >= S_k3_lwr && k2_angle_neg < S_k3_upr))) ) {
						
						// Define the generalized triad phase for the triads in the second (negative) flux term
						gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(-flux_wght), 2.0 * M_PI) - M_PI;
						
						//------------------------------------------ TRIAD TYPE 0
						// Update the combined triad phase order parameter with the appropriate contribution
						proc_data->triad_phase_order[0][a]               += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_across_sec[0][a][l] += cexp(I * gen_triad_phase);

						// Update the triad counter for the combined triad type
						proc_data->num_triads[0][a]++;
						proc_data->num_triads_across_sec[0][a][l]++;

						// Update the flux contribution for type 0
						proc_data->enst_flux[0][a]               += -flux_wght * cos(triad_phase);
						proc_data->enst_flux_across_sec[0][a][l] += -flux_wght * cos(triad_phase);

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[0][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
							}
						}

						// ------ Update the PDFs of the combined triads
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[0][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[0][a], gen_triad_phase, fabs(-flux_wght));
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 0 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}

						if (flux_pre_fac < 0) {
							//------------------------------------------ TRIAD TYPE 3
							proc_data->triad_phase_order[3][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[3][a][l] += cexp(I * gen_triad_phase);
							proc_data->num_triads[3][a]++;		
							proc_data->num_triads_across_sec[3][a][l]++;		

							// Update the flux contribution for tpye 1
							proc_data->enst_flux[3][a]               += -flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[3][a][l] += -flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(-flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[3][a] += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
								}
							}

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[3][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[3][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[3][a], gen_triad_phase, fabs(-flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 3 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
						if (flux_pre_fac > 0) {
							//------------------------------------------ TRIAD TYPE 4
							proc_data->triad_phase_order[4][a]               += cexp(I * gen_triad_phase);
							proc_data->triad_phase_order_across_sec[4][a][l] += cexp(I * gen_triad_phase);

							// Update the number of triads
							proc_data->num_triads[4][a]++;		
							proc_data->num_triads_across_sec[4][a][l]++;		

							// Update the flux contribution for tpye 2
							proc_data->enst_flux[4][a]               += -flux_wght * cos(triad_phase);
							proc_data->enst_flux_across_sec[4][a][l] += -flux_wght * cos(triad_phase);

							// Update collective phase order parameter for C_theta
							if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
								if (cabs(-flux_wght * cexp(I * triad_phase) != 0.0)) {
									proc_data->phase_order_C_theta_triads[4][a] += -flux_wght * cexp(I * triad_phase) / cabs(-flux_wght * cexp(I * triad_phase));
								}
							}

							// Update the PDFs
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[4][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[4][a], gen_triad_phase);
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
							gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[4][a], gen_triad_phase, fabs(-flux_wght));
							if (gsl_status != 0) {
								fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 4 In Time", a, s, gsl_status, gen_triad_phase);
								exit(1);
							}
						}
					}
					else if (flux_pre_fac == 0.0 || flux_wght == 0.0) {

						// Define the generalized triad phase for the zero contribution terms
						gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

						//------------------------------------------ TRIAD TYPE 5
						proc_data->triad_phase_order[5][a]               += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_across_sec[5][a][l] += cexp(I * gen_triad_phase);

						// Update the number of triads
						proc_data->num_triads[5][a]++;		
						proc_data->num_triads_across_sec[5][a][l]++;		

						// Update the flux contribution for tpye 2
						proc_data->enst_flux[5][a]               += flux_wght * cos(triad_phase);
						proc_data->enst_flux_across_sec[5][a][l] += flux_wght * cos(triad_phase);

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[5][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
							}
						}

						// Update the PDFs
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[5][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 5", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[5][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 5 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[5][a], gen_triad_phase, fabs(flux_wght)); 
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 5 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
					}
					else {

						// Define the generalized triad phase for the ignored terms
						gen_triad_phase = fmod(triad_phase + 2.0 * M_PI + carg(flux_wght), 2.0 * M_PI) - M_PI;

						//------------------------------------------ TRIAD TYPE 6
						proc_data->triad_phase_order[6][a]               += cexp(I * gen_triad_phase);
						proc_data->triad_phase_order_across_sec[6][a][l] += cexp(I * gen_triad_phase);

						// Update the number of triads
						proc_data->num_triads[6][a]++;		
						proc_data->num_triads_across_sec[6][a][l]++;		

						// Update the flux contribution for tpye 2
						proc_data->enst_flux[6][a]               += flux_wght * cos(triad_phase);
						proc_data->enst_flux_across_sec[6][a][l] += flux_wght * cos(triad_phase);

						// Update collective phase order parameter for C_theta
						if (k1_y > 0 && k2_y > 0 && k3_y > 0) {
							if (cabs(flux_wght * cexp(I * triad_phase) != 0.0)) {
								proc_data->phase_order_C_theta_triads[6][a] += flux_wght * cexp(I * triad_phase) / cabs(flux_wght * cexp(I * triad_phase));
							}
						}

						// Update the PDFs
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf[6][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 6", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_increment(proc_data->triad_sect_pdf_t[6][a], gen_triad_phase);
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase PDF Type 6 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
						gsl_status = gsl_histogram_accumulate(proc_data->triad_sect_wghtd_pdf_t[6][a], gen_triad_phase, fabs(flux_wght)); 
						if (gsl_status != 0) {
							fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Sector ["CYAN"%d"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Sector Triad Phase Weighted PDF Type 6 In Time", a, s, gsl_status, gen_triad_phase);
							exit(1);
						}
					}
				}
			}
		}
	}

	//------------------- Record the data for the triads
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
			// Normalize the phase order parameters
			if (proc_data->num_triads[i][a] != 0) {
				proc_data->triad_phase_order[i][a] /= proc_data->num_triads[i][a];
			}
			if (proc_data->num_triads_1d[i][a] != 0) {
				proc_data->triad_phase_order_1d[i][a] /= proc_data->num_triads_1d[i][a];
			}
			
			// Record the phase syncs and average phases
			proc_data->triad_R[i][a]      = cabs(proc_data->triad_phase_order[i][a]);
			proc_data->triad_Phi[i][a]    = carg(proc_data->triad_phase_order[i][a]);
			proc_data->triad_R_1d[i][a]   = cabs(proc_data->triad_phase_order_1d[i][a]);
			proc_data->triad_Phi_1d[i][a] = carg(proc_data->triad_phase_order_1d[i][a]);
			for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
				if (proc_data->num_triads_across_sec[i][a][l] != 0) {
					proc_data->triad_phase_order_across_sec[i][a][l] /= proc_data->num_triads_across_sec[i][a][l];
				}
				
				// Record the phase syncs and average phases
				proc_data->triad_R_across_sec[i][a][l]   = cabs(proc_data->triad_phase_order_across_sec[i][a][l]);
				proc_data->triad_Phi_across_sec[i][a][l] = carg(proc_data->triad_phase_order_across_sec[i][a][l]); 
			}
		}
	}

	//------------- Reset order parameters for next iteration
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int type = 0; type < NUM_TRIAD_TYPES; ++type) {
			proc_data->triad_phase_order[type][a]    = 0.0 + 0.0 * I;
			proc_data->triad_phase_order_1d[type][a] = 0.0 + 0.0 * I;
			for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
				proc_data->triad_phase_order_across_sec[type][a][l] = 0.0 + 0.0 * I;
			}
		}
	}
}
/**
 * Allocate memory and objects needed for the Phase sync computation
 * @param N Array containing the size of the each dimension
 */
void AllocatePhaseSyncMemory(const long int* N) {

	// Initialize variables
	int gsl_status;
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;

	// Get the various kmax variables
	sys_vars->kmax_sqr   = pow(sys_vars->kmax, 2.0);
	sys_vars->kmax_C   	 = (int) ceil(sys_vars->kmax_frac * sys_vars->kmax);
	sys_vars->kmax_C_sqr = pow(sys_vars->kmax_C, 2.0);


	// --------------------------------	
	//  Allocate Sector Angles
	// --------------------------------
	// Allocate the array of sector angles
	proc_data->theta = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect));
	if (proc_data->theta == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Sector Angles");
		exit(1);
	}


	// --------------------------------------------
	//  Allocate Number of Triads & Enstrophy Flux
	// ---------------------------------------------
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		///--------------- Number of Triads
		// Allocate memory for the phase order parameter for the triad phases
		proc_data->num_triads[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_sect);
		if (proc_data->num_triads[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}
		proc_data->num_triads_1d[i] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_sect);
		if (proc_data->num_triads_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector 1D");
			exit(1);
		}

		///--------------- Enstrophy Flux
		proc_data->enst_flux[i] = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect));
		if (proc_data->enst_flux[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
			exit(1);
		}
		proc_data->enst_flux_1d[i] = (double* )fftw_malloc(sizeof(double) * (sys_vars->num_sect));
		if (proc_data->enst_flux_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector 1D");
			exit(1);
		}

		///-------------- Number of Triads and Enstrophy Flux across sectors
		// Allocate memory for the number of triads per sector
		proc_data->num_triads_across_sec[i] = (int** )fftw_malloc(sizeof(int* ) * sys_vars->num_sect);
		if (proc_data->num_triads_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
			exit(1);
		}
		// Allocate memory for the flux of enstrophy across sectors
		proc_data->enst_flux_across_sec[i] = (double** )fftw_malloc(sizeof(double* ) * sys_vars->num_sect);
		if (proc_data->enst_flux_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
			exit(1);
		}	
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			proc_data->num_triads_across_sec[i][l] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k1_sectors);
			if (proc_data->num_triads_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Triads Per Sector");
				exit(1);
			}
			proc_data->enst_flux_across_sec[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
			if (proc_data->enst_flux_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Per Sector");
				exit(1);
			}
		}
	}


	// --------------------------------	
	//  Allocate Mid Angle Sum
	// --------------------------------
	// Allocate memory for the precomputed sector midpoint angle sums -> this is used to determine which sector k2 is
	proc_data->mid_angle_sum = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect * sys_vars->num_k1_sectors);
	if (proc_data->mid_angle_sum == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Midpoint Sector Angles");
		exit(1);
	}


	// -------------------------------------
	//  Allocate Precomputed Vector Angles
	// -------------------------------------
	// Allocate memory for the arctangent arrays
	proc_data->phase_angle = (double* )fftw_malloc(sizeof(double) * Nx * Ny_Fourier);
	if (proc_data->phase_angle == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ArcTangents of Negative k2");
		exit(1);
	}

	// Fill the array for the individual phases with the precomputed arctangents
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny_Fourier; ++j) {
			proc_data->phase_angle[i * Ny_Fourier + j] = atan2((double) run_data->k[0][i], (double)run_data->k[1][j]);
		}
	}


	// -------------------------------------
	//  Allocate Individual Phases Sync
	// -------------------------------------
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


	// -------------------------------------
	//  Allocate Triad Phases Sync
	// -------------------------------------
	// Allocate memory for each  of the triad types
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		//--------------- Allocate memory for the collective phase order parameter for theta
		proc_data->phase_order_C_theta_triads[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
		if (proc_data->phase_order_C_theta_triads[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		proc_data->phase_order_C_theta_triads_1d[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
		if (proc_data->phase_order_C_theta_triads_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter 1D");
			exit(1);
		}

		//--------------- Allocate memory for the phase order parameter for the triad phases
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

		//--------------- Allocate memory for the phase order parameter for the triad phases for 1d contributions
		proc_data->triad_phase_order_1d[i] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_sect);
		if (proc_data->triad_phase_order_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter 1D");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R_1d[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
		if (proc_data->triad_R_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter 1D");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi_1d[i] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_sect);
		if (proc_data->triad_Phi_1d[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase 1D");
			exit(1);
		}

		///------------- Allocate memory for arrays across sectors
		/// Allocate memory for the phase order parameter for the triad phases
		proc_data->triad_phase_order_across_sec[i] = (fftw_complex** )fftw_malloc(sizeof(fftw_complex*) * sys_vars->num_sect);
		if (proc_data->triad_phase_order_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
			exit(1);
		}
		// Allocate the array of phase sync per sector for the triad phases
		proc_data->triad_R_across_sec[i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_sect);
		if (proc_data->triad_R_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
			exit(1);
		}
		// Allocate the array of average phase per sector for the triad phases
		proc_data->triad_Phi_across_sec[i] = (double** )fftw_malloc(sizeof(double*) * sys_vars->num_sect);
		if (proc_data->triad_Phi_across_sec[i] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
			exit(1);
		}
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			/// Allocate memory for the phase order parameter for the triad phases
			proc_data->triad_phase_order_across_sec[i][l] = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->num_k1_sectors);
			if (proc_data->triad_phase_order_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Phase Order Parameter");
				exit(1);
			}
			// Allocate the array of phase sync per sector for the triad phases
			proc_data->triad_R_across_sec[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
			if (proc_data->triad_R_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Triad Phase Sync Parameter");
				exit(1);
			}
			// Allocate the array of average phase per sector for the triad phases
			proc_data->triad_Phi_across_sec[i][l] = (double* )fftw_malloc(sizeof(double) * sys_vars->num_k1_sectors);
			if (proc_data->triad_Phi_across_sec[i][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Average Triad Phase");
				exit(1);
			}
		}
	}

	//--------------- Initialize arrays
	proc_data->dtheta = 2.0 * M_PI / (double )sys_vars->num_sect;
	for (int i = 0; i < sys_vars->num_sect; ++i) {
		proc_data->theta[i] = -M_PI + i * proc_data->dtheta + proc_data->dtheta / 2.0;
		proc_data->phase_R[i]     = 0.0;
		proc_data->phase_Phi[i]   = 0.0;
		proc_data->phase_order[i] = 0.0 + 0.0 * I;
		for (int j = 0; j < NUM_TRIAD_TYPES + 1; ++j) {
			proc_data->num_triads[j][i]           = 0;
			proc_data->enst_flux[j][i]            = 0.0;
			proc_data->triad_R[j][i]              = 0.0;
			proc_data->triad_Phi[j][i]            = 0.0;
			proc_data->num_triads_1d[j][i]        = 0;                        
			proc_data->enst_flux_1d[j][i]         = 0.0;                  
			proc_data->triad_R_1d[j][i]           = 0.0;                  
			proc_data->triad_Phi_1d[j][i]         = 0.0;                    
			proc_data->triad_phase_order[j][i]    = 0.0 + 0.0 * I;
			proc_data->triad_phase_order_1d[j][i] = 0.0 + 0.0 * I;
			proc_data->phase_order_C_theta_triads[j][i] = 0.0 + 0.0 * I;
			for (int k = 0; k < sys_vars->num_k1_sectors; ++k) {
				proc_data->num_triads_across_sec[j][i][k]        = 0;
				proc_data->enst_flux_across_sec[j][i][k]   	  	 = 0.0;
				proc_data->triad_R_across_sec[j][i][k]     	  	 = 0.0;
				proc_data->triad_Phi_across_sec[j][i][k]         = 0.0;
				proc_data->triad_phase_order_across_sec[j][i][k] = 0.0 + 0.0 * I;
			}
		}
	}
	

	// -------------------------------------
	//  Allocate Phases Sync Stats
	// -------------------------------------
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


	// -------------------------------------
	//  Allocate Triad Wavevector Array
	// -------------------------------------
	//------------ Allocate memory for the triad wavevectors per sector and their data 
	proc_data->phase_sync_wave_vecs = (double**** )fftw_malloc(sizeof(double***) * sys_vars->num_sect);
	if (proc_data->phase_sync_wave_vecs == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
		exit(1);
	}
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		proc_data->phase_sync_wave_vecs[a] = (double*** )fftw_malloc(sizeof(double**) * sys_vars->num_k1_sectors);
		if (proc_data->phase_sync_wave_vecs[a] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
			exit(1);
		}	
	}
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
			proc_data->phase_sync_wave_vecs[a][l] = (double** )fftw_malloc(sizeof(double*) * NUM_K_DATA);
			if (proc_data->phase_sync_wave_vecs[a][l] == NULL) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
				exit(1);
			}	
		}
	}
	// Estimate for the number of triads across sectors -> we will resize this dimension to correct size after search is performed
	int num_triad_est               = (int) 3 * ceil(M_PI * pow(sys_vars->N[0], 2.0) + 2.0 * sqrt(2) * M_PI * sys_vars->N[0]);
	sys_vars->num_triad_per_sec_est = num_triad_est;
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
			for (int n = 0; n < NUM_K_DATA; ++n) {
				proc_data->phase_sync_wave_vecs[a][l][n] = (double* )fftw_malloc(sizeof(double) * num_triad_est);
				if (proc_data->phase_sync_wave_vecs[a][l][n] == NULL) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
					exit(1);
				}
			}	
		}
	}

	//------------ Allocate memory for the number of wavevector triads per sector
	proc_data->num_wave_vecs = (int** )fftw_malloc(sizeof(int*) * sys_vars->num_sect);
	if (proc_data->num_wave_vecs == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Phase Sync Wavevectors");
		exit(1);
	}
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		proc_data->num_wave_vecs[a] = (int* )fftw_malloc(sizeof(int) * sys_vars->num_k1_sectors);
		if (proc_data->num_wave_vecs[a] == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Number of Phase Sync Wavevectors");
			exit(1);
		}	
	}
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
			proc_data->num_wave_vecs[a][l] = sys_vars->num_triad_per_sec_est;
		}
	}
	

	//------------------- Fill the wavevector arrays
	int nn;
	double theta_k1;
	int k_x, k_y, k1_x, k1_y, k2_x, k2_y;
	double k1_sqr, k1_angle, k2_sqr, k2_angle, k_sqr, k_angle;
	double k_angle_neg, k1_angle_neg, k2_angle_neg;
	fftw_complex k1, k3;
	double C_theta_lwr, C_theta_upr, C_theta_alt_upr, C_theta_alt_lwr, k2_sect_lwr, k2_sect_upr;
	double k1_sector_angles[NUM_K1_SECTORS] = {-sys_vars->num_sect/2.0, -sys_vars->num_sect/3.0, -sys_vars->num_sect/4.0, -sys_vars->num_sect/6.0, sys_vars->num_sect/6.0, sys_vars->num_sect/4.0, sys_vars->num_sect/3.0, sys_vars->num_sect/2.0};
	proc_data->k1_sector_angles = (double* )fftw_malloc(sizeof(double) * NUM_K1_SECTORS);
	memcpy(proc_data->k1_sector_angles, k1_sector_angles, sizeof(k1_sector_angles));

	// Print to screen that a pre computation search is needed for the phase sync wavevectors and begin timeing it
	printf("\n["YELLOW"NOTE"RESET"] --- Performing search over wavevectors for Phase Sync computation...");
	struct timeval begin, end;
	gettimeofday(&begin, NULL);

	// Loop through the sectors for k3
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		
		// Get the angles for the current sector
		C_theta_lwr = proc_data->theta[a] - proc_data->dtheta / 2.0;
		C_theta_upr = proc_data->theta[a] + proc_data->dtheta / 2.0;

		// Loop through second sector choice
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {

			// Get the angles for the second sector
			if (sys_vars->num_k1_sectors == NUM_K1_SECTORS) {
				theta_k1 = MyMod(proc_data->theta[a] + (k1_sector_angles[l] * proc_data->dtheta/2.0) + M_PI, 2.0 * M_PI) - M_PI;
			}
			else {
				theta_k1 = proc_data->theta[(a + l) % sys_vars->num_sect];
			}
			C_theta_alt_lwr = theta_k1 - proc_data->dtheta / 2.0;
			C_theta_alt_upr = theta_k1 + proc_data->dtheta / 2.0;

			// Find the sector for k2 -> as the mid point of the sector of k3 - k1
			k3 = cexp(I * proc_data->theta[a]);
			k1 = cexp(I * theta_k1);

			// Compute the mid angle for the sector of k2
			if (theta_k1 == proc_data->theta[a]) {
				// Either k1, k2 and k3 are all in the same sector
				proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] = proc_data->theta[a];
			}
			else {
				// Or we find the sector for k2 using arg{(k3 - k1) / (k3^* - k1^*)}
				proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] = creal(1.0 / (2.0 * I) * (clog(k3 - k1) - clog(conj(k3) - conj(k1))));
			}

			// Ensure the angle is a mid angle of a sector
			for (int i = 0; i < sys_vars->num_sect; ++i) {
				if (proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] >= proc_data->theta[i] - proc_data->dtheta/2.0 && proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] < proc_data->theta[i] + proc_data->dtheta/2.0) {
					proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] = proc_data->theta[i];
				}
			}

			// Compute the boundaries for the k2 sector
			k2_sect_lwr = proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] - proc_data->dtheta/2.0;
			k2_sect_upr = proc_data->mid_angle_sum[a * sys_vars->num_k1_sectors + l] + proc_data->dtheta/2.0;
			
			// Initialize increment
			nn = 0;

			// Loop through the k wavevector (k is the k3 wavevector)
			for (int tmp_k_x = 0; tmp_k_x <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k_x) {
				
				// Get k_x
				k_x = tmp_k_x - (int) (sys_vars->N[0] / 2) + 1;

				for (int tmp_k_y = 0; tmp_k_y <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k_y) {
					
					// Get k_y
					k_y = tmp_k_y - (int) (sys_vars->N[0] / 2) + 1;

					// Get polar coords for the k wavevector
					k_sqr       = (double) (k_x * k_x + k_y * k_y);
					k_angle     = atan2((double)k_x, (double)k_y);
					k_angle_neg = atan2((double)-k_x, (double)-k_y);
					
					if ((k_sqr > 0 && k_sqr < sys_vars->kmax_sqr) && ((k_angle >= C_theta_lwr && k_angle < C_theta_upr) || ((k_angle_neg >= C_theta_lwr && k_angle_neg < C_theta_upr)))) {

						// Loop through the k1 wavevector
						for (int tmp_k1_x = 0; tmp_k1_x <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k1_x) {
							
							// Get k1_x
							k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

							for (int tmp_k1_y = 0; tmp_k1_y <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k1_y) {
								
								// Get k1_y
								k1_y = tmp_k1_y - (int) (sys_vars->N[0] / 2) + 1;

								// Get polar coords for k1
								k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
								k1_angle     = atan2((double) k1_x, (double) k1_y);
								k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

								if((k1_sqr > 0 && k1_sqr < sys_vars->kmax_sqr) && ((k1_angle >= C_theta_alt_lwr && k1_angle < C_theta_alt_upr) || (k1_angle_neg >= C_theta_alt_lwr && k1_angle_neg < C_theta_alt_upr))) {									
									
									// Find the k2 wavevector
									k2_x = k_x - k1_x;
									k2_y = k_y - k1_y;
									
									// Get polar coords for k2
									k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
									k2_angle     = atan2((double)k2_x, (double) k2_y);
									k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

									if ((k2_sqr > 0 && k2_sqr < sys_vars->kmax_sqr) && ((k2_angle >= k2_sect_lwr && k2_angle < k2_sect_upr) || (k2_angle_neg >= k2_sect_lwr && k2_angle_neg < k2_sect_upr))) {
										// Add k1 vector
										proc_data->phase_sync_wave_vecs[a][l][K1_X][nn] = k1_x;
										proc_data->phase_sync_wave_vecs[a][l][K1_Y][nn] = k1_y;
										// Add the k2 vector
										proc_data->phase_sync_wave_vecs[a][l][K2_X][nn] = k2_x;
										proc_data->phase_sync_wave_vecs[a][l][K2_Y][nn] = k2_y;
										// Add the k3 vector
										proc_data->phase_sync_wave_vecs[a][l][K3_X][nn] = k_x;
										proc_data->phase_sync_wave_vecs[a][l][K3_Y][nn] = k_y;
										// Add the |k1|^2, |k2|^2, |k3|^2 
										proc_data->phase_sync_wave_vecs[a][l][K1_SQR][nn] = k1_sqr;
										proc_data->phase_sync_wave_vecs[a][l][K2_SQR][nn] = k2_sqr;
										proc_data->phase_sync_wave_vecs[a][l][K3_SQR][nn] = k_sqr;
										// Add the angles for +/- k1, k2, k3
										proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE][nn]     = k1_angle;
										proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE][nn]     = k2_angle;
										proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE][nn]     = k_angle;
										proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE_NEG][nn] = k1_angle_neg;
										proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE_NEG][nn] = k2_angle_neg;
										proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE_NEG][nn] = k_angle_neg;
										
										// Increment
										nn++;
									}
								}
							}
						}
					}
					else if ((k_sqr > 0 && k_sqr < sys_vars->kmax_sqr) && !((k_angle >= C_theta_lwr && k_angle < C_theta_upr) || (k_angle_neg >= C_theta_lwr && k_angle_neg < C_theta_upr)) && ((k_angle >= C_theta_alt_lwr && k_angle < C_theta_alt_upr) || (k_angle_neg >= C_theta_alt_lwr && k_angle_neg < C_theta_alt_upr))) {

						// Loop through the k1 wavevector
						for (int tmp_k1_x = 0; tmp_k1_x <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k1_x) {
							
							// Get k1_x
							k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

							for (int tmp_k1_y = 0; tmp_k1_y <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k1_y) {
								
								// Get k1_y
								k1_y = tmp_k1_y - (int) (sys_vars->N[0] / 2) + 1;

								// Get polar coords for k1
								k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
								k1_angle     = atan2((double) k1_x, (double) k1_y);
								k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

								if((k1_sqr > 0 && k1_sqr < sys_vars->kmax_sqr) && ((k1_angle >= C_theta_lwr && k1_angle < C_theta_lwr) || (k1_angle_neg >= C_theta_lwr && k1_angle_neg < C_theta_lwr))) {									
									
									// Find the k2 wavevector
									k2_x = k_x - k1_x;
									k2_y = k_y - k1_y;
									
									// Get polar coords for k2
									k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
									k2_angle     = atan2((double)k2_x, (double) k2_y);
									k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

									if ((k2_sqr > 0 && k2_sqr < sys_vars->kmax_sqr) && ((k2_angle >= C_theta_lwr && k2_angle < C_theta_upr) || (k2_angle_neg >= C_theta_lwr && k2_angle_neg < C_theta_upr))) {
										// Add k1 vector
										proc_data->phase_sync_wave_vecs[a][l][K1_X][nn] = k1_x;
										proc_data->phase_sync_wave_vecs[a][l][K1_Y][nn] = k1_y;
										// Add the k2 vector
										proc_data->phase_sync_wave_vecs[a][l][K2_X][nn] = k2_x;
										proc_data->phase_sync_wave_vecs[a][l][K2_Y][nn] = k2_y;
										// Add the k3 vector
										proc_data->phase_sync_wave_vecs[a][l][K3_X][nn] = k_x;
										proc_data->phase_sync_wave_vecs[a][l][K3_Y][nn] = k_y;
										// Add the |k1|^2, |k2|^2, |k3|^2 
										proc_data->phase_sync_wave_vecs[a][l][K1_SQR][nn] = k1_sqr;
										proc_data->phase_sync_wave_vecs[a][l][K2_SQR][nn] = k2_sqr;
										proc_data->phase_sync_wave_vecs[a][l][K3_SQR][nn] = k_sqr;
										// Add the angles for +/- k1, k2, k3
										proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE][nn]     = k1_angle;
										proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE][nn]     = k2_angle;
										proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE][nn]     = k_angle;
										proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE_NEG][nn] = k1_angle_neg;
										proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE_NEG][nn] = k2_angle_neg;
										proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE_NEG][nn] = k_angle_neg;
										
										// Increment
										nn++;
									}
								}
							}
						}
					}
					if ((k_sqr > 0 && k_sqr < sys_vars->kmax_C_sqr) && ((k_angle >= C_theta_lwr && k_angle < C_theta_upr) || ((k_angle_neg >= C_theta_lwr && k_angle_neg < C_theta_upr)))) {

						// Loop through the k1 wavevector
						for (int tmp_k1_x = 0; tmp_k1_x <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k1_x) {
							
							// Get k1_x
							k1_x = tmp_k1_x - (int) (sys_vars->N[0] / 2) + 1;

							for (int tmp_k1_y = 0; tmp_k1_y <= 2 * (int) (sys_vars->N[0] / 2) - 1; ++tmp_k1_y) {
								
								// Get k1_y
								k1_y = tmp_k1_y - (int) (sys_vars->N[0] / 2) + 1;

								// Get polar coords for k1
								k1_sqr       = (double) (k1_x * k1_x + k1_y * k1_y);
								k1_angle     = atan2((double) k1_x, (double) k1_y);
								k1_angle_neg = atan2((double)-k1_x, (double)-k1_y);

								if((k1_sqr > 0 && k1_sqr < sys_vars->kmax_sqr) && ((k1_angle >= C_theta_lwr && k1_angle < C_theta_lwr) || (k1_angle_neg >= C_theta_lwr && k1_angle_neg < C_theta_lwr))) {									
									
									// Find the k2 wavevector
									k2_x = k_x - k1_x;
									k2_y = k_y - k1_y;
									
									// Get polar coords for k2
									k2_sqr       = (double) (k2_x * k2_x + k2_y * k2_y);
									k2_angle     = atan2((double)k2_x, (double) k2_y);
									k2_angle_neg = atan2((double)-k2_x, (double) -k2_y);

									if ((k2_sqr > 0 && k2_sqr < sys_vars->kmax_sqr) && ((k2_angle >= C_theta_lwr && k2_angle < C_theta_upr) || (k2_angle_neg >= C_theta_lwr && k2_angle_neg < C_theta_upr))) {
										// Add k1 vector
										proc_data->phase_sync_wave_vecs[a][l][K1_X][nn] = k1_x;
										proc_data->phase_sync_wave_vecs[a][l][K1_Y][nn] = k1_y;
										// Add the k2 vector
										proc_data->phase_sync_wave_vecs[a][l][K2_X][nn] = k2_x;
										proc_data->phase_sync_wave_vecs[a][l][K2_Y][nn] = k2_y;
										// Add the k3 vector
										proc_data->phase_sync_wave_vecs[a][l][K3_X][nn] = k_x;
										proc_data->phase_sync_wave_vecs[a][l][K3_Y][nn] = k_y;
										// Add the |k1|^2, |k2|^2, |k3|^2 
										proc_data->phase_sync_wave_vecs[a][l][K1_SQR][nn] = k1_sqr;
										proc_data->phase_sync_wave_vecs[a][l][K2_SQR][nn] = k2_sqr;
										proc_data->phase_sync_wave_vecs[a][l][K3_SQR][nn] = k_sqr;
										// Add the angles for +/- k1, k2, k3
										proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE][nn]     = k1_angle;
										proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE][nn]     = k2_angle;
										proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE][nn]     = k_angle;
										proc_data->phase_sync_wave_vecs[a][l][K1_ANGLE_NEG][nn] = k1_angle_neg;
										proc_data->phase_sync_wave_vecs[a][l][K2_ANGLE_NEG][nn] = k2_angle_neg;
										proc_data->phase_sync_wave_vecs[a][l][K3_ANGLE_NEG][nn] = k_angle_neg;
										
										// Increment
										nn++;
									}
								}
							}
						}
					}
				}
			}
			// Record the number of triad wavevectors
			proc_data->num_wave_vecs[a][l] = nn;
			// printf("a: %d l: %d Num: %d\tEst: %d\n", a, l, proc_data->num_wave_vecs[a][l], sys_vars->num_triad_per_sec_est);
		}
		// printf("\n");
	}

	//-------------------- Realloc the last dimension in wavevector array to its correct size
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l) {
			for (int n = 0; n < NUM_K_DATA; ++n) {
				if (proc_data->num_wave_vecs[a][l] == 0) {
					// Free empty phase sync wavevector arrays
					fftw_free(proc_data->phase_sync_wave_vecs[a][l][n]);
				}
				else {
					// Otherwise reallocate the correct amount of memory
					proc_data->phase_sync_wave_vecs[a][l][n] = (double* )realloc(proc_data->phase_sync_wave_vecs[a][l][n] , sizeof(double) * proc_data->num_wave_vecs[a][l]);
					if (proc_data->phase_sync_wave_vecs[a][l][n] == NULL) {
						fprintf(stderr, "\n["MAGENTA"WARNING"RESET"] --- Unable to Reallocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Phase Sync Wavevectors");
						exit(1);
					}
				}
			}	
		}
	}


	// Finish timing pre compute step and print to screen
	gettimeofday(&end, NULL);
	// PrintTime(begin.tv_sec, end.tv_sec);
}
/**
 * Frees memory and GSL objects allocated to perform the Phase sync computation
 */
void FreePhaseSyncObjects(void) {

	// --------------------------------
	//  Free Memory
	// --------------------------------
	fftw_free(proc_data->theta);
	fftw_free(proc_data->mid_angle_sum);
	fftw_free(proc_data->phase_angle);
	fftw_free(proc_data->phase_order);
	fftw_free(proc_data->phase_R);
	fftw_free(proc_data->phase_Phi);
	fftw_free(proc_data->k1_sector_angles);
	for (int i = 0; i < NUM_TRIAD_TYPES + 1; ++i) {
		fftw_free(proc_data->enst_flux[i]);
		fftw_free(proc_data->triad_phase_order[i]);
		fftw_free(proc_data->triad_R[i]);
		fftw_free(proc_data->triad_Phi[i]);
		fftw_free(proc_data->num_triads[i]);
		fftw_free(proc_data->enst_flux_1d[i]);
		fftw_free(proc_data->triad_phase_order_1d[i]);
		fftw_free(proc_data->phase_order_C_theta_triads[i]);
		fftw_free(proc_data->phase_order_C_theta_triads_1d[i]);
		fftw_free(proc_data->triad_R_1d[i]);
		fftw_free(proc_data->triad_Phi_1d[i]);
		fftw_free(proc_data->num_triads_1d[i]);
		for (int l = 0; l < sys_vars->num_sect; ++l) {
			fftw_free(proc_data->enst_flux_across_sec[i][l]);
			fftw_free(proc_data->triad_phase_order_across_sec[i][l]);
			fftw_free(proc_data->triad_R_across_sec[i][l]);
			fftw_free(proc_data->triad_Phi_across_sec[i][l]);
			fftw_free(proc_data->num_triads_across_sec[i][l]);
		}
		fftw_free(proc_data->enst_flux_across_sec[i]);
		fftw_free(proc_data->triad_phase_order_across_sec[i]);
		fftw_free(proc_data->triad_R_across_sec[i]);
		fftw_free(proc_data->triad_Phi_across_sec[i]);
		fftw_free(proc_data->num_triads_across_sec[i]);
	}
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		for (int l = 0; l < sys_vars->num_k1_sectors; ++l){
			if (proc_data->num_wave_vecs[a][l] != 0) {
				for (int n = 0; n < NUM_K_DATA; ++n) {
					fftw_free(proc_data->phase_sync_wave_vecs[a][l][n]);
				}	
			}
			fftw_free(proc_data->phase_sync_wave_vecs[a][l]);
		}
		fftw_free(proc_data->phase_sync_wave_vecs[a]);
	}
	for (int a = 0; a < sys_vars->num_sect; ++a) {
		fftw_free(proc_data->num_wave_vecs[a]);
	}

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
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
}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
