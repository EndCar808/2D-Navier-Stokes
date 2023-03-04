#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to plot data to check if turbulence is achieved

#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os
from numba import njit
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import getopt
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, import_sys_msr
from plot_functions import plot_sector_phase_sync_snaps
np.seterr(divide='ignore')


###############################
##       FUNCTION DEFS       ##
###############################
def parse_cml(argv):

    """
    Parses command line arguments
    """

    ## Create arguments class
    class cmd_args:

        """
        Class for command line arguments
        """

        def __init__(self, in_dir = None, out_dir = None, info_dir = None, plotting = False):
            self.in_dir         = in_dir
            self.out_dir_info   = out_dir
            self.in_file        = out_dir
            self.plotting       = plotting
            self.tag = "None"


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:t:", ["plot"])
    except Exception as e:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        print(e)
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:

        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("\nInput Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        if opt in ['-o']:
            ## Read output directory
            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

    return cargs

######################
##       MAIN       ##
######################
if __name__ == '__main__':
	# -------------------------------------
	# # --------- Parse Commnad Line
	# -------------------------------------
	cmdargs = parse_cml(sys.argv[1:])
	method = "default"
	

	# -----------------------------------------
	# # --------  Read In data
	# -----------------------------------------
	## Read in simulation parameters
	sys_vars = sim_data(cmdargs.in_dir)

	## Read in solver data
	run_data = import_data(cmdargs.in_dir, sys_vars, method)

	## Read in system measures
	sys_msr = import_sys_msr(cmdargs.in_dir, sys_vars)

	if not hasattr(sys_msr, 'TimeAveragedEnstrophyFluxSpectrum') or not hasattr(sys_msr, 'TimeAveragedEnstrophyFluxSpectrum'):
		## Read in spectra data
		spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

	# -----------------------------------------
	# # --------  Plot Data
	# -----------------------------------------
	# Make ouotput directory
	out_dir_simdata = cmdargs.out_dir + "SIM_DATA/"
	if os.path.isdir(out_dir_simdata) != True:
		print("Making folder:" + tc.C + " SIM_DATA/" + tc.Rst)
		os.mkdir(out_dir_simdata)

	# Plot Tseries of system measures
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(2, 2)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(sys_msr.time, sys_msr.tot_enrg, label=r"Total Energy")
	ax1.set_xlabel(r"t")
	ax1.set_ylabel(r"$\mathcal{K}$")
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Total Energy")
	ax1 = fig.add_subplot(gs[0, 1])
	ax1.plot(sys_msr.time, sys_msr.tot_enst, label=r"Total Enstrophy")
	ax1.set_xlabel(r"t")
	ax1.set_ylabel(r"$\mathcal{E}$")
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Total Enstrophy")
	ax1 = fig.add_subplot(gs[1, 0])
	ax1.plot(sys_msr.time, sys_msr.enrg_diss, label=r"Total Energy Dissipation")
	ax1.set_xlabel(r"t")
	ax1.set_ylabel(r"$\varepsilon_{\mathbf{u}}$")
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Total Energy")
	ax1 = fig.add_subplot(gs[1, 1])
	ax1.plot(sys_msr.time, sys_msr.enst_diss, label=r"Total Enstrophy Dissipation")
	ax1.set_xlabel(r"t")
	ax1.set_ylabel(r"$\varepsilon_{\omega}$")
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Total Enstrophy")
	plt.savefig(out_dir_simdata + "SysMsr_Tseries.png", bbox_inches='tight')
	plt.close()


	# Plot Forcing Injection
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(2, 2)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(sys_msr.time, sys_msr.tot_forc, label=r"$\omega f$")
	ax1.set_xlabel(r"t")
	ax1.set_ylabel(r"Total Forcing")
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Total Forcing Input")
	plt.savefig(out_dir_simdata + "SysMsr_Forcing_Input.png", bbox_inches='tight')
	plt.close()

	# Plot time averaged spectra
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0, 0])
	non_zero_args = np.where(sys_msr.enrg_spect_t_avg != 0)
	enrg_spect_t_avg = sys_msr.enrg_spect_t_avg[non_zero_args]
	k_range = np.arange(1, len(enrg_spect_t_avg) + 1)
	ax1.plot(k_range, enrg_spect_t_avg)
	ax1.set_xlabel(r"k")
	ax1.set_ylabel(r"$\mathcal{K}_k$")
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Time Averaged Energy Spectrum")
	ax1 = fig.add_subplot(gs[0, 1])
	enst_spect_t_avg = sys_msr.enst_spect_t_avg[non_zero_args]
	k_range = np.arange(1, len(enst_spect_t_avg) + 1)
	ax1.plot(k_range, enst_spect_t_avg)
	ax1.set_xlabel(r"k")
	ax1.set_ylabel(r"$\mathcal{E}_k$")
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Time Averaged Enstrophy Spectrum")
	plt.savefig(out_dir_simdata + "TimeAvg_Spectra.png", bbox_inches='tight')
	plt.close()


	if not hasattr(sys_msr, 'TimeAveragedEnstrophyFluxSpectrum') or not hasattr(sys_msr, 'TimeAveragedEnstrophyFluxSpectrum'):
		enrg_flux_spect_t_avg = np.mean(spec_data.enrg_flux_spectrum, axis=0)
		enst_flux_spect_t_avg = np.mean(spec_data.enst_flux_spectrum, axis=0)
	else:
		enrg_flux_spect_t_avg = sys_msr.enrg_flux_spect_t_avg
		enst_flux_spect_t_avg = sys_msr.enst_flux_spect_t_avg

	# Plot time averaged flux spectra
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0, 0])
	non_zero_args = np.where(enrg_flux_spect_t_avg != 0)
	enrg_flux_spect_t_avg = enrg_flux_spect_t_avg[non_zero_args]
	k_range = np.arange(1, len(enrg_flux_spect_t_avg) + 1)
	ax1.plot(k_range, enrg_flux_spect_t_avg)
	ax1.set_xlabel(r"k")
	ax1.set_ylabel(r"$\Pi^{\mathbf{u}}$")
	ax1.set_xscale('log')
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Time Averaged Energy Flux Spectrum")
	ax1 = fig.add_subplot(gs[0, 1])
	non_zero_args = np.where(enst_flux_spect_t_avg != 0)
	enst_flux_spect_t_avg = enst_flux_spect_t_avg[non_zero_args]
	k_range = np.arange(1, len(enst_flux_spect_t_avg) + 1)
	ax1.plot(k_range, enst_flux_spect_t_avg)
	ax1.set_xlabel(r"k")
	ax1.set_ylabel(r"$\Pi^{\omega}$")
	ax1.set_xscale('log')
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax1.set_title(r"Time Averaged Enstrophy Flux Spectrum")

	plt.savefig(out_dir_simdata + "TimeAvg_Flux_Spectra.png", bbox_inches='tight')
	plt.close()