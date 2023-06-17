#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to plot the weighted PDFs of triad phase types

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
from itertools import zip_longest
import multiprocessing as mprocs
import time as TIME
from subprocess import Popen, PIPE
from matplotlib.pyplot import cm
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, get_flux_spectrum, compute_pdf
from plot_functions import plot_sector_phase_sync_snaps


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

        def __init__(self, in_dir = None, out_dir = None, in_file = None, incr = False, wghtd = False):
            self.in_dir         = in_dir
            self.out_dir_phases = out_dir
            self.out_dir_triads = out_dir
            self.in_file        = out_dir
            self.incr           = incr
            self.wghtd          = wghtd

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:", ["incr", "wghtd"])
    except:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:

        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("Input Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        if opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

        elif opt in ['--incr']:
            ## Read in plotting indicator
            cargs.incr = True

        elif opt in ['--wghtd']:
            ## If phases are to be plotted
            cargs.wght = True


    return cargs

######################
##       MAIN       ##
######################
if __name__ == '__main__':

	# -------------------------------------
	# # --------- Parse Commnad Line
	# -------------------------------------
	cmdargs = parse_cml(sys.argv[1:])
	if cmdargs.in_file is None:
	    method = "default"
	    post_file_path = cmdargs.in_dir
	else: 
	    method = "file"
	    post_file_path = cmdargs.in_dir + cmdargs.in_file

	# -----------------------------------------
	# # --------  Make Output Folder
	# -----------------------------------------
	cmdargs.out_dir_stats = cmdargs.out_dir + "STATS_PLOTS/"
	if os.path.isdir(cmdargs.out_dir_stats) != True:
	    print("Making folder:" + tc.C + " STATS_PLOTS/" + tc.Rst)
	    os.mkdir(cmdargs.out_dir_stats)
	snaps_output_dir = cmdargs.out_dir + "RUN_SNAPS/"
	if os.path.isdir(snaps_output_dir) != True:
	    print("Making folder:" + tc.C + " RUN_SNAPS/" + tc.Rst)
	    os.mkdir(snaps_output_dir)


	# -----------------------------------------
	# # --------  Read In data
	# -----------------------------------------
	## Read in simulation parameters
	sys_vars = sim_data(cmdargs.in_dir)

	## Read in solver data
	run_data = import_data(cmdargs.in_dir, sys_vars)

	## Read in spectra data
	spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

	## Read in post processing data
	post_data = import_post_processing_data(post_file_path, sys_vars, method)

	# -----------------------------------------
	# # --------  Make Output Folder
	# -----------------------------------------
	inert_range    = np.arange(4, 15)
	k_range        = np.arange(1, int(sys_vars.Nx/3))
	mean_enrg_spec = np.mean(spec_data.enrg_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
	mean_enst_spec = np.mean(spec_data.enst_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
	p_enrg         = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enrg_spec[inert_range]), 1)
	p_enst         = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enst_spec[inert_range]), 1)

	# enst_flux_spectrum = spec_data.enst_flux_spectrum[:, :]
	# flux_spect = get_flux_spectrum(enst_flux_spectrum[:, 1:int(sys_vars.Nx/3)])

	fig = plt.figure(figsize = (21, 8))
	gs  = GridSpec(1, 2)
	ax2 = fig.add_subplot(gs[0, 0])
	ax2.plot(k_range, mean_enrg_spec, 'k')
	ax2.plot(k_range[inert_range], np.exp(p_enrg[1]) * k_range[inert_range]**p_enrg[0], '--', color='orangered',label="$E(k) \propto k^{:.2f}$".format(p_enrg[0])) ## \Rightarrow \propto$ k^{-(3 + \qi)} \Rightarrow \qi = {:.2f} , np.absolute(np.absolute(p_enrg[0]) - 3))
	ax2.set_xlabel(r"$k$")
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.legend()
	ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax2.set_title(r"$\mathcal{K}(|\mathbf{k}|)$: Energy Spectrum")
	print("Energy Spectrum Slope - 3: {}".format(np.absolute(np.absolute(p_enrg[0]) - 3)))
	zeta_2_theory = np.absolute(np.absolute(p_enrg[0]) - 3)

	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(k_range, mean_enst_spec, 'k')
	ax2.plot(k_range[inert_range], np.exp(p_enst[1]) * k_range[inert_range]**p_enst[0], '--', color='orangered',label="$E(k) \propto k^{:.2f}$".format(p_enst[0])) ## \propto$ k^{-(1 + \qi)} \Rightarrow , \qi = {:.2f} np.absolute(np.absolute(p_enst[0]) - 3))
	ax2.set_xlabel(r"$k$")
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.legend()
	ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax2.set_title(r"$\mathcal{E}(|\mathbf{k}|)$: Enstrophy Spectrum")
	print("Enstrophy Spectrum Slope - 1: {}".format(np.absolute(np.absolute(p_enst[0]) - 1)))

	plt.savefig(snaps_output_dir + "SpectraScaling.png")
	plt.close()

	# -----------------------------------------
	# # --------  Plot Longitudinal Increment PDFs
	# -----------------------------------------
	## Create Figure
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 2) 

	if post_data.vel_long_incr_ranges.shape[0] == 2:
	    plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \pi$"]
	else:
	    plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \frac{2\pi}{N}$", r"$r = \frac{4\pi}{N}$", r"$r = \frac{16\pi}{N}$",  r"$r = \pi$"]

	## Longitudinal PDFs
	ax1 = fig.add_subplot(gs[0, 0])
	for i in range(post_data.vel_long_incr_ranges.shape[0]):
	    bin_centres, pdf = compute_pdf(post_data.vel_long_incr_ranges[i, :], post_data.vel_long_incr_counts[i, :], normalized = True)
	    ax1.plot(bin_centres, pdf, label = plot_legend[i])
	ax1.set_xlabel(r"$\delta_r \mathbf{u}_{\parallel} / \langle (\delta_r \mathbf{u}_{\parallel})^2 \rangle^{1/2}$")
	ax1.set_ylabel(r"PDF")
	ax1.set_yscale('log')
	ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax1.set_title("Velocity Longitudinal Increments")
	ax1.legend()

	## Transverse PDFs
	ax2 = fig.add_subplot(gs[0, 1])
	for i in range(post_data.vort_long_incr_ranges.shape[0]):
	    bin_centres, pdf = compute_pdf(post_data.vort_long_incr_ranges[i, :], post_data.vort_long_incr_counts[i, :], normalized = True)        
	    ax2.plot(bin_centres, pdf, label = plot_legend[i])
	ax2.set_xlabel(r"$\delta_r \omega_{\parallel} / \langle (\delta_r \omega_{\parallel})^2 \rangle^{1/2}$")
	ax2.set_ylabel(r"PDF")
	ax2.set_yscale('log')
	ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax2.set_title("Vorticity Longitudinal Incrments")
	ax2.legend()

	plt.suptitle(r"Longitudinal Increment PDFs")

	plt.savefig(cmdargs.out_dir_stats + "/Longitudinal_Incrmenents_PDFs.png")
	plt.close()

	# -----------------------------------------
	# # --------  Plot Transverse Increment PDFs
	# -----------------------------------------
	## Create Figure
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 2) 

	if post_data.vel_trans_incr_ranges.shape[0] == 2:
	    plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \pi$"]
	else:
	    plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \frac{2\pi}{N}$", r"$r = \frac{4\pi}{N}$", r"$r = \frac{16\pi}{N}$",  r"$r = \pi$"]

	## Transverse PDFs
	ax1 = fig.add_subplot(gs[0, 0])
	for i in range(post_data.vel_trans_incr_ranges.shape[0]):
	    bin_centres, pdf = compute_pdf(post_data.vel_trans_incr_ranges[i, :], post_data.vel_trans_incr_counts[i, :], normalized = True)
	    ax1.plot(bin_centres, pdf, label = plot_legend[i])
	ax1.set_xlabel(r"$\delta_r \mathbf{u}_{\perp} / \langle (\delta_r \mathbf{u}_{\perp})^2 \rangle^{1/2}$")
	ax1.set_ylabel(r"PDF")
	ax1.set_yscale('log')
	ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax1.set_title("Velocity Transverse Increments")
	ax1.legend()

	## Transverse PDFs
	ax2 = fig.add_subplot(gs[0, 1])
	for i in range(post_data.vort_trans_incr_ranges.shape[0]):
	    bin_centres, pdf = compute_pdf(post_data.vort_trans_incr_ranges[i, :], post_data.vort_trans_incr_counts[i, :], normalized = True)        
	    ax2.plot(bin_centres, pdf, label = plot_legend[i])
	ax2.set_xlabel(r"$\delta_r \omega_{\perp} / \langle (\delta_r \omega_{\perp})^2 \rangle^{1/2}$")
	ax2.set_ylabel(r"PDF")
	ax2.set_yscale('log')
	ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax2.set_title("Vorticity Transverse Incrments")
	ax2.legend()

	plt.suptitle(r"Transverse Increment PDFs")

	plt.savefig(cmdargs.out_dir_stats + "/Transverse_Incrmenents_PDFs.png")
	plt.close()

	# -----------------------------------------
	# # --------  Plot Velocity Increment PDFs
	# -----------------------------------------
	## Create Figure
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 3) 

	## Longitudinal PDFs
	ax1 = fig.add_subplot(gs[0, 0])
	for i in [0, -1]:
	    bin_centres, pdf = compute_pdf(post_data.vel_long_incr_ranges[i, :], post_data.vel_long_incr_counts[i, :], normalized = True)
	    ax1.plot(bin_centres, pdf)
	bin_centres, pdf = compute_pdf(post_data.grad_u_x_ranges[:], post_data.grad_u_x_counts[:], normalized = True)
	ax1.plot(bin_centres, pdf)
	ax1.set_xlabel(r"$\delta_r \mathbf{u}_{\parallel} / \langle (\delta_r \mathbf{u}_{\parallel})^2 \rangle^{1/2}$")
	ax1.set_ylabel(r"PDF")
	ax1.set_yscale('log')
	ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax1.set_title("Longitudinal Increments")
	ax1.legend([r"$r = \frac{\pi}{N}$", r"$r = \pi$", r"$\partial \mathbf{u}/\partial x$"])

	## Transverse PDFs
	ax2 = fig.add_subplot(gs[0, 1])
	for i in [0, -1]:
	    bin_centres, pdf = compute_pdf(post_data.vel_trans_incr_ranges[i, :], post_data.vel_trans_incr_counts[i, :], normalized = True)        
	    ax2.plot(bin_centres, pdf)
	bin_centres, pdf = compute_pdf(post_data.grad_u_y_ranges[:], post_data.grad_u_y_counts[:], normalized = True)
	ax2.plot(bin_centres, pdf)
	ax2.set_xlabel(r"$\delta_r \mathbf{u}_{\perp} / \langle (\delta_r \mathbf{u}_{\perp})^2 \rangle^{1/2}$")
	ax2.set_ylabel(r"PDF")
	ax2.set_yscale('log')
	ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax2.set_title("Transverse Incrments")
	ax2.legend([r"$r = \frac{\pi}{N}$", r"$r = \pi$", r"$\partial \mathbf{u}/\partial y$"])

	ax3 = fig.add_subplot(gs[0, 2])
	bin_centres, pdf = compute_pdf(post_data.vel_long_incr_ranges[0, :], post_data.vel_long_incr_counts[0, :], normalized = True)
	p, = ax3.plot(bin_centres, pdf, '-.', label = r"Long; $r = \frac{\pi}{N}$")
	bin_centres, pdf = compute_pdf(post_data.vel_trans_incr_ranges[0, :], post_data.vel_trans_incr_counts[0, :], normalized = True)
	ax3.plot(bin_centres, pdf, color = p.get_color(), label = r"Trans; $r = \frac{\pi}{N}$")
	bin_centres, pdf = compute_pdf(post_data.vel_long_incr_ranges[-1, :], post_data.vel_long_incr_counts[-1, :], normalized = True)
	p, = ax3.plot(bin_centres, pdf, '-.', label = r"Long; $r = \pi$")
	bin_centres, pdf = compute_pdf(post_data.vel_trans_incr_ranges[-1, :], post_data.vel_trans_incr_counts[-1, :], normalized = True)
	ax3.plot(bin_centres, pdf, color = p.get_color(), label = r"Trans; $r = \pi$")
	ax3.set_xlabel(r"$\delta_r \mathbf{u} / \langle (\delta_r \mathbf{u})^2 \rangle^{1/2}$")
	ax3.set_ylabel(r"PDF")
	ax3.set_yscale('log')
	ax3.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax3.set_title("Both Incrments")
	ax3.legend()

	plt.suptitle(r"Velocity Increment PDFs")

	plt.savefig(cmdargs.out_dir_stats + "/Velocity_Incrmenent_PDFs.png")
	plt.close()


	# -----------------------------------------
	# # --------  Plot Vorticity Increment PDFs
	# -----------------------------------------
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 3) 
	## Longitudinal PDFs
	ax1 = fig.add_subplot(gs[0, 0])
	for i in [0, -1]:
	    bin_centres, pdf = compute_pdf(post_data.vort_long_incr_ranges[i, :], post_data.vort_long_incr_counts[i, :], normalized = True)
	    ax1.plot(bin_centres, pdf)
	bin_centres, pdf = compute_pdf(post_data.grad_w_x_ranges[:], post_data.grad_w_x_counts[:], normalized = True)
	ax1.plot(bin_centres, pdf)
	ax1.set_xlabel(r"$\delta_r w_{\parallel} / \langle (\delta_r w_{\parallel})^2 \rangle^{1/2}$")
	ax1.set_ylabel(r"PDF")
	ax1.set_yscale('log')
	ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax1.set_title("Longitudinal Increments")
	ax1.legend([r"$r = \frac{\pi}{N}$", r"$r = \pi$", r"$\partial \omega /\partial x$"])

	## Transverse PDFs
	ax2 = fig.add_subplot(gs[0, 1])
	for i in [0, -1]:
	    bin_centres, pdf = compute_pdf(post_data.vort_trans_incr_ranges[i, :], post_data.vort_trans_incr_counts[i, :], normalized = True)
	    ax2.plot(bin_centres, pdf)
	bin_centres, pdf = compute_pdf(post_data.grad_w_y_ranges[:], post_data.grad_w_y_counts[:], normalized = True)
	ax2.plot(bin_centres, pdf)
	ax2.set_xlabel(r"$\delta_r w_{\perp} / \langle (\delta_r w_{\perp})^2 \rangle^{1/2}$")
	ax2.set_ylabel(r"PDF")
	ax2.set_yscale('log')
	ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax2.set_title("Transverse Incrments")
	ax2.legend([r"$r = \frac{\pi}{N}$", r"$r = \pi$", r"$\partial \omega /\partial y$"])

	## Both together
	ax3 = fig.add_subplot(gs[0, 2])
	bin_centres, pdf = compute_pdf(post_data.vort_long_incr_ranges[0, :], post_data.vort_long_incr_counts[0, :], normalized = True)
	p, = ax3.plot(bin_centres, pdf, label = r"Long; $r = \frac{\pi}{N}$")
	bin_centres, pdf = compute_pdf(post_data.vort_trans_incr_ranges[0, :], post_data.vort_trans_incr_counts[0, :], normalized = True)
	ax3.plot(bin_centres, pdf, '-.', color = p.get_color(), label = r"Trans; $r = \frac{\pi}{N}$")
	bin_centres, pdf = compute_pdf(post_data.vort_long_incr_ranges[-1, :], post_data.vort_long_incr_counts[-1, :], normalized = True)
	p, = ax3.plot(bin_centres, pdf, label = r"Long; $r = \pi$")
	bin_centres, pdf = compute_pdf(post_data.vort_trans_incr_ranges[-1, :], post_data.vort_trans_incr_counts[-1, :], normalized = True)
	ax3.plot(bin_centres, pdf, '-.', color = p.get_color(), label = r"Trans; $r = \pi$")
	ax3.set_xlabel(r"$\delta_r w / \langle (\delta_r w)^2 \rangle^{1/2}$")
	ax3.set_ylabel(r"PDF")
	ax3.set_yscale('log')
	ax3.grid(color = 'k', linewidth = .5, linestyle = ':')
	ax3.set_title("Both Incrments")
	ax3.legend()

	plt.suptitle(r"Vorticity Increment PDFs")

	plt.savefig(cmdargs.out_dir_stats + "/Vorticity_Incrment_PDFs.png")
	plt.close()