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

# @njit
def compute_pdf(bin_ranges, bin_counts, normalized = False):

    ## Get nonzero bin indexs
    non_zero_args = np.where(bin_counts != 0)

    ## Get the bin centres
    bin_centres = (bin_ranges[1:] + bin_ranges[:-1]) * 0.5
    bin_centres = bin_centres[non_zero_args]

    ## Compute the bin width
    bin_width = bin_ranges[1] - bin_ranges[0]

    ## Compute the pdf
    bin_counts = bin_counts[non_zero_args]
    pdf = bin_counts / (np.sum(bin_counts) * bin_width)

    if normalized:
        var         = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        pdf         *= var
        bin_centres /= var 
        bin_width   /= var

    return bin_centres, pdf

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
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    ## Read in system measures
    sys_msr = import_sys_msr(cmdargs.in_dir, sys_vars)

    ## Read in post processing data
    post_data = import_post_processing_data(post_file_path, sys_vars, method)
    
    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

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
    # # --------  Time Averaged Spectra to Measure the Spectra Scaling Exponent
    # ----------------------------------------- 
    inert_range = np.arange(4, 25)
    k_range = np.arange(1, int(sys_vars.Nx/3))
    mean_enrg_spec = np.mean(spec_data.enrg_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
    mean_enst_spec = np.mean(spec_data.enst_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
    p_enrg = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enrg_spec[inert_range]), 1)
    p_enst = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enst_spec[inert_range]), 1)

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


    # -----------------------------------------
    # # --------  Compare Structure Funcs
    # -----------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    r = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)
    L = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2

    ## Velocity
    ax1 = fig.add_subplot(gs[0, 0])
    p, = ax1.plot(r / L, 3.0 / 2.0 * (r / L), linestyle = '--', label = r"$\frac{3}{2} \epsilon_l r$")
    ax1.plot(r / L, post_data.mxd_vel_str_func[:], linestyle = '-.', color = p.get_color(), label = r"Mixed; $3\left\langle\delta u_{\parallel}(r)\left[\delta u_{\perp}(r)\right]^2\right\rangle$")
    ax1.plot(r / L, post_data.vel_long_str_func[2, :], linestyle = ':', color = p.get_color(), label = r"Third Order; $\left\langle\left[\delta u_{\|}(r)\right]^3\right\rangle$")
    ax1.set_xlabel(r"$r / L$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax1.set_title("Holds For Inverse Cascade")
    ax1.legend()

    ## Vorticity
    ax2 = fig.add_subplot(gs[0, 1])
    p, = ax2.plot(r / L, -2.0 * r / L, linestyle = '--', label = r"$-2 \eta_I r$")
    ax2.plot(r / L, post_data.mxd_vel_str_func[:], linestyle = '-.', color = p.get_color(), label = r"Mixed; $\left\langle\delta u_{\|}(r)[\delta \omega(r)]^2\right\rangle$")
    p, = ax2.plot(r / L, 1.0 / 8.0 * (r / L) ** 3, linestyle = '--', label = r"$\frac{1}{8} \eta_I r^3$")
    ax2.plot(r / L, post_data.vel_long_str_func[2, :], '-.', color = p.get_color(), label = r"Third Order; $\left\langle\left[\delta u_{\|}(r)\right]^3\right\rangle$")
    ax2.set_xlabel(r"$r / L$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax2.set_title("Holds For Direct Cascade")
    ax2.legend()

    plt.suptitle(r"Compare Structure Funcitons")
    
    plt.savefig(cmdargs.out_dir_stats + "/Compare_Structure_Functions.png")
    plt.close()



    # ---------------------------------------------
    # # --------  Plot Velocity Structure Functions
    # ---------------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2, hspace = 0.3) 
    r = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)
    L = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
    powers = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax1 = fig.add_subplot(gs[0, 0])
    print(post_data.vel_long_str_func[0, :])
    for i in range(post_data.vel_long_str_func.shape[0]):
        # ax1.plot(r / L, np.absolute(post_data.vel_long_str_func[i, :]))
        ax1.plot(np.log2(r), np.log2(np.absolute(post_data.vel_long_str_func[i, :])))
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$|S^p(r)|$")
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax1.set_title(r"Longitudinal Structure Functions")
    ax1.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vel_long_str_func.shape[0] + 1)

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(post_data.vel_trans_str_func.shape[0]):
        # ax2.plot(r, np.absolute(post_data.vel_trans_str_func[i, :]))
        ax2.plot(np.log2(r), np.log2(np.absolute(post_data.vel_trans_str_func[i, :])))
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|S^p(r)|$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax2.set_title(r"Transverse Structure Functions")
    ax2.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vel_trans_str_func.shape[0] + 1)

    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(post_data.vel_long_str_func_abs.shape[0]):
        # ax3.plot(r, np.absolute(post_data.vel_long_str_func_abs[i, :]))
        ax3.plot(np.log2(r), np.log2(np.absolute(post_data.vel_long_str_func_abs[i, :])))
    ax3.set_xlabel(r"$r$")
    ax3.set_ylabel(r"$|S^p_{abs}(r)|$")
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')
    ax3.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax3.set_title(r"Absolute Longitudinal Structure Functions")
    ax3.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vel_long_str_func_abs.shape[0] + 1)

    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(post_data.vel_trans_str_func_abs.shape[0]):
        # ax4.plot(r, np.absolute(post_data.vel_trans_str_func_abs[i, :]))
        ax4.plot(np.log2(r), np.log2(np.absolute(post_data.vel_trans_str_func_abs[i, :])))
    ax4.set_xlabel(r"$r$")
    ax4.set_ylabel(r"$|S^p_{abs}(r)|$")
    # ax4.set_xscale('log')
    # ax4.set_yscale('log')
    ax4.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax4.set_title(r"Absolute Transverse Structure Functions")
    ax4.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vel_trans_str_func_abs.shape[0] + 1)

    plt.suptitle(r"Veloocity Structure Functions")
    
    plt.savefig(cmdargs.out_dir_stats + "/Velocity_Structure_Functions.png")
    plt.close()



    # ---------------------------------------------
    # # --------  Plot Vorticity Structure Functions
    # ---------------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2, hspace = 0.3) 
    r = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)
    # L = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
    powers = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax1 = fig.add_subplot(gs[0, 0])
    print(post_data.vort_long_str_func[0, :])
    for i in range(post_data.vort_long_str_func.shape[0]):
        # ax1.plot(r / , np.absolute(post_data.vort_long_str_func[i, :]))
        ax1.plot(np.log2(r), np.log2(np.absolute(post_data.vort_long_str_func[i, :])))
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$|S^p(r)|$")
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax1.set_title(r"Longitudinal Structure Functions")
    ax1.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vort_long_str_func.shape[0] + 1)

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(post_data.vort_trans_str_func.shape[0]):
        # ax2.plot(r / L, np.absolute(post_data.vort_trans_str_func[i, :]))
        ax2.plot(np.log2(r) , np.log2(np.absolute(post_data.vort_trans_str_func[i, :])))
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|S^p(r)|$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax2.set_title(r"Transverse Structure Functions")
    ax2.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vort_trans_str_func.shape[0] + 1)

    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(post_data.vort_long_str_func_abs.shape[0]):
        # ax3.plot(r / L, np.absolute(post_data.vort_long_str_func_abs[i, :]))
        ax3.plot(np.log2(r) , np.log2(np.absolute(post_data.vort_long_str_func_abs[i, :])))
    ax3.set_xlabel(r"$r$")
    ax3.set_ylabel(r"$|S^p_{abs}(r)|$")
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')
    ax3.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax3.set_title(r"Absolute Longitudinal Structure Functions")
    ax3.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vort_long_str_func_abs.shape[0] + 1)

    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(post_data.vort_trans_str_func_abs.shape[0]):
        # ax4.plot(r / L, np.absolute(post_data.vort_trans_str_func_abs[i, :]))
        ax4.plot(np.log2(r) , np.log2(np.absolute(post_data.vort_trans_str_func_abs[i, :])))
    ax4.set_xlabel(r"$r$")
    ax4.set_ylabel(r"$|S^p_{abs}(r)|$")
    # ax4.set_xscale('log')
    # ax4.set_yscale('log')
    ax4.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax4.set_title(r"Absolute Transverse Structure Functions")
    ax4.legend([r"$p = {}$".format(p) for p in powers]) ## range(1, post_data.vort_trans_str_func_abs.shape[0] + 1)

    plt.suptitle(r"Vorticity Structure Functions")
    
    plt.savefig(cmdargs.out_dir_stats + "/Vorticity_Structure_Functions.png")
    plt.close()



    # -----------------------------------------------------------------------------
    # # --------  Plot Vorticity Structure Functions w/ Fit & Anomonlous Exponent
    # -----------------------------------------------------------------------------
    indx_shift = 1

    inert_lim_low  = inert_range[0]
    inert_lim_high = inert_range[-1]

    mark_style = ['o','s','^','x','D','p']

    zeta_p_long = []
    powers = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5])
    r      = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)

    x0     = 0.85 
    y0     = 0.85
    width  = 0.3
    height = 0.2
    fig   = plt.figure(figsize = (10, 10))
    gs    = GridSpec(2, 2, hspace = 0.3)

    ax1   = fig.add_subplot(gs[0, 0])
    ## Add insert
    # ax1in = fig.add_axes([x0, y0, width, height])
    for i in range(post_data.vort_long_str_func_abs.shape[0]):
        ## Plot strucure function
        p, = ax1.plot(np.log2(r), np.log2(post_data.vort_long_str_func_abs[i, :]), label = "$p = {}$".format(powers[i])) # marker = mark_style[i], markerfacecolor = 'None', markersize = 5.0, markevery = 2**4
        ## Find polynomial fit and plot
        pfit_info  = np.polyfit(np.log2(r[inert_lim_low:inert_lim_high]), np.log2(post_data.vort_long_str_func_abs[i, inert_lim_low:inert_lim_high]), 1)
        pfit_slope = pfit_info[0]
        pfit_c     = pfit_info[1]
        zeta_p_long.append(np.absolute(pfit_slope))
        print(i, pfit_slope, pfit_c)
        ax1.plot(np.log2(r[inert_lim_low:inert_lim_high]), np.log2(r[inert_lim_low:inert_lim_high])*pfit_slope + pfit_c + 0.25, '--', color = p.get_color())
        ## Compute the local derivative and plot in insert
        # d_str_func  = np.diff(np.log2(post_data.vort_long_str_func_abs[i, :]))
        # d_r         = np.diff(np.log2(r))
        # local_deriv = d_str_func / d_r
        # local_deriv = np.concatenate((local_deriv, [(np.log2(post_data.vort_long_str_func_abs[-1, i]) - np.log2(post_data.vort_long_str_func_abs[-2, i])) / (np.log2(r[-1] - np.log2(r[-2])))]))
        # ax1in.plot(np.log2(r), local_deriv, color = p.get_color(), marker = mark_style[i], markerfacecolor = 'None', markersize = 5.0, markevery = 2)
        # ax1in.set_ylabel(r"$\zeta_p$", labelpad = -40)
        # ax1in.set_xlabel(r"$log2(r)$", labelpad = -30)
    ax1.set_xlabel(r"$log_2 (r)$")
    ax1.set_ylabel(r"$log_2 (S_{2p}(r))$")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax1in.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax1.set_title(r"Longitudinal Structure Function")
    ax1.legend()

    # # --------  Plot Anomalous Exponent
    ax2   = fig.add_subplot(gs[1, 0])
    p = powers
    # zeta_2_theory = 
    # if hasattr(sys_vars, "alpha_high_k"):
    #     if sys_vars.alpha_high_k == 0.1:
    #         zeta_2_theory = 0.63
    #     elif sys_vars.alpha_high_k == 0.2:
    #         zeta_2_theory = 1.10
    #     else:
    #         zeta_2_theory = 1.0
    print("Long: {}".format(zeta_p_long[2]), np.array(zeta_p_long[:]) / zeta_2_theory, zeta_2_theory)
    ns_zeta_p = [0.72, 1, 1.273, 1.534, 1.786]
    ax2.plot(p, np.array(zeta_p_long[:]) / zeta_2_theory, marker = mark_style[0], markerfacecolor = 'None', markersize = 5.0, markevery = 1, label = "DNS")
    ax2.plot(p, np.array(zeta_p_long[:]) / zeta_p_long[2], marker = mark_style[0], markerfacecolor = 'None', markersize = 5.0, markevery = 1, label = "DNS / $\zeta_2,DNS$")
    ax2.plot(p, p, 'k--', label = "K41")
    ax2.set_xlabel(r"$p$")
    ax2.set_ylabel(r"$\zeta_{2p} / \zeta_{2, th}$")
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"Longitudinal $\zeta_{2p}$")
    ax2.legend()

    zeta_p_trans = []
    ax3   = fig.add_subplot(gs[0, 1])
    ## Add insert
    # ax3in = fig.add_axes([x0, y0, width, height])
    for i in range(post_data.vort_trans_str_func_abs.shape[0]):
        ## Plot strucure function
        p, = ax3.plot(np.log2(r), np.log2(post_data.vort_trans_str_func_abs[i, :]), label = "$p = {}$".format(powers[i])) # marker = mark_style[i], markerfacecolor = 'None', markersize = 5.0, markevery = 2**4
        ## Find polynomial fit and plot
        pfit_info  = np.polyfit(np.log2(r[inert_lim_low:inert_lim_high]), np.log2(post_data.vort_trans_str_func_abs[i, inert_lim_low:inert_lim_high]), 1)
        pfit_slope = pfit_info[0]
        pfit_c     = pfit_info[1]
        zeta_p_trans.append(np.absolute(pfit_slope))
        print(i, pfit_slope, pfit_c)
        ax3.plot(np.log2(r[inert_lim_low:inert_lim_high]), np.log2(r[inert_lim_low:inert_lim_high])*pfit_slope + pfit_c + 0.25, '--', color = p.get_color())
        ## Compute the local derivative and plot in insert
        # d_str_func  = np.diff(np.log2(post_data.vort_trans_str_func_abs[i, :]))
        # d_r         = np.diff(np.log2(r))
        # local_deriv = d_str_func / d_r
        # local_deriv = np.concatenate((local_deriv, [(np.log2(post_data.vort_trans_str_func_abs[-1, i]) - np.log2(post_data.vort_trans_str_func_abs[-2, i])) / (np.log2(r[-1] - np.log2(r[-2])))]))
        # ax3in.plot(np.log2(r), local_deriv, color = p.get_color(), marker = mark_style[i], markerfacecolor = 'None', markersize = 5.0, markevery = 2)
        # ax3in.set_ylabel(r"$\zeta_p$", labelpad = -40)
        # ax3in.set_xlabel(r"$log2(r)$", labelpad = -30)
    ax3.set_xlabel(r"$log_2 (r)$")
    ax3.set_ylabel(r"$log_2 (S_{2p}(r))$")
    ax3.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax3in.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax3.set_title(r"Transverse $\zeta_{2p}$")
    ax3.legend()

    # # --------  Plot Anomalous Exponent
    ax4   = fig.add_subplot(gs[1, 1])
    p = powers
    # zeta_2_theory = 1.0
    # if hasattr(sys_vars, "alpha_high_k"):
    #     if sys_vars.alpha_high_k == 0.1:
    #         zeta_2_theory = 0.63
    #     elif sys_vars.alpha_high_k == 0.2:
    #         zeta_2_theory = 1.10
    #     else:
    #         zeta_2_theory = 1.0
    print("Trans: {}".format(zeta_p_trans[2]), np.array(zeta_p_trans[:]) / zeta_2_theory, zeta_2_theory)
    ns_zeta_p = [0.72, 1, 1.273, 1.534, 1.786]
    ax4.plot(p, np.array(zeta_p_trans[:]) / zeta_2_theory, marker = mark_style[0], markerfacecolor = 'None', markersize = 5.0, markevery = 1, label = "DNS")
    ax4.plot(p, np.array(zeta_p_trans[:]) / zeta_p_trans[2], marker = mark_style[0], markerfacecolor = 'None', markersize = 5.0, markevery = 1, label = "DNS / $\zeta_2,DNS$")
    ax4.plot(p, p, 'k--', label = "K41")
    ax4.set_xlabel(r"$p$")
    ax4.set_ylabel(r"$\zeta_{2p} / \zeta_{2, th}$")
    ax4.set_xlim(0, 2)
    ax4.set_ylim(0, 2)

    ax4.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax4.legend()
    ax4.set_title(r"Transverse $\zeta_{2p}$")

    plt.savefig(cmdargs.out_dir_stats + "Vorticity_Structure_Func_Anonalous_Exponent_Zeta_p.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------
    # # --------  Plot Radial Vorticity Structure Functions
    # ---------------------------------------------
    if hasattr(post_data, "vort_rad_str_func") and hasattr(post_data, "vort_rad_str_func_abs"):
        fig = plt.figure(figsize = (16, 8))
        gs  = GridSpec(1, 2, hspace = 0.3) 
        max_incr = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
        r = np.arange(1, np.round(np.sqrt(max_incr**2 + max_incr**2)) + 1)

        powers = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(post_data.vort_rad_str_func.shape[0]):
            ax1.plot(np.log2(r), np.log2(np.absolute(post_data.vort_rad_str_func[i, :])))
        ax1.set_xlabel(r"$r$")
        ax1.set_ylabel(r"$|S^p(r)|$")
        ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax1.set_title(r"Radial Structure Functions")
        ax1.legend([r"$p = {}$".format(p) for p in powers])

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(post_data.vort_rad_str_func_abs.shape[0]):
            ax2.plot(np.log2(r), np.log2(np.absolute(post_data.vort_rad_str_func_abs[i, :])))
        ax2.set_xlabel(r"$r$")
        ax2.set_ylabel(r"$|S_abs^p(r)|$")
        ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax2.set_title(r"Absolute Radial Structure Functions")
        ax2.legend([r"$p = {}$".format(p) for p in powers])
        
        plt.suptitle(r"Radial Vorticity Structure Functions")

        plt.savefig(cmdargs.out_dir_stats + "/Radial_Vorticity_Structure_Functions.png")
        plt.close()


    if hasattr(post_data, "vort_rad_str_func_abs"):
        inert_lim_low  = inert_range[0]
        inert_lim_high = inert_range[-1]

        mark_style = ['o','s','^','x','D','p']

        zeta_p_long = []
        powers = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5])
        max_incr = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
        r      = np.arange(1, np.round(np.sqrt(max_incr**2 + max_incr**2)) + 1)

        x0     = 0.85 
        y0     = 0.85
        width  = 0.3
        height = 0.2
        fig   = plt.figure(figsize = (16, 8))
        gs    = GridSpec(1, 2)

        ax1   = fig.add_subplot(gs[0, 0])
        for i in range(post_data.vort_rad_str_func_abs.shape[0]):
            ## Plot strucure function
            p, = ax1.plot(np.log2(r), np.log2(post_data.vort_rad_str_func_abs[i, :]), label = "$p = {}$".format(powers[i])) # marker = mark_style[i], markerfacecolor = 'None', markersize = 5.0, markevery = 2**4
            ## Find polynomial fit and plot
            pfit_info  = np.polyfit(np.log2(r[inert_lim_low:inert_lim_high]), np.log2(post_data.vort_rad_str_func_abs[i, inert_lim_low:inert_lim_high]), 1)
            pfit_slope = pfit_info[0]
            pfit_c     = pfit_info[1]
            zeta_p_long.append(np.absolute(pfit_slope))
            ax1.plot(np.log2(r[inert_lim_low:inert_lim_high]), np.log2(r[inert_lim_low:inert_lim_high])*pfit_slope + pfit_c + 0.25, '--', color = p.get_color())
        ax1.set_xlabel(r"$log_2 (r)$")
        ax1.set_ylabel(r"$log_2 (S_{2p}(r))$")
        ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax1.set_title(r"Absolute Radial Vorticity Structure Function")
        ax1.legend()

        ax4   = fig.add_subplot(gs[0, 1])
        p = powers
        # zeta_2_theory = 1.0
        # if hasattr(sys_vars, "alpha_high_k"):
        #     if sys_vars.alpha_high_k == 0.1:
        #         zeta_2_theory = 0.63
        #     elif sys_vars.alpha_high_k == 0.2:
        #         zeta_2_theory = 1.10
        #     else:
        #         zeta_2_theory = 1.0
        ns_zeta_p = [0.72, 1, 1.273, 1.534, 1.786]
        ax4.plot(p, np.array(zeta_p_trans[:]) / zeta_2_theory, marker = mark_style[0], markerfacecolor = 'None', markersize = 5.0, markevery = 1, label = "DNS")
        ax4.plot(p, p, 'k--', label = "K41")
        ax4.set_xlabel(r"$p$")
        ax4.set_ylabel(r"$\zeta_{2p} / \zeta_{2, th}$")
        # ax4.set_xlim(0, 2)
        ax4.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax4.legend()
        ax4.set_title(r"Radial $\zeta_{2p}$")

        plt.savefig(cmdargs.out_dir_stats + "Radial_Vorticity_Structure_Func_Anonalous_Exponent_Zeta_p.png", bbox_inches='tight')
        plt.close()