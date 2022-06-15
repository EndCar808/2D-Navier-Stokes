#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to compare solver results with decaying turbulence papers
#        Solver data

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
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data
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

        def __init__(self, in_dir = None, out_dir = None, in_file = None, plotting = False):
            self.in_dir         = in_dir
            self.in_file        = out_dir
            self.plotting       = plotting


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:", ["plot", "phase", "triads="])
    except:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:

        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("\nInput Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        if opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True


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

    if run_data.no_w:
        print("\nPreparing real space vorticity...", end = " ")
        for i in range(sys_vars.ndata):
            run_data.w[i, :, :] = np.fft.irfft2(run_data.w_hat[i, :, :])
        print("Finished!")


    # -----------------------------------------
    # # --------  Plot Initial Condition
    # -----------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(run_data.w[0, :, :], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
    ax1.set_xlabel(r"$y$")
    ax1.set_ylabel(r"$x$")
    ax1.set_xlim(0.0, run_data.y[-1])
    ax1.set_ylim(0.0, run_data.x[-1])
    ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$\omega(x, y)$")
    ax1.set_title(r"Vorticity")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.real(run_data.w_hat[0, :, :]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
    ax2.set_xlabel(r"$y$")
    ax2.set_ylabel(r"$x$")
    ax2.set_xlim(0.0, run_data.y[-1])
    ax2.set_ylim(0.0, run_data.x[-1])
    ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_label(r"$\Re \hat{\omega}_{\mathbf{k}}$")
    ax2.set_title(r"Real Part Fourier Vorticity")

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(np.imag(run_data.w_hat[0, :, :]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
    ax3.set_xlabel(r"$y$")
    ax3.set_ylabel(r"$x$")
    ax3.set_xlim(0.0, run_data.y[-1])
    ax3.set_ylim(0.0, run_data.x[-1])
    ax3.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax3.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    div3  = make_axes_locatable(ax3)
    cbax3 = div3.append_axes("right", size = "10%", pad = 0.05)
    cb3   = plt.colorbar(im3, cax = cbax3)
    cb3.set_label(r"$\Im \hat{\omega}_{\mathbf{k}}$")
    ax3.set_title(r"Real Part Fourier Vorticity")

    kindx = int(sys_vars.Nx / 3 + 1)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(np.arange(1, kindx), spec_data.enrg_spectrum[0, 1:kindx])
    ax4.set_xlabel(r"$|\mathbf{k}|$")
    ax4.set_ylabel(r"$\mathcal{K}(| \mathbf{k} |)$") 
    ax4.set_title(r"Energy Spectrum")
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    kindx = int(sys_vars.Nx / 3 + 1)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(np.arange(1, kindx), spec_data.enst_spectrum[0, 1:kindx])
    ax4.set_xlabel(r"$|\mathbf{k}|$")
    ax4.set_ylabel(r"$\mathcal{E}(| \mathbf{k} |)$") 
    ax4.set_title(r"Enstrophy Spectrum")
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)


    plt.savefig(cmdargs.out_dir + "Initial_Condition.png", bbox_inches='tight') 
    plt.close()


    # -----------------------------------------
    # # --------  Plot System Measures
    # -----------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(run_data.time, run_data.tot_palin[:])
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\mathcal{P}$")
    ax1.grid()

    plt.savefig(cmdargs.out_dir + "TotalPalin.png", bbox_inches='tight') 
    plt.close()