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

        def __init__(self, in_dir = None, out_dir = None, in_file = None, plotting = False, video = False, triads = False, phases = False, triad_type = None):
            self.in_dir         = in_dir
            self.out_dir_phases = out_dir
            self.out_dir_triads = out_dir
            self.in_file        = out_dir
            self.plotting       = plotting
            self.phases         = phases
            self.triads         = triads
            self.triad_type     = triad_type


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
            print("Input Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        if opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

        elif opt in ['--par']:
            ## Read in parallel indicator
            cargs.parallel = True

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--phase']:
            ## If phases are to be plotted
            cargs.phases = True

        elif opt in ['--triads']:
            ## If triads are to be plotted
            cargs.triads = True

            ## Read in the triad type
            cargs.triad_type = str(arg)

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



    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2) 

    ## Compute pdf
    phase_counts = np.sum(post_data.phase_sector_counts, axis = 1)
    bin_centres  = (post_data.phase_sector_ranges[0, 0, 1:] + post_data.phase_sector_ranges[0, 0, :-1]) * 0.5
    bin_width    = post_data.phase_sector_ranges[0, 0, 1] - post_data.phase_sector_ranges[0, 0, 0]
    pdf = np.empty_like(phase_counts)
    for i in range(phase_counts.shape[0]):
        pdf[i, :] =  phase_counts[i, :] / (np.sum(phase_counts[i, :]) * bin_width)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pdf, extent = (-np.pi, np.pi, run_data.time[-1], run_data.time[0]), aspect = 'auto', cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]))
    ax1.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2.0, np.pi])
    ax1.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    ax1.set_xlabel(r"$\phi_{\mathbf{k}}$")
    ax1.set_ylabel(r"$t$")
    ax1.set_title([r"PDF of The Phases Over Time"])
    ax1.set_ylim(run_data.time[0], run_data.time[-1])
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"PDF")

    ## Compute weighted pdf
    phase_counts = np.sum(post_data.phase_sector_counts, axis = 1)
    bin_centres  = (post_data.phase_sector_ranges[0, 0, 1:] + post_data.phase_sector_ranges[0, 0, :-1]) * 0.5
    bin_width    = post_data.phase_sector_ranges[0, 0, 1] - post_data.phase_sector_ranges[0, 0, 0]
    pdf = np.empty_like(phase_counts)
    for i in range(phase_counts.shape[0]):
        pdf[i, :] =  phase_counts[i, :] / (np.sum(phase_counts[i, :]) * bin_width)


    ax1 = fig.add_subplot(gs[0, 1])

    plt.savefig(cmdargs.out_dir + "/Phase_PDF_OverTime.png")
    plt.close()


    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2) 

    ## Compute pdf
    triad_counts = np.sum(post_data.triad_sector_counts, axis = 1)
    bin_centres  = (post_data.triad_sector_ranges[0, 0, 1:, 0] + post_data.triad_sector_ranges[0, 0, :-1, 0]) * 0.5
    bin_width    = post_data.triad_sector_ranges[0, 0, 1, 0] - post_data.triad_sector_ranges[0, 0, 0, 0]
    pdf = np.empty_like(triad_counts)
    for i in range(triad_counts.shape[0]):
        pdf[i, :] =  triad_counts[i, :] / (np.sum(triad_counts[i, :]) * bin_width)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pdf, extent = (-np.pi, np.pi, run_data.time[-1], run_data.time[0]), aspect = 'auto', cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]))
    ax1.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2.0, np.pi])
    ax1.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    ax1.set_xlabel(r"$\phi_{\mathbf{k}}$")
    ax1.set_ylabel(r"$t$")
    ax1.set_title([r"PDF of The Triad Phases Over Time"])
    ax1.set_ylim(run_data.time[0], run_data.time[-1])
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"PDF")

    # ## Compute weighted pdf
    # triad_wghtd_counts = np.sum(post_data.triad_sector_wghtd_counts, axis = 1)
    # bin_centres  = (post_data.triad_sector_wghtd_ranges[0, 0, 1:, 0] + post_data.triad_sector_wghtd_ranges[0, 0, :-1, 0]) * 0.5
    # bin_width    = post_data.triad_sector_wghtd_ranges[0, 0, 1, 0] - post_data.triad_sector_wghtd_ranges[0, 0, 0, 0]
    # pdf = np.empty_like(triad_wghtd_counts)
    # for i in range(triad_wghtd_counts.shape[0]):
    #     pdf[i, :] =  triad_wghtd_counts[i, :] / (np.sum(triad_wghtd_counts[i, :]) * bin_width)
        
    # ax2 = fig.add_subplot(gs[0, 1])
    # im2 = ax2.imshow(pdf, extent = (-np.pi, np.pi, run_data.time[-1], run_data.time[0]), aspect = 'auto', cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]))
    # ax2.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2.0, np.pi])
    # ax2.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax2.set_xlabel(r"$\phi_{\mathbf{k}}$")
    # ax2.set_ylabel(r"$t$")
    # ax2.set_title([r"Weighted PDF of The Triad Phases Over Time"])
    # ax2.set_ylim(run_data.time[0], run_data.time[-1])
    # div2  = make_axes_locatable(ax2)
    # cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    # cb2   = plt.colorbar(im2, cax = cbax2)
    # cb2.set_label(r"PDF")
    
    plt.savefig(cmdargs.out_dir + "/TriadPhase_All_PDF_OverTime.png")
    plt.close()