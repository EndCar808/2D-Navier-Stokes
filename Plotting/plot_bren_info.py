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
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, import_bren_data
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

    snaps_output_dir = cmdargs.out_dir + "RUN_INFO_SNAPS/"
    if os.path.isdir(snaps_output_dir) != True:
        print("Making folder:" + tc.C + " RUN_INFO_SNAPS/" + tc.Rst)
        os.mkdir(snaps_output_dir)
        print()
        
    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    bren_data = import_bren_data(cmdargs.in_dir)



    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    ##---------- Energy Enstrophy
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(2, 2)
    ## Plot the relative energy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bren_data.time, bren_data.tot_enrg / bren_data.tot_enrg[0] - 1)
    ax1.set_xlabel(r"$t$")
    ax1.set_title(r"Relative Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the relative enstrophy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(bren_data.time, 1 - bren_data.tot_enst / bren_data.tot_enst[0])
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Relative Enstrophy")
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the relative energy
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(bren_data.time, bren_data.tot_enrg)
    ax1.set_xlabel(r"$t$")
    ax1.set_title(r"Total Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the relative helicity
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(bren_data.time, bren_data.tot_enst)
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Total Enstrophy")
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    plt.savefig(snaps_output_dir + "Energy_Enstrophy_Tseries.png")
    plt.close()


    ##---------- Spectra
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bren_data.enrg_spec[-1, :], label = "$$")
    ax1.set_xlabel(r"$k$")
    ax1.set_title(r"Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(bren_data.enst_spec[-1, :], label = "$$")
    ax2.set_xlabel(r"$k$")
    ax2.set_title(r"Energy")
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.savefig(snaps_output_dir + "Spectra.png")
    plt.close()



    ##---------- Flux Spectra
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(1, int(bren_data.Nx/3)), bren_data.enrg_spec[-1, 1:int(bren_data.Nx//3)], label = "$$")
    ax1.set_xlabel(r"$k$")
    ax1.set_title(r"Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.arange(1, int(bren_data.Nx/3)), bren_data.enst_spec[-1, 1:int(bren_data.Nx//3)], label = "$$")
    ax2.set_xlabel(r"$k$")
    ax2.set_title(r"Energy")
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.savefig(snaps_output_dir + "Spectra.png")
    plt.close()


    ##------------------------ Time Averaged Enstorphy Spectra and Flux Spectra
    fig = plt.figure(figsize = (21, 8))
    gs  = GridSpec(1, 2)
    ax2 = fig.add_subplot(gs[0, 0])
    for i in range(bren_data.enst_spec.shape[0]):
        ax2.plot(np.arange(1, int(bren_data.Nx/3)), bren_data.enst_spec[i, 1:int(bren_data.Nx/3)], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(bren_data.Nx/3)), np.mean(bren_data.enst_spec[:, 1:int(bren_data.Nx/3)], axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\mathcal{E}(|\mathbf{k}|)$: Enstrophy Spec")

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(bren_data.flux_w_spec.shape[0]):
        ax2.plot(np.arange(1, int(bren_data.Nx/3)), bren_data.flux_w_spec[i, 1:int(bren_data.Nx/3)], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(bren_data.Nx/3)), np.mean(bren_data.flux_w_spec[:, 1:int(bren_data.Nx/3)], axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\Pi(|\mathbf{k}|)$: Enstrophy Flux Spec")
    
    plt.savefig(snaps_output_dir + "TimeAveragedEnstrophySpectra.png")
    plt.close()
