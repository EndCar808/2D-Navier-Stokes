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
    # post_data = import_post_processing_data(post_file_path, sys_vars, method)


    ## Read in Brendan's data
    data_dir = "/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/navierstokes_mpi/3D_Euler_Spectral/results/RESULTS_NAVIER_AB4_N[256][256]_T[0-0.1]_[20-14-17]_[Kolo-Test]/"
    print(data_dir)

    with h5py.File(data_dir + "HDF_Global_REAL.h5", "r") as f:
        nn = 0
        for group in f.keys():
            if "Timestep" in group:
                nn += 1
        bren_w     = np.zeros((nn, 256, 256))
        bren_w_hat = np.ones((nn, 256, 129)) * np.complex(0.0, 0.0)

        nn = 0

        # Read in the vorticity
        for group in f.keys():
            if "Timestep" in group:
                if 'W' in list(f[group].keys()):
                    ww            = f[group]["W"][:, :]
                    bren_w_hat[nn,:, :] = np.fft.rfft2(ww)
                    bren_w[nn,    :, :] = ww
                    nn += 1

        fig, ax = plt.subplots(1, 2)
        im0 = ax[0].imshow(np.real(bren_w_hat[0, :, :]))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')

        im1 = ax[1].imshow(np.imag(bren_w_hat[0, :, :]))
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        
        plt.savefig(cmdargs.out_dir + "SNAPS/" + "BrendansIC_Fourier.png")
        plt.close()

        fig, ax = plt.subplots(1, 3)
        im = ax[0].imshow(bren_w[0, :, :])
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        im = ax[1].imshow(run_data.w[0, :, :])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        im = ax[2].imshow(np.absolute(bren_w[0, :, :] / (256 * 256) - run_data.w[0, :, :]))
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(cmdargs.out_dir + "SNAPS/" + "BrendansIC_Real.png")
        plt.close()

        print(bren_w[0, :, :] /(run_data.w[0, :, :]))
        print()
