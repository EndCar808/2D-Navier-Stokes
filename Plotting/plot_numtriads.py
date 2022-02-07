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
    # ## Get the number of sectors
    # num_sects = [5, 10, 50, 100, 200]

    # avg_num_triads = []
    # for s in num_sects:

    #     print("S = {}".format(s))

    #     ## Get the file path for the current number of sectors
    #     post_file_path = cmdargs.in_dir + "PostProcessing_HDF_Data_SECTORS[{}].h5".format(s)

    #     ## Read in simulation parameters
    #     sys_vars = sim_data(cmdargs.in_dir)

    #     ## Read in solver data
    #     run_data = import_data(cmdargs.in_dir, sys_vars)

    #     ## Read in spectra data
    #     spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

    #     ## Read in post processing data
    #     post_data = import_post_processing_data(post_file_path, sys_vars, method)

    #     ## Compute the average number of triads per sector
    #     avg_num_triads.append(np.mean(post_data.num_triads[0, :]))

    #     print("Avg: {}\nNum: {}\n".format(np.mean(post_data.num_triads[0, :]), post_data.num_triads[0, :]))
    

    # # -----------------------------------------
    # # # --------  Plot data
    # # -----------------------------------------
    # ## Create Figure
    # fig = plt.figure(figsize = (16, 8))
    # gs  = GridSpec(1, 1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(num_sects, avg_num_triads)
    # ax1.set_xlabel(r"Number of Sectors")
    # ax1.set_ylabel(r"Average Number of Triads Per Sector")
    # ax1.set_xticks(num_sects)
    # ax1.set_xticklabels([r"$5$", r"$10$", r"$50$", r"$100$", r"$200$"])
    # ax1.set_yscale('log')
    # # ax1.set_xscale('log')
    # ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # plt.savefig(cmdargs.out_dir + "AverageNumberOfTriads.png")
    # plt.close()
    # 
    

    ## Get the file path for the current number of sectors
    post_file_path = cmdargs.in_dir + "PostProcessing_HDF_Data_SECTORS[{}]_KFRAC[{:.2f}].h5".format(50, 0.50)

    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars)

    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

    ## Read in post processing data
    post_data = import_post_processing_data(post_file_path, sys_vars, method)