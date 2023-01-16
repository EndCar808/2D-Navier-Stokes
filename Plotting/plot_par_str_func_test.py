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
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    ## Read in base post processing data
    post_data_base = import_post_processing_data(post_file_path, sys_vars, method)

    ## Create threads list
    thread_pow = 2
    num_pow    = 6
    threads_list = [2**i for i in range(thread_pow + 1)]
    print(threads_list)

    ## Allocate memory for data
    rad_vort_str_func_abs = np.zeros((thread_pow + 1, post_data_base.vort_rad_str_func_abs.shape[0], post_data_base.vort_rad_str_func_abs.shape[1]))
    rad_vort_str_func     = np.zeros((thread_pow + 1, post_data_base.vort_rad_str_func.shape[0], post_data_base.vort_rad_str_func.shape[1]))
    vort_str_func         = np.zeros((thread_pow + 1, num_pow, sys_vars.Nx//2, 2))
    vort_str_func_abs     = np.zeros((thread_pow + 1, num_pow, sys_vars.Nx//2, 2))
    vel_str_func          = np.zeros((thread_pow + 1, num_pow, sys_vars.Nx//2, 2))
    vel_str_func_abs      = np.zeros((thread_pow + 1, num_pow, sys_vars.Nx//2, 2))

    ## Read in post data from each of the post data files
    for i, thrd in enumerate(threads_list):
        print("Num Thread = {}".format(thrd))
        ## Post file
        post_file_path_nthread = cmdargs.in_dir +  "PostProcessing_HDF_Data_THREADS[{},1]_SECTORS[24,24]_KFRAC[0.50]_TAG[ParStrFunc-Test].h5".format(thrd)

        ## Read in from post file
        post_data = import_post_processing_data(post_file_path_nthread, sys_vars, method)

        ## Record the data
        rad_vort_str_func[i, :, :]     = post_data.vort_rad_str_func[:, :]
        rad_vort_str_func_abs[i, :, :] = post_data.vort_rad_str_func_abs[:, :]
        vort_str_func[i, :, :, 0]      = post_data.vort_long_str_func[:, :]
        vort_str_func[i, :, :, 1]      = post_data.vort_trans_str_func[:, :]
        vort_str_func_abs[i, :, :, 0]  = post_data.vort_long_str_func_abs[:, :]
        vort_str_func_abs[i, :, :, 1]  = post_data.vort_trans_str_func_abs[:, :]
        vel_str_func[i, :, :, 0]       = post_data.vel_long_str_func[:, :]
        vel_str_func[i, :, :, 1]       = post_data.vel_trans_str_func[:, :]
        vel_str_func_abs[i, :, :, 0]   = post_data.vel_long_str_func_abs[:, :]
        vel_str_func_abs[i, :, :, 1]   = post_data.vel_trans_str_func_abs[:, :]
            
    print()
    print()

    for i, thrd in enumerate(threads_list):
        print("Num Thread = {}".format(thrd))
        print("Radial Vort Str Func:\t{}".format(np.allclose(post_data_base.vort_rad_str_func[:, :], rad_vort_str_func[i, :, :], equal_nan = True)))
        print("Radial Vort Str Func Abs:\t{}".format(np.allclose(post_data_base.vort_rad_str_func_abs[:, :], rad_vort_str_func_abs[i, :, :], equal_nan = True)))
        print("Long Vort Str Func:\t{}".format(np.allclose(post_data_base.vort_long_str_func[:, :], vort_str_func[i, :, :, 0], equal_nan = True)))
        print("Long Vort Str Func Abs:\t{}".format(np.allclose(post_data_base.vort_long_str_func_abs[:, :], vort_str_func_abs[i, :, :, 0], equal_nan = True)))
        print("Trans Vort Str Func:\t{}".format(np.allclose(post_data_base.vort_trans_str_func[:, :], vort_str_func[i, :, :, 1], equal_nan = True)))
        print("Trans Vort Str Func Abs:\t{}".format(np.allclose(post_data_base.vort_trans_str_func_abs[:, :], vort_str_func_abs[i, :, :, 1], equal_nan = True)))
        print("Long Vel Str Func:\t{}".format(np.allclose(post_data_base.vel_long_str_func[:, :], vel_str_func[i, :, :, 0], equal_nan = True)))
        print("Long Vel Str Func Abs:\t{}".format(np.allclose(post_data_base.vel_long_str_func_abs[:, :], vel_str_func_abs[i, :, :, 0], equal_nan = True)))
        print("Trans Vel Str Func:\t{}".format(np.allclose(post_data_base.vel_trans_str_func[:, :], vel_str_func[i, :, :, 1], equal_nan = True)))
        print("Trans Vel Str Func Abs:\t{}".format(np.allclose(post_data_base.vel_trans_str_func_abs[:, :], vel_str_func_abs[i, :, :, 1], equal_nan = True)))