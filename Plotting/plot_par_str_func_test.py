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
            self.thread_pow     = 0


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:", ["incr", "wghtd"])
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

        elif opt in ['-p']:
            ## Read Max thread power (number of threads in power of 2)
            cargs.thread_pow = int(arg)


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

    # ## Read in base post processing data
    # post_data_base = import_post_processing_data(post_file_path, sys_vars, method)

    ## Create threads list
    thread_pow = cmdargs.thread_pow
    num_pow    = 6
    threads_list = [2**i for i in range(thread_pow + 1)]
    # print(threads_list)

    ## Allocate memory for data
    N_max_incr            = sys_vars.Nx//2
    num_rad_incr          = int(np.round(np.sqrt((N_max_incr) * (N_max_incr) + (N_max_incr) * (N_max_incr))))
    rad_vort_str_func_abs = np.zeros((thread_pow, num_pow, num_rad_incr))
    rad_vort_str_func     = np.zeros((thread_pow, num_pow, num_rad_incr))
    vort_str_func         = np.zeros((thread_pow, num_pow, N_max_incr, 2))
    vort_str_func_abs     = np.zeros((thread_pow, num_pow, N_max_incr, 2))
    vel_str_func          = np.zeros((thread_pow, num_pow, N_max_incr, 2))
    vel_str_func_abs      = np.zeros((thread_pow, num_pow, N_max_incr, 2))

    ## Read in post data from each of the post data files
    for i, thrd in enumerate(threads_list):
        # print("Num Thread = {}".format(thrd))

        ## Post file
        post_file_path_nthread = cmdargs.in_dir +  "PostProcessing_HDF_Data_THREADS[{},1]_SECTORS[24,24]_KFRAC[0.50]_TAG[ParStrFunc-Test].h5".format(thrd)

        ## Read in from post file
        if i == 0:
            post_data_base = import_post_processing_data(post_file_path_nthread, sys_vars, method)
        else:
            post_data = import_post_processing_data(post_file_path_nthread, sys_vars, method)

            ## Record the data
            rad_vort_str_func[i - 1, :, :]     = post_data.vort_rad_str_func[:, :]
            rad_vort_str_func_abs[i - 1, :, :] = post_data.vort_rad_str_func_abs[:, :]
            vort_str_func[i - 1, :, :, 0]      = post_data.vort_long_str_func[:, :]
            vort_str_func[i - 1, :, :, 1]      = post_data.vort_trans_str_func[:, :]
            vort_str_func_abs[i - 1, :, :, 0]  = post_data.vort_long_str_func_abs[:, :]
            vort_str_func_abs[i - 1, :, :, 1]  = post_data.vort_trans_str_func_abs[:, :]
            vel_str_func[i - 1, :, :, 0]       = post_data.vel_long_str_func[:, :]
            vel_str_func[i - 1, :, :, 1]       = post_data.vel_trans_str_func[:, :]
            vel_str_func_abs[i - 1, :, :, 0]   = post_data.vel_long_str_func_abs[:, :]
            vel_str_func_abs[i - 1, :, :, 1]   = post_data.vel_trans_str_func_abs[:, :]

    print()
    print()

    # -----------------------------------------
    # # --------  Compare Data
    # -----------------------------------------
    rel_tol = 1e-10
    abs_tol = 1e-14  
    print("Comparing Data For N = {}".format(sys_vars.Nx))
    for i, thrd in enumerate(threads_list[1:]):
        print("Num Thread = {}\n-----------------------------------------------------".format(thrd))
        print("Radial Vort Str Func:\t{}".format(np.allclose(post_data_base.vort_rad_str_func[:, :], rad_vort_str_func[i, :, :], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Radial Vort Str Func Abs:\t{}".format(np.allclose(post_data_base.vort_rad_str_func_abs[:, :], rad_vort_str_func_abs[i, :, :], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Long Vort Str Func:\t{}".format(np.allclose(post_data_base.vort_long_str_func[:, :], vort_str_func[i, :, :, 0], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Long Vort Str Func Abs:\t{}".format(np.allclose(post_data_base.vort_long_str_func_abs[:, :], vort_str_func_abs[i, :, :, 0], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Trans Vort Str Func:\t{}".format(np.allclose(post_data_base.vort_trans_str_func[:, :], vort_str_func[i, :, :, 1], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Trans Vort Str Func Abs:\t{}".format(np.allclose(post_data_base.vort_trans_str_func_abs[:, :], vort_str_func_abs[i, :, :, 1], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Long Vel Str Func:\t{}".format(np.allclose(post_data_base.vel_long_str_func[:, :], vel_str_func[i, :, :, 0], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Long Vel Str Func Abs:\t{}".format(np.allclose(post_data_base.vel_long_str_func_abs[:, :], vel_str_func_abs[i, :, :, 0], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Trans Vel Str Func:\t{}".format(np.allclose(post_data_base.vel_trans_str_func[:, :], vel_str_func[i, :, :, 1], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print("Trans Vel Str Func Abs:\t{}".format(np.allclose(post_data_base.vel_trans_str_func_abs[:, :], vel_str_func_abs[i, :, :, 1], equal_nan = True, rtol = rel_tol, atol = abs_tol)))
        print()
        print()

    # -----------------------------------------
    # # --------  Plot Data
    # -----------------------------------------
    ## Get the plotting directory
    plot_dir = "/home/enda/PhD/2D-Navier-Stokes/Data/Testing/ParStrFunc/"
    
    powers = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    plot_chars = ['o','s','^','x','D','p']

    ## Plot Radial Structure Function Data
    print("Plotting Data For N = {}".format(sys_vars.Nx))
    print("Plotting Radial Str Func Data")
    max_incr = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
    r        = np.arange(1, np.round(np.sqrt(max_incr**2 + max_incr**2)) + 1)
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2, hspace = 0.3) 
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(post_data_base.vort_rad_str_func.shape[0]):
        p_colr, = ax1.plot(np.log2(r), np.log2(np.absolute(post_data_base.vort_rad_str_func[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax1.plot(np.log2(r), np.log2(np.absolute(rad_vort_str_func[j, i, :])), color = p_colr.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$|S^p(r)|$")
    ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax1.set_title(r"Radial Structure Functions")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(post_data_base.vort_rad_str_func_abs.shape[0]):
        p_colr, = ax2.plot(np.log2(r), np.log2(np.absolute(post_data_base.vort_rad_str_func_abs[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax2.plot(np.log2(r), np.log2(np.absolute(rad_vort_str_func_abs[j, i, :])), color = p_colr.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|S_abs^p(r)|$")
    ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax2.set_title(r"Absolute Radial Structure Functions")
    ax2.legend()
    plt.suptitle(r"Radial Vorticity Structure Functions")
    plt.savefig(plot_dir + "Radial_Vorticity_Structure_Functions_N[{},{}].png".format(sys_vars.Nx, sys_vars.Ny))
    plt.close()

    ## Plot Vorticity Str Func Data
    print("Plotting Vorticity Str Func Data")
    fig = plt.figure(figsize = (32, 16))
    gs  = GridSpec(2, 2, hspace = 0.3) 
    r = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)
    # L = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
    powers = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(post_data_base.vort_long_str_func.shape[0]):
        # ax1.plot(r / , np.absolute(post_data_base.vort_long_str_func[i, :]))
        p_col,  = ax1.plot(np.log2(r), np.log2(np.absolute(post_data_base.vort_long_str_func[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax1.plot(np.log2(r), np.log2(np.absolute(vort_str_func[j, i, :, 0])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$|S^p(r)|$")
    ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax1.set_title(r"Longitudinal Structure Functions")
    ax1.legend()
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(post_data_base.vort_trans_str_func.shape[0]):
        # ax2.plot(r / L, np.absolute(post_data_base.vort_trans_str_func[i, :]))
        p_col,  = ax2.plot(np.log2(r) , np.log2(np.absolute(post_data_base.vort_trans_str_func[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax2.plot(np.log2(r), np.log2(np.absolute(vort_str_func[j, i, :, 1])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|S^p(r)|$")
    ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax2.set_title(r"Transverse Structure Functions")
    ax2.legend()
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(post_data_base.vort_long_str_func_abs.shape[0]):
        # ax3.plot(r / L, np.absolute(post_data_base.vort_long_str_func_abs[i, :]))
        p_col,  = ax3.plot(np.log2(r) , np.log2(np.absolute(post_data_base.vort_long_str_func_abs[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax3.plot(np.log2(r), np.log2(np.absolute(vort_str_func_abs[j, i, :, 0])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax3.set_xlabel(r"$r$")
    ax3.set_ylabel(r"$|S^p_{abs}(r)|$")
    ax3.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax3.set_title(r"Absolute Longitudinal Structure Functions")
    ax3.legend()
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(post_data_base.vort_trans_str_func_abs.shape[0]):
        # ax4.plot(r / L, np.absolute(post_data_base.vort_trans_str_func_abs[i, :]))
        p_col,  = ax4.plot(np.log2(r) , np.log2(np.absolute(post_data_base.vort_trans_str_func_abs[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax4.plot(np.log2(r), np.log2(np.absolute(vort_str_func_abs[j, i, :, 1])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax4.set_xlabel(r"$r$")
    ax4.set_ylabel(r"$|S^p_{abs}(r)|$")
    ax4.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax4.set_title(r"Absolute Transverse Structure Functions")
    ax4.legend()
    plt.suptitle(r"Vorticity Structure Functions")
    plt.savefig(plot_dir + "Vorticity_Structure_Functions_N[{},{}].png".format(sys_vars.Nx, sys_vars.Ny))
    plt.close()


    ## Plot Vorticity Str Func Data
    print("Plotting Velocity Str Func Data")
    fig = plt.figure(figsize = (32, 16))
    gs  = GridSpec(2, 2, hspace = 0.3) 
    r = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)
    # L = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2
    powers = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(post_data_base.vel_long_str_func.shape[0]):
        # ax1.plot(r / , np.absolute(post_data_base.vel_long_str_func[i, :]))
        p_col,  = ax1.plot(np.log2(r), np.log2(np.absolute(post_data_base.vel_long_str_func[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax1.plot(np.log2(r), np.log2(np.absolute(vel_str_func[j, i, :, 0])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$|S^p(r)|$")
    ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax1.set_title(r"Longitudinal Structure Functions")
    ax1.legend()
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(post_data_base.vel_trans_str_func.shape[0]):
        # ax2.plot(r / L, np.absolute(post_data_base.vel_trans_str_func[i, :]))
        p_col,  = ax2.plot(np.log2(r) , np.log2(np.absolute(post_data_base.vel_trans_str_func[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax2.plot(np.log2(r), np.log2(np.absolute(vel_str_func[j, i, :, 1])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$|S^p(r)|$")
    ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax2.set_title(r"Transverse Structure Functions")
    ax2.legend()
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(post_data_base.vel_long_str_func_abs.shape[0]):
        # ax3.plot(r / L, np.absolute(post_data_base.vel_long_str_func_abs[i, :]))
        p_col,  = ax3.plot(np.log2(r) , np.log2(np.absolute(post_data_base.vel_long_str_func_abs[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax3.plot(np.log2(r), np.log2(np.absolute(vel_str_func_abs[j, i, :, 0])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax3.set_xlabel(r"$r$")
    ax3.set_ylabel(r"$|S^p_{abs}(r)|$")
    ax3.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax3.set_title(r"Absolute Longitudinal Structure Functions")
    ax3.legend()
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(post_data_base.vel_trans_str_func_abs.shape[0]):
        # ax4.plot(r / L, np.absolute(post_data_base.vel_trans_str_func_abs[i, :]))
        p_col,  = ax4.plot(np.log2(r) , np.log2(np.absolute(post_data_base.vel_trans_str_func_abs[i, :])), label = "p = {}".format(powers[i]))
        for j in range(len(threads_list[1:])):
            ax4.plot(np.log2(r), np.log2(np.absolute(vel_str_func_abs[j, i, :, 1])), color = p_col.get_color(), marker = plot_chars[j], markerfacecolor = 'None', markersize = 10.0, markevery = 2, label = "thrd = {}".format(2**(j + 1)))
    ax4.set_xlabel(r"$r$")
    ax4.set_ylabel(r"$|S^p_{abs}(r)|$")
    ax4.grid(color = 'k', linewidth = .5, linestyle = ':')
    ax4.set_title(r"Absolute Transverse Structure Functions")
    ax4.legend()
    plt.suptitle(r"Vorticity Structure Functions")
    plt.savefig(plot_dir + "Velocity_Structure_Functions_N[{},{}].png".format(sys_vars.Nx, sys_vars.Ny))
    plt.close()