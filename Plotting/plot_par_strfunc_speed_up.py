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
from functions import tc
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
            self.n_pow          = 0


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:n:", ["incr", "wghtd"])
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

        elif opt in ['-n']:
            ## Read Max thread power (number of threads in power of 2)
            cargs.n_pow = int(arg)

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
    plot_dir = "/home/enda/PhD/2D-Navier-Stokes/Data/Testing/ParStrFunc/"
    # -------------------------------------
    # # --------- Read in Data from Input File
    # -------------------------------------
    N       = [2**i for i in range(5, cmdargs.n_pow + 1)]
    threads = [2**i for i in range(cmdargs.thread_pow + 1)]
    ex_time = {}
    for n in N:
        for t in threads:
            key = str(n) + "-" + str(t)
            ex_time[key] = None

    times = []
    with open(cmdargs.in_file) as f:
        for line in f.readlines():
            
            ## Get the execution time
            if line.startswith("Total Execution Time:"):
                time = float(line.split()[3].split("m")[1].split("\x1b")[0])
                times.append(time)


    for i, n in enumerate(N):
        for j, t in enumerate(threads):
            key = str(n) + "-" + str(t)
            ex_time[key] = times[i * len(threads) + j]

    for n in np.unique(N):
        key = str(n) + "-" + str(1)
        base_ex_time = ex_time[key]

        thread_counts = []
        speed_up = []
        print("Speed Up for N: {}\n------------------------".format(n))
        for i, t in enumerate(threads[1:]):
            key = str(n) + "-" + str(t)
            print("Threads: {}\tSpeed Up: {}".format(t,  base_ex_time / ex_time[key]))
            
            thread_counts.append(t)
            speed_up.append(base_ex_time / ex_time[key])

        fig = plt.figure(figsize = (16, 8))
        gs  = GridSpec(1, 1, hspace = 0.3) 
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(thread_counts, speed_up, '-')
        ax1.set_ylabel("Speed Up")
        ax1.set_xlabel("Threads")
        ax1.set_xticks(threads)
        plt.savefig(plot_dir + "Speed_Up_N[{}].png".format(n))
        plt.close()

