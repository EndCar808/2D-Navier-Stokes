#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##  
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
import numpy as np
import h5py
import sys
import os
import getopt
from itertools import zip_longest
import multiprocessing as mprocs
import time as TIME
from subprocess import Popen, PIPE
from numba import njit
import pyfftw as fftw

from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data



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
            self.out_dir_phases = out_dir
            self.out_dir_triads = out_dir
            self.in_file        = out_dir


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:", ["plot"])
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


    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars)

    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)


    ## Compute the enstrophy flux as a function of k and t
    enst_flux = np.zeros((sys_vars.ndata, sys_vars.spec_size))
    for t in range(sys_vars.ndata):
    	for k in range(sys_vars.spec_size):

    		enst_flux[t, k] = np.sum(spec_data.enst_flux_spectrum[t, :k])


    ## Plot data
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ## Create grid
    x, y = np.meshgrid(np.arange(1, sys_vars.spec_size + 1), run_data.time)
    ## Plot surface
    surf = ax.plot_surface(x, y, enst_flux, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$t$")
    ax.set_zlabel(r"$\Pi_k$")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(cmdargs.out_dir + "/MaxEnstrophyFlux.png")

    max_enst = np.where(enst_flux == np.amax(enst_flux))

    print(max_enst)
    print(type(max_enst))

    print(max_enst[0])

    print(max_enst[0][0])

    print(enst_flux[max_enst[0], max_enst[1]])