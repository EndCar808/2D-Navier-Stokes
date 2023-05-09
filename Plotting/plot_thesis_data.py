import numpy as np
import h5py as h5
import sys
import os
from numba import njit
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
if mpl.__version__ > '2':    
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
from subprocess import Popen, PIPE, run
from matplotlib.pyplot import cm
import concurrent.futures as cf
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, get_flux_spectrum, compute_pdf


rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = 'Computer Modern Roman'
rcParams['font.size']   = 12.5 # def: 10.0
## Lines
rcParams['lines.linewidth']  = 1.5 # def: 1.5
rcParams['lines.markersize'] = 5.0 # def: 6
rcParams['lines.markersize'] = 5.0 # def: 6
## Grid lines
rcParams['grid.color']     = 'k'
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['grid.alpha']     = 0.8
##Ticks
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top']       = True
rcParams['ytick.right']     = True
## Figsize
textwidth = 12.25
rcParams['figure.figsize'] = [textwidth, 0.4 * textwidth]

inset_lab_size = 10
label_size = 16

plot_colours = plt.cm.YlGnBu(np.linspace(0.3, 1.0, 10))

fig_size_nx1 = (0.7 * textwidth, 1.2 * textwidth)
fig_size_2x2 = (textwidth, 0.8 * textwidth)

my_magma = matplotlib.colors.ListedColormap(cm.magma.colors[::-1])
my_magma.set_under(color = "white")
my_cmap = matplotlib.colors.ListedColormap(cm.YlGnBu(np.arange(0,cm.YlGnBu.N)))
my_cmap.set_under(color = "white")

fig_format = 'pdf'

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

        def __init__(self, in_dir = None, out_dir = None, in_file = None, num_procs = 20, parallel = False, plotting = False, video = False, triads = False, phases = False, triad_type = None, full = False):
            self.in_dir         = in_dir
            self.out_dir_phases = out_dir
            self.out_dir_triads = out_dir
            self.in_file        = out_dir
            self.parallel       = parallel
            self.plotting       = plotting
            self.video          = video 
            self.phases         = phases
            self.triads         = triads
            self.triad_type     = triad_type
            self.full           = full
            self.triad_plot_type = None
            self.num_procs = num_procs
            self.tag = "None"
            self.phase_order = False


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:t:", ["par", "plot", "vid", "phase", "triads=", "full=", "phase_order"])
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

        elif opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

        elif opt in ['-p']:
            ## Read in number of processes
            cargs.num_procs = int(arg)

        elif opt in ['--par']:
            ## Read in parallel indicator
            cargs.parallel = True

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--vid']:
            ## If videos are to be made
            cargs.video = True

        elif opt in ['--phase']:
            ## If phases are to be plotted
            cargs.phases = True

        elif opt in ['--full']:
            ## If full figure is to be plotted
            cargs.full = True

            ## Read in the triad type
            cargs.triad_plot_type = str(arg)
        elif opt in ['--triads']:
            ## If triads are to be plotted
            cargs.triads = True

            ## Read in the triad type
            cargs.triad_type = str(arg)
        elif opt in ['--phase_order']:
            ## If phase_order are to be plotted
            cargs.phase_order = True

        elif opt in ['-t']:
            cargs.tag = str(arg)

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
    print("Reading in Data")
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)



    ## Read in post processing data
    post_data = import_post_processing_data(post_file_path, sys_vars, method)


    # -----------------------------------------
    # # --------  Plot Validation 
    # -----------------------------------------
    ## Normalizing constants
    const_fac_sector = 4.0 * np.pi**2 
    const_fac = -0.5 #4.0 * np.pi**2 
    norm_fac  = (sys_vars.Nx * sys_vars.Ny)**2

    cmdargs.out_dir = "/home/enda/PhD/2D-Navier-Stokes/Data/ThesisPlots"

    fig = plt.figure()
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.sum(post_data.enst_flux_per_sec[:, triad_type, :], axis = -1) * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5, label=r"$\sum_\theta\Pi_{\mathcal{S}_\theta^{U}}$ Direct")
    ax3.plot(np.sum(np.real(post_data.phase_order_C_theta[:, :]), axis = -1), '*-', markevery=5, label=r"$\sum_\theta \Re\{Re^{\Phi}\}$ NL")
    ax3.plot(np.sum(np.real(post_data.phase_order_C_theta_triads[:, triad_type, :]) * norm_fac / 2, axis = -1), '--', label=r"$\sum_\theta \Re\{\mathcal{R}_\theta\}$ Direct")
    ax3.grid()
    ax3.legend()
    ax3.set_xlabel(r"$t$", fontsize=label_size)
    plt.savefig(cmdargs.out_dir + "/2d_validation" + "." + fig_format, format=fig_format, bbox_inches="tight", dpi=1200)
    plt.close()