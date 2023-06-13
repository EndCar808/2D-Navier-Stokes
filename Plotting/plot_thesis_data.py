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
from matplotlib import rc, rcParams
import matplotlib
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

    # Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars)

    ## Read in post processing data
    post_data = import_post_processing_data(post_file_path, sys_vars, method)

    fig_format = 'pdf'
    
    cmdargs.out_dir = "/home/enda/PhD/2D-Navier-Stokes/Data/ThesisPlots"

    theta_k3 = post_data.theta_k3
    dtheta_k3 = theta_k3[1] - theta_k3[0]
    angticks      = [-np.pi/2, -np.pi/4.0, 0.0, np.pi/4.0, np.pi/2.0]
    angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]
    # angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
    # angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
    angtickLabels_alpha = np.linspace(0, len(theta_k3) + 1, num = len(angtickLabels), endpoint = False, dtype = "int64").tolist()
    theta_k3_min     = -np.pi/2 - dtheta_k3 / 2
    theta_k3_max     = np.pi/2 + dtheta_k3 /2
    alpha_min = theta_k3_min
    alpha_max = theta_k3_max

    # # -----------------------------------------
    # # # --------  Plot Validation 
    # # -----------------------------------------
    # ## Normalizing constants
    # const_fac = -1.0 #4.0 * np.pi**2 
    # norm_fac  = (sys_vars.Nx * sys_vars.Ny)**2


    # fig = plt.figure()
    # gs  = GridSpec(1, 1)
    # ax3 = fig.add_subplot(gs[0, 0])
    # triad_type=0
    # ax3.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, triad_type, :], axis = -1) * const_fac * norm_fac, '-', marker='o', markersize=7.5, markevery=2, label=r"$\sum_\theta\Pi_{\mathcal{S}_\theta^U}$ ", color=plot_colours[-3])
    # ax3.plot(run_data.time, np.sum(np.real(post_data.phase_order_C_theta[:, :]), axis = -1) * 2, '*-', markevery=2, label=r"$\sum_\theta \mathcal{NL}_\theta$", color=plot_colours[0])
    # ax3.plot(run_data.time, np.sum(np.real(post_data.phase_order_C_theta_triads[:, triad_type, :]) * norm_fac, axis = -1), '--', label=r"$\sum_\theta \Re \left\{\mathcal{R}_{\mathcal{S}_\theta^U}\right\}$", color=plot_colours[-1])
    # ax3.set_xlim(run_data.time[0], run_data.time[-1])
    # ax3.grid()
    # ax3.legend()
    # ax3.set_xlabel(r"$t$", fontsize=label_size)
    # plt.savefig(cmdargs.out_dir + "/2d_validation" + "." + fig_format, format=fig_format, bbox_inches="tight", dpi=1200)
    # plt.close()


    # # -----------------------------------------
    # # # --------  The number of triads 
    # # -----------------------------------------
    wave_vec_indir = "/home/enda/PhD/2D-Navier-Stokes/Data/PostProcess/PhaseSync"
    with h5.File(wave_vec_indir + "/Wavevector_Data_N[128,128]_SECTORS[24,24]_KFRAC[0.00,0.20].h5") as infile:
        num_wv_02 = infile["NumWavevectors"][:, :]

    with h5.File(wave_vec_indir + "/Wavevector_Data_N[128,128]_SECTORS[24,24]_KFRAC[0.00,0.50].h5") as infile:
        num_wv_05 = infile["NumWavevectors"][:, :]

    with h5.File(wave_vec_indir + "/Wavevector_Data_N[128,128]_SECTORS[24,24]_KFRAC[0.00,0.80].h5") as infile:
        num_wv_08 = infile["NumWavevectors"][:, :]

    fig = plt.figure(figsize=(textwidth, 0.6 * textwidth))
    gs  = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.flipud(num_wv_02),  extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap=my_cmap, norm = mpl.colors.LogNorm())
    ax1.set_yticks(angticks)
    ax1.set_yticklabels(angtickLabels)
    ax1.set_xticks(angticks)
    ax1.set_xticklabels(angtickLabels)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_xlabel(r"$\tilde{\theta}$", fontsize=label_size)
    ax1.set_ylabel(r"$\theta$", fontsize=label_size)
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"Number of Terms")
    # ax1 = fig.add_subplot(gs[0, 1])
    # im1 = ax1.imshow(np.flipud(num_wv_05),  extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap=my_cmap, norm = mpl.colors.LogNorm())
    # ax1.set_yticks(angticks)
    # ax1.set_yticklabels(angtickLabels)
    # ax1.set_xticks(angticks)
    # ax1.set_xticklabels(angtickLabels)
    # ax1.tick_params(axis='both', which='major', labelsize=8)
    # ax1.set_xlabel(r"$\tilde{\theta}$", fontsize=label_size)
    # ax1.set_ylabel(r"$\theta$", fontsize=label_size)
    # div1  = make_axes_locatable(ax1)
    # cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    # cb1   = plt.colorbar(im1, cax = cbax1)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(np.flipud(num_wv_08),  extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap=my_cmap, norm = mpl.colors.LogNorm())
    ax1.set_yticks(angticks)
    ax1.set_yticklabels(angtickLabels)
    ax1.set_xticks(angticks)
    ax1.set_xticklabels(angtickLabels)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_xlabel(r"$\tilde{\theta}$", fontsize=label_size)
    ax1.set_ylabel(r"$\theta$", fontsize=label_size)
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"Number of Terms")

    with h5.File("/home/enda/PhD/2D-Navier-Stokes/Data/SectorSyncResults/NAV_AB4_FULL_N[128,128]_T[0.0,0.001,200.000]_NU[5e-13,1,4.0]_DRAG[0.1,1,0.0]_FORC[BODY_FORC_COS,2,1]_u0[RANDOM_ENRG]_TAG[Tag-S24-0.20]/Spectra_HDF_Data.h5") as infile:
        ## Initialize counter
        nn = 0
        for group in infile.keys():
            if "Iter" in group:
                nn+=1
        enst_flux_spectrum = np.zeros((nn, 92))
        nn=0
        # Read in the spectra
        for group in infile.keys():
            if "Iter" in group:
                if 'EnergyFluxSpectrum' in list(infile[group].keys()):
                    enst_flux_spectrum[nn, :] = infile[group]["EnstrophyFluxSpectrum"][:]
                nn+=1
    flux_spect = get_flux_spectrum(enst_flux_spectrum[:, 1:int(128/3)])
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(np.arange(1, int(128/3)), np.mean(flux_spect, axis = 0), color=plot_colours[-1])
    print(128, 128/3)
    ax2.axvline(x=int(128/3 * 0.2), linestyle=':', color='k', label=r"$k_{0.2}$" + "$ = {}$".format(int(128/3 * 0.2)))
    # ax2.axvline(x=int(128/3 * 0.5), linestyle=':', color='k', label=r"$k_{0.5}$" + "$ = {}$".format(int(128/3 * 0.5)))
    ax2.axvline(x=int(128/3 * 0.8), linestyle=':', color='k', label=r"$k_{0.8}$" + "$ = {}$".format(int(128/3 * 0.8)))
    ax2.set_ylabel(r"$\left \langle \Pi_k \right \rangle_t$", fontsize=label_size)
    ax2.set_xlabel(r"$k$", fontsize=label_size)
    ax2.set_xscale('log')
    ax2.set_xlim(1, int(128/3))
    ax2.legend()
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    plt.savefig(cmdargs.out_dir + "/2d_num_wavectors" + "." + fig_format, format=fig_format, bbox_inches="tight", dpi=1200)
    plt.close()