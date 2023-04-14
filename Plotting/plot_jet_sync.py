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
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, get_flux_spectrum
from plot_functions import plot_sector_phase_sync_snaps, plot_sector_phase_sync_snaps_full, plot_sector_phase_sync_snaps_full_sec, plot_sector_phase_sync_snaps_all
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

@njit
def jet_phase_order(phases, theta, kx, ky):

    ## Initialize variables
    xdim = phases.shape[0]
    kmax = int((xdim + 1) / 2)

    ## Initialize phase_order parameter
    phase_order = np.ones((len(theta), )) * np.complex(0., 0.)

    ## Loop through sectors of wavenumber space
    for t in range(int(len(theta) / 2)): # len(theta) - 1
        
        ## Initialize phase counter
        num_phase = 0

        ## Loop through phases
        for i in range(phases.shape[0]):
            for j in range(kmax):

                if ky[j] != 0:
                    ## Compute the polar coords
                    r     = kx[i]**2 + ky[j]**2
                    angle = np.arctan(kx[i] / ky[j])

                    ## If phase is in current sector update phase order parameter
                    if r <= kmax**2 and angle >= theta[t] and angle < theta[t + 1]:
                        phase_order[t]                        += np.exp(np.complex(0., 1.) * phases[(kmax - 1) + kx[i], (kmax - 1) + ky[j]])
                        phase_order[len(phase_order) - 1 - t] += np.exp(np.complex(0., 1.) * phases[(kmax - 1) - kx[i], (kmax - 1) + ky[j]])
                        num_phase      += 1

        ## Normalize the mean phase
        phase_order[t]                        /= num_phase
        phase_order[len(phase_order) - 1 - t] /= num_phase

    return phase_order

@njit
def get_normed_collective_phase(collect_phase, norm_const):

    ## Get dims
    s = collect_phase.shape
    r    = np.zeros(s, dtype='float64')
    phi  = np.zeros(s, dtype='float64')
    enst = np.zeros(s, dtype='float64')
    
    for i in range(s[0]):
        if norm_const[i] != 0.0:
            r[i]    = np.absolute(collect_phase[i] / norm_const[i])
            phi[i]  = np.angle(collect_phase[i] / norm_const[i])
            enst[i] = np.real(collect_phase[i])

    return r, phi, enst

@njit
def get_normed_collective_phase_2d(collect_phase, norm_const):

    ## Get dims
    s = collect_phase.shape
    r    = np.zeros(s, dtype='float64')
    phi  = np.zeros(s, dtype='float64')
    enst = np.zeros(s, dtype='float64')

    for i in range(s[0]):
        for j in range(s[1]):
            if norm_const[i, j] != 0.0:
                r[i, j]    = np.absolute(collect_phase[i, j] / norm_const[i, j])
                phi[i, j]  = np.angle(collect_phase[i, j] / norm_const[i, j])
                enst[i, j] = np.real(collect_phase[i, j])

    return r, phi, enst

@njit
def compute_pdf_t(counts, bin_width):

    num_t, nbins = counts.shape

    pdf = np.zeros(counts.shape)
    for t in range(num_t):
        pdf[t, :] = counts[t, :] / (np.sum(counts[t, :], axis=-1) * bin_width)

    return pdf
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

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars)

    if run_data.no_w:
        print("\nPreparing real space vorticity...", end = " ")
        for i in range(sys_vars.ndata):
            run_data.w[i, :, :] = np.fft.irfft2(run_data.w_hat[i, :, :])
        print("Finished!")
        
    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

    ## Read in post processing data
    post_data = import_post_processing_data(post_file_path, sys_vars, method)

    if cmdargs.full:
        flux_min = np.amin(post_data.enst_flux_per_sec[:, 0, :])
        flux_max = np.amax(post_data.enst_flux_per_sec[:, 0, :])
        flux_1d_min = np.amin(post_data.enst_flux_per_sec_1d[:, 0, :])
        flux_1d_max = np.amax(post_data.enst_flux_per_sec_1d[:, 0, :])
        flux_2d_min = np.amin(post_data.enst_flux_per_sec_2d[:, 0, :, :])
        # flux_2d_max = np.amax(post_data.enst_flux_per_sec_2d[:, 0, :, :])
        flux_2d_max = -flux_2d_min 

    print("Finished Reading in Data")
    # -----------------------------------------
    # # --------  Make Output Directories
    # -----------------------------------------
    ## Make output directory for snaps
    cmdargs.out_dir_info     = cmdargs.out_dir + "RUN_INFO/"
    cmdargs.out_dir_avg     = cmdargs.out_dir + "PHASE_SYNC_AVG_SNAPS/"
    cmdargs.out_dir_valid     = cmdargs.out_dir + "PHASE_SYNC_VALID_SNAPS/"
    cmdargs.out_dir_phases     = cmdargs.out_dir + "PHASE_SYNC_SNAPS/"
    cmdargs.out_dir_triads     = cmdargs.out_dir + "TRIAD_PHASE_SYNC_SNAPS/"
    if os.path.isdir(cmdargs.out_dir_phases) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_phases)
    if os.path.isdir(cmdargs.out_dir_info) != True:
        print("Making folder:" + tc.C + " RUN_INFO/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_info)
    if os.path.isdir(cmdargs.out_dir_triads) != True:
        print("Making folder:" + tc.C + " TRIAD_PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_triads)
    if os.path.isdir(cmdargs.out_dir_valid) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_VALID_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_valid)
    if os.path.isdir(cmdargs.out_dir_avg) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_AVG_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_avg)
    print("Run info Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_phases) + tc.Rst)
    print("Phases Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_phases) + tc.Rst)
    print("Triads Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_triads) + tc.Rst)
    print("Validation Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_valid) + tc.Rst)
    print("Time Avg Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_avg) + tc.Rst)
    ## Make subfolder for validation snaps dependent on parameters
    cmdargs.out_dir_valid_snaps = cmdargs.out_dir_valid + "SECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TAG[{}]/".format(post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, cmdargs.tag)
    cmdargs.out_dir_avg_snaps = cmdargs.out_dir_avg + "SECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TAG[{}]/".format(post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, cmdargs.tag)
    if os.path.isdir(cmdargs.out_dir_valid_snaps) != True:
        print("Making folder:" + tc.C + " SECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TAG[{}]/".format(post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, cmdargs.tag) + tc.Rst)
        os.mkdir(cmdargs.out_dir_valid_snaps)
    if os.path.isdir(cmdargs.out_dir_avg_snaps) != True:
        print("Making folder:" + tc.C + " SECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TAG[{}]/".format(post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, cmdargs.tag) + tc.Rst)
        os.mkdir(cmdargs.out_dir_avg_snaps)
    

    # -----------------------------------------
    # # --------  Plot Flux Spectrum
    # -----------------------------------------
    ##------------------------ Time Averaged Enstorphy Spectra and Flux Spectra
    fig = plt.figure(figsize = (21, 8))
    gs  = GridSpec(1, 2)
    ax2 = fig.add_subplot(gs[0, 0])
    for i in range(spec_data.enst_spectrum.shape[0]):
        ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enst_spectrum[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(spec_data.enst_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\mathcal{E}(|\mathbf{k}|)$: Enstrophy Spectrum")

    flux_spect = np.zeros((spec_data.enst_flux_spectrum.shape[0], int(sys_vars.Nx/3)))
    for i in range(spec_data.enst_flux_spectrum.shape[0]):
        flux_spect[i, :] = get_flux_spectrum(spec_data.enst_flux_spectrum[i, 1:int(sys_vars.Nx/3)])

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(spec_data.enst_flux_spectrum.shape[0]):
        ax2.plot(np.arange(1, int(sys_vars.Nx/3)), flux_spect[i, :], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(flux_spect, axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\Pi(|\mathbf{k}|)$: Enstrophy Flux Spectrum")
    
    plt.savefig(cmdargs.out_dir_info + "TimeAveraged_Enstrophy_EnstrophyFlux_Spectra.png")
    plt.close()


    # -----------------------------------------
    # # --------  Plot Validation 
    # -----------------------------------------
    ## Normalizing constants
    const_fac_sector = 4.0 * np.pi**2 
    const_fac = -0.5 #4.0 * np.pi**2 
    norm_fac  = (sys_vars.Nx * sys_vars.Ny)**2 #0.5 / (sys_vars.Nx * sys_vars.Ny)**3
    # NL and direct are off by -0.5 (Nx * Ny)^2 

    # Triad type all
    triad_type = int(0)

    ## Enstrophy Flux Compare --- Sum over sectors -> Direct w/ NL w/ Test -> 1D + 2D = All -> Complexification
    print("Plotting Validation Checks")
    fig = plt.figure(figsize = (21, 9))
    gs  = GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.sum(post_data.enst_flux_per_sec[:, triad_type, :], axis = -1) * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5, label=r"$\sum_\theta$E Flux Per Sector $\mathcal{S}_\theta^{U}$ Direct")
    ax1.plot(np.sum(post_data.enst_flux_C_theta[:, :], axis = -1), '*-', markevery=5, label=r"$\sum_\theta$ E Flux Per Sector $\mathcal{S}_\theta^{U}$ NL")
    if hasattr(post_data, "enst_flux_test"):
        ax1.plot(post_data.enst_flux_test[:, triad_type] * const_fac * norm_fac, '--', label=r"Enstrophy Flux $\mathcal{S}_\theta$ Direct (Test)")
    ax1.set_title(r"Compare Flux: Sum over Sectors: Totals (Type 0)")
    ax1.grid()
    ax1.legend()
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.sum(post_data.enst_flux_per_sec[:, triad_type, :], axis=-1) * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5, label=r"E Flux Per Sector $\mathcal{S}_\theta^{U}$ Direct")
    ax2.plot(np.sum(post_data.enst_flux_C_theta[:, :], axis=-1), '*-', markevery=5, label=r" E Flux Per Sector $\mathcal{S}_\theta^{U}$ NL")
    ax2.plot(np.sum(post_data.enst_flux_per_sec_1d[:, triad_type, :] + np.sum(post_data.enst_flux_per_sec_2d[:, triad_type, :, :], axis = -1), axis=-1) * const_fac * norm_fac, '--', label=r"1D + 2D Direct")
    ax2.set_title(r"Compare Flux: 1D + 2D = All - Direct w/ NL")
    ax2.grid()
    ax2.legend()
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.sum(post_data.enst_flux_per_sec[:, triad_type, :], axis = -1) * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5, label=r"$\sum_\theta\Pi_{\mathcal{S}_\theta^{U}}$ Direct")
    ax3.plot(np.sum(np.real(post_data.phase_order_C_theta[:, :]), axis = -1), '*-', markevery=5, label=r"$\sum_\theta \Re\{Re^{\Phi}\}$ NL")
    ax3.plot(np.sum(np.real(post_data.phase_order_C_theta_triads[:, triad_type, :]) * norm_fac / 2, axis = -1), '--', label=r"$\sum_\theta \Re\{\mathcal{R}_\theta\}$ Direct")
    ax3.set_title(r"Compare Complexification: Sum over Sectors: Totals (Type 0)")
    ax3.grid()
    ax3.legend()
    plt.suptitle("Sum over Theta")
    plt.savefig(cmdargs.out_dir_valid_snaps + "/EnstrophyFlux_SumOverSectors_Compare.png", bbox_inches="tight")
    plt.close()

    ## Enstrophy Flux Compare --- Each sector -> Direct w/ NL -> 1D + 2D = All -> Complexification
    for k3 in range(post_data.num_k3_sect):
        print("Validation Check - Sector {}/{}".format(k3 + 1, post_data.num_k3_sect))
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(post_data.enst_flux_per_sec[:, triad_type, k3] * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5, label=r"E Flux Per Sector $\mathcal{S}_\theta^{U}$ Direct")
        ax1.plot(post_data.enst_flux_C_theta[:, k3], '*-', markevery=5, label=r"E Flux Per Sector $\mathcal{S}_\theta^{U}$ NL")
        ax1.set_title(r"Comparing Flux: Direct w/ NL: Sector {}".format(k3 + 1))
        ax1.grid()
        ax1.legend()
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(post_data.enst_flux_per_sec[:, triad_type, k3] * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5,  label=r"E Flux Per Sector $\mathcal{S}_\theta^{U}$ Direct")
        ax2.plot(post_data.enst_flux_C_theta[:, k3], '*-', markevery=5, label=r"E Flux Per Sector $\mathcal{S}_\theta^{U}$ NL")
        ax2.plot(post_data.enst_flux_per_sec_1d[:, triad_type, k3] * const_fac * norm_fac + np.sum(post_data.enst_flux_per_sec_2d[:, triad_type, k3, :], axis = -1) * const_fac * norm_fac, '--', label=r"1D + 2D Direct")
        ax2.set_title(r"Compare Flux: 1D + 2D = All - Direct w/ NL: Sector {}".format(k3 + 1))
        ax2.grid()
        ax2.legend()
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(post_data.enst_flux_per_sec[:, triad_type, k3] * const_fac * norm_fac, '-', marker='o', markersize=10, markevery=5, label=r"$\Pi_{\mathcal{S}_\theta^{U}}$")
        ax3.plot(np.real(post_data.phase_order_C_theta[:, k3]), '*-', markevery=5, label=r"$\Re\{Re^{\Phi}\}$ NL")
        ax3.plot(np.real(post_data.phase_order_C_theta_triads[:, triad_type, k3]) * norm_fac / 2,'--', label=r"$\Re\{\mathcal{R}_\theta\}$ Direct")
        ax3.set_title(r"Compare Complexification: Sector {}".format(k3 + 1))
        ax3.grid()
        ax3.legend()
        plt.suptitle("Sector {}".format(k3 + 1))
        plt.savefig(cmdargs.out_dir_valid_snaps + "/EnstrophyFlux_ComparePerk3Sector_k3[{}].png".format(k3 + 1), bbox_inches="tight")
        plt.close()

    # # -----------------------------------------
    # # # --------  Plot Time Averaged Data
    # # -----------------------------------------
    theta_k3 = post_data.theta_k3
    dtheta_k3 = theta_k3[1] - theta_k3[0]
    angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
    angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
    angtickLabels_alpha = np.linspace(0, len(theta_k3) + 1, num = len(angtickLabels), endpoint = False, dtype = "int64").tolist()
    theta_k3_min     = -np.pi/2 - dtheta_k3 / 2
    theta_k3_max     = np.pi/2 + dtheta_k3 /2
    alpha_min = theta_k3_min
    alpha_max = theta_k3_max

    my_magma = mpl.colors.ListedColormap(cm.magma.colors[::-1])
    my_magma.set_under(color = "white")
    my_magma.set_over(color = "white")
    my_hsv = mpl.cm.hsv
    my_hsv.set_under(color = "white")
    my_hsv.set_over(color = "white")
    my_jet = mpl.cm.jet
    my_jet.set_under(color = "white") 
    my_jet.set_over(color = "white") 

    # print("Plotting Data")
    # try_type = 0

    for try_type in range(post_data.triad_R_2d.shape[1] -1):  ## Ignore the last type as that is always 0
        print("Plotting Triad Type {}".format(try_type))

        #####################################################################
        ## NORMAL PHASE ORDER PARAMETER ---- Time Averaged ---- 2D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.flipud(np.mean(post_data.triad_R_2d[:, try_type, :, :], axis=0)), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_xticks(angticks)
        ax6.set_xticklabels(angtickLabels_alpha)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_ylabel(r"$\theta$")
        ax6.set_xlabel(r"$\alpha$")
        ax6.set_title(r"Sync Across Sectors (2D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$\mathcal{R}^{2D}$")
        ax7 = fig.add_subplot(gs[0, 1])
        im7 = ax7.imshow(np.flipud(np.mean(np.mod(post_data.triad_Phi_2d[:, try_type, :, :] + 2.0 * np.pi, 2.0 * np.pi), axis=0)), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_jet, vmin = 0.0, vmax = 2.0 * np.pi)
        ax7.set_xticks(angticks)
        ax7.set_xticklabels(angtickLabels_alpha)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_ylabel(r"$\theta$")
        ax7.set_xlabel(r"$\alpha$")
        ax7.set_title(r"Average Angle Across Sectors (2D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        # cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        # cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
        cb7.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        cb7.set_label(r"$\Phi^{2D}$")
        ax8 = fig.add_subplot(gs[0, 2])
        data = np.flipud(np.mean(post_data.enst_flux_per_sec_2d[:, try_type, :, :], axis=0))
        im8 = ax8.imshow(data, extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = np.min(data), vmax = np.absolute(np.min(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_xticks(angticks)
        ax8.set_xticklabels(angtickLabels_alpha)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_ylabel(r"$\theta$")
        ax8.set_xlabel(r"$\alpha$")
        ax8.set_title(r"Enstrophy Flux Across Sectors (2D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Pi_{\mathcal{S}_\theta}^{2D}$")
        plt.suptitle(r"Time Averaged 2D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_NORMAL_Sync_2D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## NORMAL PHASE ORDER PARAMETER ---- Time Averaged ---- 2D + 1D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.flipud(np.mean(post_data.triad_R_2d[:, try_type, :, :], axis=0) + np.diag(np.mean(post_data.triad_R_1d[:, try_type, :], axis=0))), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_xticks(angticks)
        ax6.set_xticklabels(angtickLabels_alpha)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_ylabel(r"$\theta$")
        ax6.set_xlabel(r"$\alpha$")
        ax6.set_title(r"Sync Across Sectors (2D and 1D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$\mathcal{R}^{2D}$")
        ax7 = fig.add_subplot(gs[0, 1])
        im7 = ax7.imshow(np.flipud(np.mean(np.mod(post_data.triad_Phi_2d[:, try_type, :, :] + 2.0 * np.pi, 2.0 * np.pi), axis=0) + np.diag(np.mean(np.mod(post_data.triad_Phi_1d[:, try_type, :] + 2.0*np.pi, 2.0*np.pi), axis=0))), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_jet, vmin = 0.0, vmax = 2.0 * np.pi)
        ax7.set_xticks(angticks)
        ax7.set_xticklabels(angtickLabels_alpha)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_ylabel(r"$\theta$")
        ax7.set_xlabel(r"$\alpha$")
        ax7.set_title(r"Average Angle Across Sectors (2D and 1D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        # cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        # cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
        cb7.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        cb7.set_label(r"$\Phi^{2D}$")
        ax8 = fig.add_subplot(gs[0, 2])
        data = np.flipud(np.mean(post_data.enst_flux_per_sec_2d[:, try_type, :, :], axis=0) + np.diag(np.mean(post_data.enst_flux_per_sec_1d[:, try_type, :], axis=0)))
        im8 = ax8.imshow(data, extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = np.min(data), vmax = np.absolute(np.min(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_xticks(angticks)
        ax8.set_xticklabels(angtickLabels_alpha)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_ylabel(r"$\theta$")
        ax8.set_xlabel(r"$\alpha$")
        ax8.set_title(r"Enstrophy Flux Across Sectors (2D and 1D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Pi_{\mathcal{S}_\theta}^{2D}$")
        plt.suptitle(r"Time Averaged 2D and 1 Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_NORMAL_Sync_2Dand1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## NORMAL PHASE ORDER PARAMETER ---- Time Averaged ---- 1D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 2)
        ax3 = fig.add_subplot(gs[0, 0])
        div3   = make_axes_locatable(ax3)
        axtop3 = div3.append_axes("top", size = "100%", pad = 0.2)
        axtop3.plot(theta_k3, np.mean(np.mod(post_data.triad_Phi_1d[:, try_type, :] + 2.0 * np.pi, 2.0 * np.pi), axis=0), '.-', color = "orange")
        axtop3.set_xlim(theta_k3_min, theta_k3_max)
        axtop3.set_xticks(angticks)
        axtop3.set_xticklabels([])
        axtop3.set_ylim(0.0, 2.0 * np.pi)
        axtop3.set_yticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
        axtop3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        axtop3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
        axtop3.set_title(r"Order Parameters (1D)")
        axtop3.set_ylabel(r"$\Phi^{1D}$")
        ax3.plot(theta_k3, np.mean(post_data.triad_R_1d[:, try_type, :], axis=0))
        ax3.set_xlim(theta_k3_min, theta_k3_max)
        ax3.set_xticks(angticks)
        ax3.set_xticklabels(angtickLabels)
        ax3.set_ylim(0 - 0.05, 1 + 0.05)
        ax3.set_xlabel(r"$\theta$")
        ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
        ax3.set_ylabel(r"$\mathcal{R}^{1D}$")
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.plot(theta_k3, np.mean(post_data.enst_flux_per_sec_1d[:, try_type, :], axis=0), '.-')
        ax4.set_xlim(theta_k3_min, theta_k3_max)
        ax4.set_xticks(angticks)
        ax4.set_xticklabels(angtickLabels)
        ax4.set_xlabel(r"$\theta$")
        ax4.set_ylabel(r"$\Pi_{\mathcal{S}_\theta}^{1D}$")
        ax4.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
        ax4.set_title(r"Enstrophy Flux Per Sector (1D)")
        plt.suptitle(r"Time Averaged 1D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_NORMAL_Sync_1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## NORMAL PHASE ORDER PARAMETER ---- SpaceTime Plots ---- 1D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 13))
        gs  = GridSpec(3, 1, hspace = 0.25)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.rot90(np.flipud(post_data.triad_R_1d[:, try_type, :]), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_xlabel(r"$t$")
        ax6.set_ylabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector (1D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "5%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$\mathcal{R}^{1D}$")
        ax7 = fig.add_subplot(gs[1, 0])
        im7 = ax7.imshow(np.rot90(np.flipud(post_data.triad_Phi_1d[:, try_type, :]), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_hsv, vmin = -np.pi, vmax = np.pi)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_xlabel(r"$t$")
        ax7.set_ylabel(r"$\theta$")
        ax7.set_title(r"Sync Per Sector (1D)")
        ax7.set_title(r"Average Angle Per Sector (1D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "5%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_label(r"$\Phi^{1D}$")
        ax8 = fig.add_subplot(gs[2, 0])
        data = post_data.enst_flux_per_sec_1d[:, try_type, :]
        im8 = ax8.imshow(np.rot90(np.flipud(post_data.enst_flux_per_sec_1d[:, try_type, :]), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_xlabel(r"$t$")
        ax8.set_ylabel(r"$\theta$")
        ax8.set_title(r"Sync Per Sector (1D)")
        ax8.set_title(r"Enstrophy Flux Per Sector (1D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "5%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Pi_{\mathcal{S}_\theta}^{1D}$")
        plt.suptitle(r"Spacetime 1D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_NORMAL_SpaceTime_1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## NORMAL PHASE ORDER PARAMETER ---- SpaceTime Plots ---- All Data
        #####################################################################
        fig = plt.figure(figsize = (21, 13))
        gs  = GridSpec(3, 1, hspace = 0.25)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.rot90(np.flipud(post_data.triad_R[:, try_type, :]), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_xlabel(r"$t$")
        ax6.set_ylabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$\mathcal{R}^{1D}$")
        ax7 = fig.add_subplot(gs[1, 0])
        im7 = ax7.imshow(np.rot90(np.flipud(post_data.triad_Phi[:, try_type, :]), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_hsv, vmin = -np.pi, vmax = np.pi)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_ylabel(r"$\theta$")
        ax7.set_xlabel(r"$t$")
        ax7.set_title(r"Average Angle Per Sector")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_label(r"$\Phi^{1D}$")
        ax8 = fig.add_subplot(gs[2, 0])
        data = np.rot90(np.flipud(post_data.enst_flux_per_sec[:, try_type, :]), k=-1)
        im8 = ax8.imshow(data, aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_ylabel(r"$\theta$")
        ax8.set_xlabel(r"$t$")
        ax8.set_title(r"Enstrophy Flux Per Sector")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Pi_{\mathcal{S}_\theta}^{1D}$")
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_NORMAL_SpaceTime_PerSector.png".format(try_type), bbox_inches="tight")
        plt.close()
        #
        #
        #
        # ----------------------------------------------------------- COLLECTIVE PHASE ORDER PARAMETER
        #
        #
        # 
        if post_data.phase_order_normed_const:
            phase_order_R, phase_order_Phi, phase_order_enst          = np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect))
            phase_order_R_1D, phase_order_Phi_1D, phase_order_enst_1D = np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect))
            phase_order_R_2D, phase_order_Phi_2D, phase_order_enst_2D = np.zeros((sys_vars.ndata, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect, post_data.num_k1_sect))
            for i in range(sys_vars.ndata):
                phase_order_R[i, :], phase_order_Phi[i, :], phase_order_enst[i, :]                  = get_normed_collective_phase(post_data.phase_order_C_theta_triads[i, try_type, :], np.sum(post_data.phase_order_norm_const[i, 0, try_type, :, :], axis=-1))
                phase_order_R_1D[i,:], phase_order_Phi_1D[i, :], phase_order_enst_1D[i, :]          = get_normed_collective_phase(post_data.phase_order_C_theta_triads_1d[i, try_type, :], np.diag(post_data.phase_order_norm_const[i, 0, try_type, :, :]))
                phase_order_R_2D[i,:, :], phase_order_Phi_2D[i, :, :], phase_order_enst_2D[i, :, :] = get_normed_collective_phase_2d(post_data.phase_order_C_theta_triads_2d[i, try_type, :, :], post_data.phase_order_norm_const[i, 0, try_type, :, :])
        else:
            phase_order_R_2D    = np.absolute(post_data.phase_order_C_theta_triads_2d[:, try_type, :, :])
            phase_order_Phi_2D  = np.angle(post_data.phase_order_C_theta_triads_2d[:, try_type, :, :])
            phase_order_enst_2D = np.real(post_data.phase_order_C_theta_triads_2d[:, try_type, :]) 
            phase_order_R_1D    = np.absolute(post_data.phase_order_C_theta_triads_1d[:, try_type, :])
            phase_order_Phi_1D  = np.angle(post_data.phase_order_C_theta_triads_1d[:, try_type, :])
            phase_order_enst_1D = np.real(post_data.phase_order_C_theta_triads_1d[:, try_type, :])
            phase_order_R       = np.absolute(post_data.phase_order_C_theta_triads[:, try_type, :])
            phase_order_Phi     = np.angle(post_data.phase_order_C_theta_triads[:, try_type, :])
            phase_order_enst    = np.real(post_data.phase_order_C_theta_triads[:, try_type, :])


        #####################################################################
        ## COLLECTIVE PHASE ORDER PARAMETER ---- Time Averaged ---- 2D Data
        #####################################################################
        ## Collective phase order parameter
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.flipud(np.mean(np.mod(phase_order_R_2D + 2.0 * np.pi, 2.0 * np.pi), axis=0)), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_xticks(angticks)
        ax6.set_xticklabels(angtickLabels_alpha)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_ylabel(r"$\theta$")
        ax6.set_xlabel(r"$\alpha$")
        ax6.set_title(r"Sync Across Sectors (2D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$|\mathcal{R}^{2D}|$")
        ax7 = fig.add_subplot(gs[0, 1])
        im7 = ax7.imshow(np.flipud(np.mean(np.mod(phase_order_Phi_2D + 2.0 * np.pi, 2.0 * np.pi), axis=0)), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_jet, vmin = 0.0, vmax = 2.0 * np.pi)
        ax7.set_xticks(angticks)
        ax7.set_xticklabels(angtickLabels_alpha)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_ylabel(r"$\theta$")
        ax7.set_xlabel(r"$\alpha$")
        ax7.set_title(r"Average Angle Across Sectors (2D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        # cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        # cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
        cb7.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        cb7.set_label(r"$\arg\{\mathcal{R}^{2D} \}$")
        ax8 = fig.add_subplot(gs[0, 2])
        data = np.flipud(np.mean(phase_order_enst_2D, axis=0))
        im8 = ax8.imshow(data, extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = np.min(data), vmax = np.absolute(np.min(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_xticks(angticks)
        ax8.set_xticklabels(angtickLabels_alpha)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_ylabel(r"$\theta$")
        ax8.set_xlabel(r"$\alpha$")
        ax8.set_title(r"Enstrophy Flux Across Sectors (2D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Re\{\mathcal{R}^{2D} \}$")
        plt.suptitle(r"Time Averaged Collective Phase Order 2D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_COLLECTIVE_Sync_2D.png".format(try_type), bbox_inches="tight")
        plt.close()
        
        #####################################################################
        ## COLLECTIVE PHASE ORDER PARAMETER ---- Time Averaged ---- 1D + 2D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.flipud(np.mean(phase_order_R_2D, axis=0) + np.diag(np.mean(phase_order_R_1D, axis=0))), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_xticks(angticks)
        ax6.set_xticklabels(angtickLabels_alpha)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_ylabel(r"$\theta$")
        ax6.set_xlabel(r"$\alpha$")
        ax6.set_title(r"Sync Across Sectors (2D and 1D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$|\mathcal{R}^{2D} |$")
        ax7 = fig.add_subplot(gs[0, 1])
        im7 = ax7.imshow(np.flipud(np.mean(np.mod(phase_order_Phi_2D + 2.0 * np.pi, 2.0 * np.pi), axis=0) + np.diag(np.mean(np.mod(phase_order_Phi_1D + 2.0 * np.pi, 2.0 * np.pi), axis=0))), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_jet, vmin = 0.0, vmax = 2.0 * np.pi)
        ax7.set_xticks(angticks)
        ax7.set_xticklabels(angtickLabels_alpha)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_ylabel(r"$\theta$")
        ax7.set_xlabel(r"$\alpha$")
        ax7.set_title(r"Average Angle Across Sectors (2D and 1D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        # cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        # cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
        cb7.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        cb7.set_label(r"$\arg\{\mathcal{R}^{2D} \}$")
        ax8 = fig.add_subplot(gs[0, 2])
        data = np.flipud(np.mean(phase_order_enst_2D, axis=0) + np.diag(np.mean(phase_order_enst_1D, axis=0)))
        im8 = ax8.imshow(data, extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = np.min(data), vmax = np.absolute(np.min(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_xticks(angticks)
        ax8.set_xticklabels(angtickLabels_alpha)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_ylabel(r"$\theta$")
        ax8.set_xlabel(r"$\alpha$")
        ax8.set_title(r"Enstrophy Flux Across Sectors (2D and 1D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Re\{\mathcal{R}^{2D}\}$")
        plt.suptitle(r"Time Averaged -- Collective Phase Order -- 2D and 1 Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_COLLECTIVE_Sync_2Dand1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## COLLECTIVE PHASE ORDER PARAMETER ---- Time Averaged ---- 1D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 2)
        ax3 = fig.add_subplot(gs[0, 0])
        div3   = make_axes_locatable(ax3)
        axtop3 = div3.append_axes("top", size = "100%", pad = 0.2)
        axtop3.plot(theta_k3, np.mean(np.mod(phase_order_Phi_1D + 2.0 * np.pi, 2.0 * np.pi), axis=0), '.-', color = "orange")
        axtop3.set_xlim(theta_k3_min, theta_k3_max)
        axtop3.set_xticks(angticks)
        axtop3.set_xticklabels([])
        axtop3.set_ylim(0.0, 2.0 * np.pi)
        axtop3.set_yticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
        axtop3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        axtop3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
        axtop3.set_title(r"Order Parameters (1D)")
        axtop3.set_ylabel(r"$\arg \{\mathcal{R}^{1D} \}$")
        ax3.plot(theta_k3, np.mean(phase_order_R_1D, axis=0))
        ax3.set_xlim(theta_k3_min, theta_k3_max)
        ax3.set_xticks(angticks)
        ax3.set_xticklabels(angtickLabels)
        ax3.set_ylim(0 - 0.05, 1 + 0.05)
        ax3.set_xlabel(r"$|\mathcal{R}^{1D}|$")
        ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
        ax3.set_ylabel(r"$\mathcal{R}^{1D}$")
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.plot(theta_k3, np.mean(phase_order_enst_1D, axis=0), '.-')
        ax4.set_xlim(theta_k3_min, theta_k3_max)
        ax4.set_xticks(angticks)
        ax4.set_xticklabels(angtickLabels)
        ax4.set_xlabel(r"$\theta$")
        ax4.set_ylabel(r"$\Re\{\mathcal{R}^{1D} \}$")
        ax4.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
        ax4.set_title(r"Enstrophy Flux Per Sector (1D)")
        plt.suptitle(r"Time Averaged 1D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_COLLECTIVE_Sync_1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## COLLECTIVE PHASE ORDER PARAMETER ---- Spacetime Plots ---- 1D Data
        #####################################################################
        fig = plt.figure(figsize = (21, 13))
        gs  = GridSpec(3, 1, hspace=0.25)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(np.rot90(np.flipud(phase_order_R_1D), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_xlabel(r"$t$")
        ax6.set_ylabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector (1D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "5%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$|\mathcal{R}^{1D}|$")
        ax7 = fig.add_subplot(gs[1, 0])
        im7 = ax7.imshow(np.rot90(np.flipud(phase_order_Phi_1D), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_hsv, vmin = -np.pi, vmax = np.pi)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_xlabel(r"$t$")
        ax7.set_ylabel(r"$\theta$")
        ax7.set_title(r"Sync Per Sector (1D)")
        ax7.set_title(r"Average Angle Per Sector (1D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "5%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_label(r"$\arg\{\mathcal{R}^{1D} \}$")
        ax8 = fig.add_subplot(gs[2, 0])
        data = np.rot90(np.flipud(phase_order_enst_1D), k=-1)
        im8 = ax8.imshow(data, aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_xlabel(r"$t$")
        ax8.set_ylabel(r"$\theta$")
        ax8.set_title(r"Sync Per Sector (1D)")
        ax8.set_title(r"Enstrophy Flux Per Sector (1D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "5%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Re\{\mathcal{R}^{1D} \}$")
        plt.suptitle(r"Spacetime Collective Phase 1D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_COLLECTIVE_SpaceTime_1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        #####################################################################
        ## COLLECTIVE PHASE ORDER PARAMETER ---- SpaceTime Plots ---- All Data
        #####################################################################
        fig = plt.figure(figsize = (21, 13))
        gs  = GridSpec(3, 1, hspace=0.25)
        ax6 = fig.add_subplot(gs[0, 0])
        if try_type == 0:
            v_max = 0.2
        else:
            v_max = 1.0
        im6 = ax6.imshow(np.rot90(np.flipud(phase_order_R), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_magma, vmin = 0.0, vmax = v_max)
        ax6.set_yticks(angticks)
        ax6.set_yticklabels(angtickLabels)
        ax6.set_xlabel(r"$t$")
        ax6.set_ylabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "5%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$| \mathcal{R}_\theta |$")
        ax7 = fig.add_subplot(gs[1, 0])
        im7 = ax7.imshow(np.rot90(np.flipud(phase_order_Phi), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_hsv, vmin = -np.pi, vmax = np.pi)
        ax7.set_yticks(angticks)
        ax7.set_yticklabels(angtickLabels)
        ax7.set_ylabel(r"$\theta$")
        ax7.set_title(r"Average Angle Per Sector")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "5%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_label(r"$\arg \{ \mathcal{R}_\theta \}$")
        ax8 = fig.add_subplot(gs[2, 0])
        data = np.rot90(np.flipud(phase_order_enst), k=-1)
        im8 = ax8.imshow(data, aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_yticks(angticks)
        ax8.set_yticklabels(angtickLabels)
        ax8.set_ylabel(r"$\theta$")
        ax8.set_title(r"Enstrophy Flux Per Sector")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "5%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Re \{ \mathcal{R}_\theta \}$")
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_COLLECTIVE_SpaceTime_PerSector.png".format(try_type), bbox_inches="tight")
        plt.close()
        

        # # -----------------------------------------
        # # # --------  Plot In Time Stats Data
        # # -----------------------------------------
        # Plot in time stats
        n_k3       = 0
        for class_type in range(2):
            print("In Time Stats: Class {} Type {} Sector {}".format(class_type, try_type, n_k3))

            #####################################################################
            ## IN TIME STATS -- ALL POSSIBLE TRIADS
            #####################################################################
            if post_data.all_triads_pdf_t and post_data.all_wghtd_triads_pdf_t:
                if n_k3 == 0:
                    indxs = [tt, tt + 10, tt + 100]
                    print(post_data.all_triads_pdf_ranges_t[class_type, try_type, :], post_data.all_triads_pdf_counts_t[indxs, class_type, try_type, :])
                    tt = 10
                    fig = plt.figure(figsize = (21, 9))
                    gs  = GridSpec(1, 2)
                    bin_centres = (post_data.all_triads_pdf_ranges_t[class_type, try_type, 1:] + post_data.all_triads_pdf_ranges_t[class_type, try_type, :-1]) * 0.5
                    bin_width = post_data.all_triads_pdf_ranges_t[class_type, try_type, 1] - post_data.all_triads_pdf_ranges_t[class_type, try_type, 0]
                    ax8 = fig.add_subplot(gs[0, 0])
                    for i in range(len(indxs)):
                        pdf = compute_pdf_t(post_data.all_triads_pdf_counts_t[indxs[i], class_type, try_type, :], bin_width)
                        ax8.plot(bin_centres, pdf, label="t = {}".format(i))
                    ax8.legend()
                    bin_centres = (post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, 1:] + post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, :-1]) * 0.5
                    bin_width = post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, 1] - post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, 0]
                    pdf = compute_pdf_t(post_data.all_wghtd_triads_pdf_counts_t[indxs, class_type, try_type, :], bin_width)
                    ax8 = fig.add_subplot(gs[0, 1])
                    for i in range(len(indxs)):
                        ax8.plot(bin_centres, pdf[i], label="t = {}".format(i))
                    ax8.legend()
                    plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_ALL_Triad_HISTOGRAM_InTime_Class[{}].png".format(try_type, class_type), bbox_inches="tight")
                    plt.close()

                fig = plt.figure(figsize = (21, 9))
                gs  = GridSpec(1, 2)
                ax8 = fig.add_subplot(gs[0, 0])
                bin_centres = (post_data.all_triads_pdf_ranges_t[class_type, try_type, 1:] + post_data.all_triads_pdf_ranges_t[class_type, try_type, :-1]) * 0.5
                bin_width = post_data.all_triads_pdf_ranges_t[class_type, try_type, 1] - post_data.all_triads_pdf_ranges_t[class_type, try_type, 0]
                pdf = compute_pdf_t(post_data.all_triads_pdf_counts_t[:, class_type, try_type, :], bin_width)
                im8 = ax8.imshow(np.flipud(pdf), aspect='auto', extent = (-np.pi + dtheta_k3/2, np.pi - dtheta_k3/2, 1, sys_vars.ndata), cmap = my_magma) # , , norm=mpl.colors.LogNorm()
                ax8.set_xticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
                ax8.set_xticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
                ax8.set_ylabel(r"$t$")
                if class_type == 0:
                    ax8.set_xlabel(r"$\varphi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                else:
                    ax8.set_xlabel(r"$\Phi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                ax8.set_title(r"All Triads PDF in Time")
                div8  = make_axes_locatable(ax8)
                cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
                cb8   = plt.colorbar(im8, cax = cbax8)
                cb8.set_label(r"PDF")
                ax8 = fig.add_subplot(gs[0, 1])
                bin_centres = (post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, 1:] + post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, :-1]) * 0.5
                bin_width = post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, 1] - post_data.all_wghtd_triads_pdf_ranges_t[class_type, try_type, 0]
                pdf = compute_pdf_t(post_data.all_wghtd_triads_pdf_counts_t[:, class_type, try_type, :], bin_width)
                im8 = ax8.imshow(np.flipud(pdf), aspect='auto', extent = (-np.pi + dtheta_k3/2, np.pi - dtheta_k3/2, 1, sys_vars.ndata), cmap = my_magma) # , norm=mpl.colors.LogNorm()
                ax8.set_xticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
                ax8.set_xticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
                ax8.set_ylabel(r"$t$")
                if class_type == 0:
                    ax8.set_xlabel(r"$\varphi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                else:
                    ax8.set_xlabel(r"$\Phi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                ax8.set_title(r"Weight Flux Density All Triads PDF in Time")
                div8  = make_axes_locatable(ax8)
                cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
                cb8   = plt.colorbar(im8, cax = cbax8)
                cb8.set_label(r"wPDF")
                if class_type == 0:
                    plt.suptitle("All Possible Triads - PDF In Time: Normal Triads; Triad Type {}; Sector {}".format(try_type, n_k3))
                else:
                    plt.suptitle("All Possible Triads - PDF In Time: Generalized Triads; Triad Type {}; Sector {}".format(try_type, n_k3))
                plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_ALL_TriadPDF_InTime_Class[{}]_Sec[{}].png".format(try_type, class_type, n_k3), bbox_inches="tight")
                plt.close()




            #####################################################################
            ## IN TIME STATS -- OVER SECTORS
            #####################################################################
            if post_data.triads_all_pdf_t or post_data.wghtd_all_pdf_t:
                fig = plt.figure(figsize = (21, 9))
                gs  = GridSpec(1, 2)
                ax8 = fig.add_subplot(gs[0, 0])
                bin_centres = (post_data.triads_all_pdf_ranges_t[1:] + post_data.triads_all_pdf_ranges_t[:-1]) * 0.5
                bin_width = post_data.triads_all_pdf_ranges_t[1] - post_data.triads_all_pdf_ranges_t[0]
                pdf = compute_pdf_t(post_data.triads_all_pdf_counts_t[:, class_type, try_type, n_k3, :], bin_width)
                im8 = ax8.imshow(np.flipud(pdf), aspect='auto', extent = (-np.pi + dtheta_k3/2, np.pi - dtheta_k3/2, 1, sys_vars.ndata), cmap = my_magma) # , norm=mpl.colors.LogNorm()
                ax8.set_xticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
                ax8.set_xticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
                ax8.set_ylabel(r"$t$")
                if class_type == 0:
                    ax8.set_xlabel(r"$\varphi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                else:
                    ax8.set_xlabel(r"$\Phi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                ax8.set_title(r"Triad PDF in Time")
                div8  = make_axes_locatable(ax8)
                cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
                cb8   = plt.colorbar(im8, cax = cbax8)
                cb8.set_label(r"PDF")
                ax8 = fig.add_subplot(gs[0, 1])
                bin_centres = (post_data.wghtd_triads_all_pdf_ranges_t[1:] + post_data.wghtd_triads_all_pdf_ranges_t[:-1]) * 0.5
                bin_width = post_data.wghtd_triads_all_pdf_ranges_t[1] - post_data.wghtd_triads_all_pdf_ranges_t[0]
                pdf = compute_pdf_t(post_data.wghtd_triads_all_pdf_counts_t[:, class_type, try_type, n_k3, :], bin_width)
                im8 = ax8.imshow(np.flipud(pdf), aspect='auto', extent = (-np.pi + dtheta_k3/2, np.pi - dtheta_k3/2, 1, sys_vars.ndata), cmap = my_magma) # , norm=mpl.colors.LogNorm()
                ax8.set_xticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
                ax8.set_xticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
                ax8.set_ylabel(r"$t$")
                if class_type == 0:
                    ax8.set_xlabel(r"$\varphi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                else:
                    ax8.set_xlabel(r"$\Phi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                ax8.set_title(r"Weight Flux Density Triad PDF in Time")
                div8  = make_axes_locatable(ax8)
                cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
                cb8   = plt.colorbar(im8, cax = cbax8)
                cb8.set_label(r"wPDF")
                if class_type == 0:
                    plt.suptitle("All Contributions - PDF In Time: Normal Triads; Triad Type {}; Sector {}".format(try_type, n_k3))
                else:
                    plt.suptitle("All Contributions - PDF In Time: Generalized Triads; Triad Type {}; Sector {}".format(try_type, n_k3))
                plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_TriadPDF_All_InTime_Class[{}]_Sec[{}].png".format(try_type, class_type, n_k3), bbox_inches="tight")
                plt.close()

            if post_data.triads_1d_pdf_t or post_data.wghtd_1d_pdf_t:
                fig = plt.figure(figsize = (21, 9))
                gs  = GridSpec(1, 2)
                ax8 = fig.add_subplot(gs[0, 0])
                bin_centres = (post_data.triads_1d_pdf_ranges_t[1:] + post_data.triads_1d_pdf_ranges_t[:-1]) * 0.5
                bin_width = post_data.triads_1d_pdf_ranges_t[1] - post_data.triads_1d_pdf_ranges_t[0]
                pdf = compute_pdf_t(post_data.triads_1d_pdf_counts_t[:, class_type, try_type, n_k3, :], bin_width)
                im8 = ax8.imshow(np.flipud(pdf), aspect='auto', extent = (-np.pi + dtheta_k3/2, np.pi - dtheta_k3/2, 1, sys_vars.ndata), cmap = my_magma) # , norm=mpl.colors.LogNorm()
                ax8.set_xticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
                ax8.set_xticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
                ax8.set_ylabel(r"$t$")
                if class_type == 0:
                    ax8.set_xlabel(r"$\varphi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                else:
                    ax8.set_xlabel(r"$\Phi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                ax8.set_title(r"Triad PDF in Time")
                div8  = make_axes_locatable(ax8)
                cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
                cb8   = plt.colorbar(im8, cax = cbax8)
                cb8.set_label(r"PDF")
                ax8 = fig.add_subplot(gs[0, 1])
                bin_centres = (post_data.wghtd_triads_1d_pdf_ranges_t[1:] + post_data.wghtd_triads_1d_pdf_ranges_t[:-1]) * 0.5
                bin_width = post_data.wghtd_triads_1d_pdf_ranges_t[1] - post_data.wghtd_triads_1d_pdf_ranges_t[0]
                pdf = compute_pdf_t(post_data.wghtd_triads_1d_pdf_counts_t[:, class_type, try_type, n_k3, :], bin_width)
                im8 = ax8.imshow(np.flipud(pdf), aspect='auto', extent = (-np.pi + dtheta_k3/2, np.pi - dtheta_k3/2, 1, sys_vars.ndata), cmap = my_magma) # , norm=mpl.colors.LogNorm()
                ax8.set_xticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
                ax8.set_xticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
                ax8.set_ylabel(r"$t$")
                if class_type == 0:
                    ax8.set_xlabel(r"$\varphi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                else:
                    ax8.set_xlabel(r"$\Phi_{\mathbf{k}_1, \mathbf{k}_2}^{\mathbf{k}_3}$")
                ax8.set_title(r"Weight Flux Density Triad PDF in Time")
                div8  = make_axes_locatable(ax8)
                cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
                cb8   = plt.colorbar(im8, cax = cbax8)
                cb8.set_label(r"wPDF")
                if class_type == 0:
                    plt.suptitle("1D Contributions - PDF In Time: Normal Triads; Triad Type {}; Sector {}".format(try_type, n_k3))
                else:
                    plt.suptitle("1D Contributions - PDF In Time: Generalized Triads; Triad Type {}; Sector {}".format(try_type, n_k3))
                plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_TriadPDF_1D_InTime_Class[{}]_Sec[{}].png".format(try_type, class_type, n_k3), bbox_inches="tight")
                plt.close()

    # -----------------------------------------
    # # --------  Plot Data For Videos
    # -----------------------------------------
    if cmdargs.plotting:

        ## Start timer
        start = TIME.perf_counter()
        print("\n" + tc.Y + "Printing Snaps..." + tc.Rst + "Total Snaps to Print: [" + tc.C + "{}".format(sys_vars.ndata) + tc.Rst + "]")

        if cmdargs.phases:

            if cmdargs.parallel:
                ## No. of processes
                proc_lim = cmdargs.num_procs

                ## Create tasks for the process pool
                if cmdargs.full:
                    groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

                else:
                    groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_phases, post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.phase_R[i, :], post_data.phase_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

                ## Loop of grouped iterable
                for procs in zip_longest(*groups_args):
                    pipes     = []
                    processes = []
                    for p in filter(None, procs):
                        recv, send = mprocs.Pipe(False)
                        processes.append(p)
                        pipes.append(recv)
                        p.start()

                    for process in processes:
                        process.join()
            else:
                ## Loop through simulation and plot data
                for i in range(sys_vars.ndata):
                    ## Plot the data
                    if cmdargs.full:
                        plot_sector_phase_sync_snaps_full(i, cmdargs.out_dir_phases, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.phase_R[i, :], post_data.phase_Phi[i, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                    else:
                        plot_sector_phase_sync_snaps(i, cmdargs.out_dir_phases, post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.phase_R[i, :], post_data.phase_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

        if cmdargs.triads:

            if cmdargs.parallel:
                ## No. of processes
                proc_lim = cmdargs.num_procs

                if cmdargs.triad_type != "all":
                    print("TRIAD TYPE: {}".format(int(cmdargs.triad_type)), end = " ")
                    print(cmdargs.phase_order)
                    if cmdargs.phase_order:
                        if post_data.phase_order_normed_const:
                            R, Phi, enst          = np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect))
                            R_1d, Phi_1d, enst_1d = np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect))
                            R_2d, Phi_2d, enst_2d = np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect))
                            for try_type in range(post_data.NUM_TRIAD_TYPES):
                                for i in range(sys_vars.ndata):
                                    R[i, try_type, :], Phi[i, try_type, :], enst[i, try_type, :]                   = get_normed_collective_phase(post_data.phase_order_C_theta_triads[i, try_type, :], np.sum(post_data.phase_order_norm_const[i, 0, try_type, :, :], axis=-1))
                                    R_1d[i, try_type, :], Phi_1d[i, try_type, :], enst_1d[i, try_type, :]          = get_normed_collective_phase(post_data.phase_order_C_theta_triads_1d[i, try_type, :], np.diag(post_data.phase_order_norm_const[i, 0, try_type, :, :]))
                                    R_2d[i, try_type, :, :], Phi_2d[i, try_type, :, :], enst_2d[i, try_type, :, :] = get_normed_collective_phase_2d(post_data.phase_order_C_theta_triads_2d[i, try_type, :, :], post_data.phase_order_norm_const[i, 0, try_type, :, :])
                        else:
                            R      = np.absolute(post_data.phase_order_C_theta_triads[:, :, :])
                            R_1d   = np.absolute(post_data.phase_order_C_theta_triads_1d[:, :, :])
                            R_2d   = np.absolute(post_data.phase_order_C_theta_triads_2d[:, :, :, :])
                            Phi    = np.angle(post_data.phase_order_C_theta_triads[:, :, :])
                            Phi_1d = np.angle(post_data.phase_order_C_theta_triads_1d[:, :, :])
                            Phi_2d = np.angle(post_data.phase_order_C_theta_triads_2d[:, :, :, :])
                            enst    = np.real(post_data.phase_order_C_theta_triads[:, :, :])
                            enst_1d = np.real(post_data.phase_order_C_theta_triads_1d[:, :, :])
                            enst_2d = np.real(post_data.phase_order_C_theta_triads_2d[:, :, :, :])
                    else:
                        R      = post_data.triad_R[:, :, :]
                        R_1d   = post_data.triad_R_1d[:, :, :]
                        R_2d   = post_data.triad_R_2d[:, :, :, :]
                        Phi    = post_data.triad_Phi[:, :, :] 
                        Phi_1d = post_data.triad_Phi_1d[:, :, :]
                        Phi_2d = post_data.triad_Phi_2d[:, :, :, :]
                        enst    =post_data.enst_flux_per_sec[:, :, :]
                        enst_1d =post_data.enst_flux_per_sec_1d[:, :, :]
                        enst_2d =post_data.enst_flux_per_sec_2d[:, :, :, :]

                    flux_min    = np.amin(enst[:, int(cmdargs.triad_type), :])
                    flux_max    = np.amax(enst[:, int(cmdargs.triad_type), :])
                    flux_1d_min = np.amin(enst_1d[:, int(cmdargs.triad_type), :])
                    flux_1d_max = np.amax(enst_1d[:, int(cmdargs.triad_type), :])
                    flux_2d_min = np.amin(enst_2d[:, int(cmdargs.triad_type), :, :])
                    # flux_2d_max = np.amax(enst_2d[:, int(cmdargs.triad_type), :, :])
                    flux_2d_max = -flux_2d_min 

                    ## Create tasks for the process pool
                    if cmdargs.full:
                        if cmdargs.triad_plot_type == "sec":
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full_sec, args = (i, cmdargs.out_dir_triads, 
                                                    run_data.w[i, :, :], 
                                                    post_data.enst_spectrum[i, :, :], 
                                                    enst_1d[i, int(cmdargs.triad_type), :], 
                                                    enst_2d[i, int(cmdargs.triad_type), :, :], 
                                                    post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                    post_data.theta_k3, 
                                                    R_1d[i, int(cmdargs.triad_type), :], R_2d[i, int(cmdargs.triad_type), :, :], 
                                                    Phi_1d[i, int(cmdargs.triad_type), :], Phi_2d[i, int(cmdargs.triad_type), :, :], 
                                                    [flux_min, flux_max, flux_2d_min, flux_2d_max],
                                                    run_data.time[i], 
                                                    run_data.x, run_data.y, 
                                                    sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                        elif cmdargs.triad_plot_type == "all":
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_all, args = (i, cmdargs.out_dir_triads, 
                                                                    run_data.w[i, :, :], 
                                                                    post_data.enst_spectrum[i, :, int(sys_vars.Nx/3):], 
                                                                    enst[i, int(cmdargs.triad_type), :], 
                                                                    enst_1d[i, int(cmdargs.triad_type), :], 
                                                                    enst_2d[i, int(cmdargs.triad_type), :, :], 
                                                                    post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                    post_data.theta_k3, 
                                                                    R[i, int(cmdargs.triad_type), :], R_1d[i, int(cmdargs.triad_type), :], R_2d[i, int(cmdargs.triad_type), :, :], 
                                                                    Phi[i, int(cmdargs.triad_type), :], Phi_1d[i, int(cmdargs.triad_type), :], Phi_2d[i, int(cmdargs.triad_type), :, :],
                                                                    [flux_min, flux_max, flux_1d_min, flux_1d_max, flux_2d_min, flux_2d_max], 
                                                                    run_data.time[i], 
                                                                    run_data.x, run_data.y, 
                                                                    sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                        else:
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, int(cmdargs.triad_type), :], post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                    else:
                        groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

                    ## Loop of grouped iterable
                    for procs in zip_longest(*groups_args): 
                        pipes     = []
                        processes = []
                        for p in filter(None, procs):
                            recv, send = mprocs.Pipe(False)
                            processes.append(p)
                            pipes.append(recv)
                            p.start()

                        for process in processes:
                            process.join()
                else:
                    num_triad_types = post_data.triad_R_2d.shape[1]
                    if cmdargs.phase_order:
                        if post_data.phase_order_normed_const:
                            R, Phi, enst          = np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect))
                            R_1d, Phi_1d, enst_1d = np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect))
                            R_2d, Phi_2d, enst_2d = np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect))
                            for try_type in range(post_data.NUM_TRIAD_TYPES):
                                for i in range(sys_vars.ndata):
                                    R[i, try_type, :], Phi[i, try_type, :], enst[i, try_type, :]                   = get_normed_collective_phase(post_data.phase_order_C_theta_triads[i, try_type, :], np.sum(post_data.phase_order_norm_const[i, 0, try_type, :, :], axis=-1))
                                    R_1d[i, try_type, :], Phi_1d[i, try_type, :], enst_1d[i, try_type, :]          = get_normed_collective_phase(post_data.phase_order_C_theta_triads_1d[i, try_type, :], np.diag(post_data.phase_order_norm_const[i, 0, try_type, :, :]))
                                    R_2d[i, try_type, :, :], Phi_2d[i, try_type, :, :], enst_2d[i, try_type, :, :] = get_normed_collective_phase_2d(post_data.phase_order_C_theta_triads_2d[i, try_type, :, :], post_data.phase_order_norm_const[i, 0, try_type, :, :])
                        else:
                            R      = np.absolute(post_data.phase_order_C_theta_triads[:, :, :])
                            R_1d   = np.absolute(post_data.phase_order_C_theta_triads_1d[:, :, :])
                            R_2d   = np.absolute(post_data.phase_order_C_theta_triads_2d[:, :, :, :])
                            Phi    = np.angle(post_data.phase_order_C_theta_triads[:, :, :])
                            Phi_1d = np.angle(post_data.phase_order_C_theta_triads_1d[:, :, :])
                            Phi_2d = np.angle(post_data.phase_order_C_theta_triads_2d[:, :, :, :])
                            enst    = np.real(post_data.phase_order_C_theta_triads[:, :, :])
                            enst_1d = np.real(post_data.phase_order_C_theta_triads_1d[:, :, :])
                            enst_2d = np.real(post_data.phase_order_C_theta_triads_2d[:, :, :, :])
                    else:
                        R      = post_data.triad_R[:, :, :]
                        R_1d   = post_data.triad_R_1d[:, :, :]
                        R_2d   = post_data.triad_R_2d[:, :, :, :]
                        Phi    = post_data.triad_Phi[:, :, :] 
                        Phi_1d = post_data.triad_Phi_1d[:, :, :]
                        Phi_2d = post_data.triad_Phi_2d[:, :, :, :]
                        enst    =post_data.enst_flux_per_sec[:, :, :]
                        enst_1d =post_data.enst_flux_per_sec_1d[:, :, :]
                        enst_2d =post_data.enst_flux_per_sec_2d[:, :, :, :]

                    for t in range(num_triad_types):
                        
                        flux_min    = np.amin(enst[:, t, :])
                        flux_max    = np.amax(enst[:, t, :])
                        flux_1d_min = np.amin(enst_1d[:, t, :])
                        flux_1d_max = np.amax(enst_1d[:, t, :])
                        flux_2d_min = np.amin(enst_2d[:, t, :, :])
                        # flux_2d_max = np.amax(enst_2d[:, t, :, :])
                        flux_2d_max = -flux_2d_min 

                        print("TRIAD TYPE: {}".format(t), end = " ")
                        ## Create tasks for the process pool
                        if cmdargs.full:
                            if cmdargs.triad_plot_type == "sec":
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full_sec, args = (i, cmdargs.out_dir_triads, 
                                                        run_data.w[i, :, :], 
                                                        post_data.enst_spectrum[i, :, :], 
                                                        enst_1d[i, t, :], 
                                                        enst_2d[i, t, :, :], 
                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                        post_data.theta_k3, 
                                                        R_1d[i, t, :], R_2d[i, t, :, :], 
                                                        Phi_1d[i, t, :], Phi_2d[i, t, :, :], 
                                                        [flux_min, flux_max, flux_2d_min, flux_2d_max],
                                                        run_data.time[i], 
                                                        run_data.x, run_data.y, 
                                                        sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                            elif cmdargs.triad_plot_type == "all":
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_all, args = (i, cmdargs.out_dir_triads, 
                                                                        run_data.w[i, :, :], 
                                                                        post_data.enst_spectrum[i, :, int(sys_vars.Nx/3):], 
                                                                        enst[i, t, :], 
                                                                        enst_1d[i, t, :], 
                                                                        enst_2d[i, t, :, :], 
                                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                        post_data.theta_k3, 
                                                                        R[i, t, :], R_1d[i, t, :], R_2d[i, t, :, :], 
                                                                        Phi[i, t, :], Phi_1d[i, t, :], Phi_2d[i, t, :, :], 
                                                                        [flux_min, flux_max, flux_1d_min, flux_1d_max, flux_2d_min, flux_2d_max], 
                                                                        run_data.time[i], 
                                                                        run_data.x, run_data.y, 
                                                                        sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                            else:
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, t, :], post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                        else:
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, t, :], post_data.triad_Phi[i, t, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

                        ## Loop of grouped iterable
                        for procs in zip_longest(*groups_args): 
                            pipes     = []
                            processes = []
                            for p in filter(None, procs):
                                recv, send = mprocs.Pipe(False)
                                processes.append(p)
                                pipes.append(recv)
                                p.start()

                            for process in processes:
                                process.join()
                    
                        ## Video variables
                        framesPerSec = 15
                        inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
                        videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TYPE[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, t)
                        cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
                        # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

                        process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
                        [runCodeOutput, runCodeErr] = process.communicate()
                        print(runCodeOutput)
                        print(runCodeErr)
                        process.wait()

                        ## Prin summary of timmings to screen
                        print("\n" + tc.Y + "Finished making video..." + tc.Rst)
                        print("Video Location: " + tc.C + videoName + tc.Rst + "\n")

                        ## Remove the generated snaps after video is created
                        run("cd {};".format(cmdargs.out_dir_triads) + "rm {};".format("./*.png") + "cd -;", shell = True)
            else:   
                if cmdargs.triad_type != "all":
                    ## Loop through simulation and plot data
                    for i in range(sys_vars.ndata):
                        if cmdargs.phase_order:
                            if post_data.phase_order_normed_const:
                                R, Phi, enst          = np.zeros((post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect))
                                R_1d, Phi_1d, enst_1d = np.zeros((post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect))
                                R_2d, Phi_2d, enst_2d = np.zeros((post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.NUM_TRIAD_TYPES, post_data.num_k3_sect, post_data.num_k1_sect))
                                for try_type in range(post_data.NUM_TRIAD_TYPES):
                                        R[i, try_type, :], Phi[i, try_type, :], enst[i, try_type, :]                   = get_normed_collective_phase(post_data.phase_order_C_theta_triads[i, try_type, :], np.sum(post_data.phase_order_norm_const[i, 0, try_type, :, :], axis=-1))
                                        R_1d[i, try_type, :], Phi_1d[i, try_type, :], enst_1d[i, try_type, :]          = get_normed_collective_phase(post_data.phase_order_C_theta_triads_1d[i, try_type, :], np.diag(post_data.phase_order_norm_const[i, 0, try_type, :, :]))
                                        R_2d[i, try_type, :, :], Phi_2d[i, try_type, :, :], enst_2d[i, try_type, :, :] = get_normed_collective_phase_2d(post_data.phase_order_C_theta_triads_2d[i, try_type, :, :], post_data.phase_order_norm_const[i, 0, try_type, :, :])
                            else:
                                R       = np.absolute(post_data.phase_order_C_theta_triads[i, :, :])
                                R_1d    = np.absolute(post_data.phase_order_C_theta_triads_1d[i, :, :])
                                R_2d    = np.absolute(post_data.phase_order_C_theta_triads_2d[i, :, :, :])
                                Phi     = np.angle(post_data.phase_order_C_theta_triads[i, i, :])
                                Phi_1d  = np.angle(post_data.phase_order_C_theta_triads_1d[i, :, :])
                                Phi_2d  = np.angle(post_data.phase_order_C_theta_triads_2d[i, :, :, :])
                                enst    = np.real(post_data.phase_order_C_theta_triads[i, :, :])
                                enst_1d = np.real(post_data.phase_order_C_theta_triads_1d[i, :, :])
                                enst_2d = np.real(post_data.phase_order_C_theta_triads_2d[i, :, :, :])
                        else:
                            R      = post_data.triad_R[i, :, :]
                            R_1d   = post_data.triad_R_1d[i, :, :]
                            R_2d   = post_data.triad_R_2d[i, :, :, :]
                            Phi    = post_data.triad_Phi[i, :, :] 
                            Phi_1d = post_data.triad_Phi_1d[i, :, :]
                            Phi_2d = post_data.triad_Phi_2d[i, :, :, :]
                            enst    =post_data.enst_flux_per_sec[i, :, :]
                            enst_1d =post_data.enst_flux_per_sec_1d[i, :, :]
                            enst_2d =post_data.enst_flux_per_sec_2d[i, :, :, :]
                        ## Plot the data
                        if cmdargs.full:
                            if cmdargs.triad_plot_type == "sec":
                                plot_sector_phase_sync_snaps_full_sec(i, cmdargs.out_dir_triads, 
                                                                        run_data.w[i, :, :], 
                                                                        post_data.enst_spectrum[i, :, :], 
                                                                        enst_1d[i, int(cmdargs.triad_type), :], 
                                                                        enst_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                        post_data.theta_k3, 
                                                                        R_1d[int(cmdargs.triad_type), :], R_2d[int(cmdargs.triad_type), :, :], 
                                                                        Phi_1d[int(cmdargs.triad_type), :], Phi_2d[int(cmdargs.triad_type), :, :], 
                                                                        [flux_min, flux_max, flux_2d_min, flux_2d_max], 
                                                                        run_data.time[i], 
                                                                        run_data.x, run_data.y, 
                                                                        sys_vars.Nx, sys_vars.Ny)
                            elif cmdargs.triad_plot_type == "all":
                                plot_sector_phase_sync_snaps_all(i, cmdargs.out_dir_triads, 
                                                                        run_data.w[i, :, :], 
                                                                        post_data.enst_spectrum[i, :, int(sys_vars.Nx/3):], 
                                                                        enst[i, int(cmdargs.triad_type), :], 
                                                                        enst_1d[i, int(cmdargs.triad_type), :], 
                                                                        enst_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                        post_data.theta_k3, 
                                                                        R[int(cmdargs.triad_type), :], R_1d[int(cmdargs.triad_type), :], R_2d[int(cmdargs.triad_type), :, :], 
                                                                        Phi[int(cmdargs.triad_type), :], Phi_1d[int(cmdargs.triad_type), :], Phi_2d[int(cmdargs.triad_type), :, :], 
                                                                        [flux_min, flux_max, flux_1d_min, flux_1d_max, flux_2d_min, flux_2d_max], 
                                                                        run_data.time[i], 
                                                                        run_data.x, run_data.y, 
                                                                        sys_vars.Nx, sys_vars.Ny)
                            else:
                                plot_sector_phase_sync_snaps_full(i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                        else:
                            plot_sector_phase_sync_snaps(i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)
                else:
                    num_triad_types = post_data.triad_R_2d.shape[1]
                    for t in range(num_triad_types):
                        print("TRIAD TYPE: {}".format(t), end = " ")
                        ## Loop through simulation and plot data
                        for i in range(sys_vars.ndata):
                            ## Plot the data
                            if cmdargs.full: 
                                if cmdargs.triad_plot_type == "sec":
                                    plot_sector_phase_sync_snaps_full_sec(i, cmdargs.out_dir_triads, 
                                                                            run_data.w[i, :, :], 
                                                                            post_data.enst_spectrum[i, :, :], 
                                                                            post_data.enst_flux_per_sec_1d[i, t, :], 
                                                                            post_data.enst_flux_per_sec_2d[i, t, :, :], 
                                                                            post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                            post_data.theta_k3, 
                                                                            post_data.triad_R_1d[i, t, :], post_data.triad_R_2d[i, t, :, :], 
                                                                            post_data.triad_Phi_1d[i, t, :], post_data.triad_Phi_2d[i, t, :, :], 
                                                                            [flux_min, flux_max, flux_2d_min, flux_2d_max], 
                                                                            run_data.time[i], 
                                                                            run_data.x, run_data.y, 
                                                                            sys_vars.Nx, sys_vars.Ny)
                                elif cmdargs.triad_plot_type == "all":
                                    plot_sector_phase_sync_snaps_all(i, cmdargs.out_dir_triads, 
                                                                            run_data.w[i, :, :], 
                                                                            post_data.enst_spectrum[i, :, int(sys_vars.Nx/3):], 
                                                                            post_data.enst_flux_per_sec[i, t, :], 
                                                                            post_data.enst_flux_per_sec_1d[i, t, :], 
                                                                            post_data.enst_flux_per_sec_2d[i, t, :, :], 
                                                                            post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                            post_data.theta_k3, 
                                                                            post_data.triad_R[i, t, :], post_data.triad_R_1d[i, t, :], post_data.triad_R_2d[i, t, :, :], 
                                                                            post_data.triad_Phi[i, t, :], post_data.triad_Phi_1d[i, t, :], post_data.triad_Phi_2d[i, t, :, :], 
                                                                            [flux_min, flux_max, flux_1d_min, flux_1d_max, flux_2d_min, flux_2d_max], 
                                                                            run_data.time[i], 
                                                                            run_data.x, run_data.y, 
                                                                            sys_vars.Nx, sys_vars.Ny)
                                else:
                                    plot_sector_phase_sync_snaps_full(i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, t, :], post_data.triad_Phi[i, t, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                            else:
                                plot_sector_phase_sync_snaps(i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3):], post_data.theta_k3, post_data.triad_R[i, t, :], post_data.triad_Phi[i, t, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

                        ## Video variables
                        framesPerSec = 15
                        inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
                        videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TYPE[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, t)
                        cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
                        # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

                        process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
                        [runCodeOutput, runCodeErr] = process.communicate()
                        print(runCodeOutput)
                        print(runCodeErr)
                        process.wait()

                        ## Prin summary of timmings to screen
                        print("\n" + tc.Y + "Finished making video..." + tc.Rst)
                        print("Video Location: " + tc.C + videoName + tc.Rst + "\n")

                        ## Remove the generated snaps after video is created
                        run("cd {};".format(cmdargs.out_dir_triads) + "rm {};".format("./*.png") + "cd -;", shell = True)

        ## End timer
        end       = TIME.perf_counter()
        plot_time = end - start
        print("\n" + tc.Y + "Finished Plotting..." + tc.Rst)
        print("\n\nPlotting Time: " + tc.C + "{:5.8f}s\n\n".format(plot_time) + tc.Rst)


    #------------------------------------
    # # ----- Make Video
    #------------------------------------
    if cmdargs.video:

        ## Start timer
        start = TIME.perf_counter()

        if cmdargs.phases:

            ## Video variables
            framesPerSec = 15
            inputFile    = cmdargs.out_dir_phases + "Phase_Sync_SNAP_%05d.png"
            videoName    = cmdargs.out_dir_phases + "PhaseSync_N[{},{}]_u0[{}]_NSECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TAG[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, cmdargs.tag)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
            [runCodeOutput, runCodeErr] = process.communicate()
            print(runCodeOutput)
            print(runCodeErr)
            process.wait()

            ## Prin summary of timmings to screen
            print("\n" + tc.Y + "Finished making video..." + tc.Rst)
            print("Video Location: " + tc.C + videoName + tc.Rst + "\n")

            ## Remove the generated snaps after video is created
            print("cd {};".format(cmdargs.out_dir_phases) + "rm {};".format("./*.png") + "cd -;")
            run("cd {};".format(cmdargs.out_dir_phases) + "rm {};".format("./*.png") + "cd -;", shell = True)

        if cmdargs.triads and cmdargs.triad_type != "all":

            ## Video variables
            framesPerSec = 15
            inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
            videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{},{}]_KFRAC[{:1.2f},{:1.2f}]_TYPE[{}]_TAG[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_k3_sect, post_data.num_k1_sect, post_data.kmin_sqr, post_data.kmax_frac, int(cmdargs.triad_type), cmdargs.tag)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
            [runCodeOutput, runCodeErr] = process.communicate()
            print(runCodeOutput)
            print(runCodeErr)
            process.wait()

            ## Remove the generated snaps after video is created
            print("cd {};".format(cmdargs.out_dir_triads) + "rm {};".format("./*.png") + "cd -;")
            run("cd {};".format(cmdargs.out_dir_triads) + "rm {};".format("./*.png") + "cd -;", shell = True)
            

        ## Start timer
        end = TIME.perf_counter()
        print("Movie Time:" + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)