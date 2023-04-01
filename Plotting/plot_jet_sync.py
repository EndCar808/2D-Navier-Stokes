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
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data
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


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:t:", ["par", "plot", "vid", "phase", "triads=", "full="])
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
        flux_2d_max = np.amax(post_data.enst_flux_per_sec_2d[:, 0, :, :])
    # -----------------------------------------
    # # --------  Make Output Directories
    # -----------------------------------------
    ## Make output directory for snaps
    cmdargs.out_dir_avg     = cmdargs.out_dir + "PHASE_SYNC_AVG_SNAPS/"
    cmdargs.out_dir_valid     = cmdargs.out_dir + "PHASE_SYNC_VALID_SNAPS/"
    cmdargs.out_dir_phases     = cmdargs.out_dir + "PHASE_SYNC_SNAPS/"
    cmdargs.out_dir_triads     = cmdargs.out_dir + "TRIAD_PHASE_SYNC_SNAPS/"
    if os.path.isdir(cmdargs.out_dir_phases) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_phases)
    if os.path.isdir(cmdargs.out_dir_triads) != True:
        print("Making folder:" + tc.C + " TRIAD_PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_triads)
    if os.path.isdir(cmdargs.out_dir_valid) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_VALID_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_valid)
    if os.path.isdir(cmdargs.out_dir_avg) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_AVG_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_avg)
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
    ax3.plot(np.sum(np.real(post_data.phase_order_C_theta_triads[:, triad_type, :]), axis = -1) * const_fac * norm_fac, '--', label=r"$\sum_\theta \Re\{Re^{\Phi}\}$ Direct")
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
        ax3.plot(np.real(post_data.phase_order_C_theta_triads[:, triad_type, k3]) * const_fac * norm_fac,'--', label=r"$\Re\{Re^{\Phi}\}$ Direct")
        ax3.set_title(r"Compare Complexification: Sector {}".format(k3 + 1))
        ax3.grid()
        ax3.legend()
        plt.suptitle("Sector {}".format(k3 + 1))
        plt.savefig(cmdargs.out_dir_valid_snaps + "/EnstrophyFlux_ComparePerk3Sector_k3[{}].png".format(k3 + 1), bbox_inches="tight")
        plt.close()

    # -----------------------------------------
    # # --------  Plot Data
    # -----------------------------------------
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

    print("Plotting Data")
    try_type = 0

    # ## Time averaged 2D Data
    for try_type in range(post_data.triad_R_2d.shape[1]):
        print("Plotting Triad Type {}".format(try_type))
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
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_Sync_2D.png".format(try_type), bbox_inches="tight")
        plt.close()

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
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_Sync_2Dand1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        # ## Time averaged 1d data
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
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_Sync_1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        ## Spacetime plots
        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax6 = fig.add_subplot(gs[0, 0])
        im6 = ax6.imshow(post_data.triad_R_1d[:, try_type, :], aspect='auto', extent = (theta_k3_min, theta_k3_max, 1, sys_vars.ndata), cmap = my_magma, vmin = 0.0, vmax = 1.0)
        ax6.set_xticks(angticks)
        ax6.set_xticklabels(angtickLabels)
        ax6.set_ylabel(r"$t$")
        ax6.set_xlabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector (1D)")
        div6  = make_axes_locatable(ax6)
        cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
        cb6   = plt.colorbar(im6, cax = cbax6)
        cb6.set_label(r"$\mathcal{R}^{1D}$")
        ax7 = fig.add_subplot(gs[0, 1])
        im7 = ax7.imshow(post_data.triad_Phi_1d[:, try_type, :], aspect='auto', extent = (theta_k3_min, theta_k3_max, 1, sys_vars.ndata), cmap = my_hsv, vmin = -np.pi, vmax = np.pi)
        ax7.set_xticks(angticks)
        ax7.set_xticklabels(angtickLabels)
        ax6.set_ylabel(r"$t$")
        ax6.set_xlabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector (1D)")
        ax7.set_title(r"Average Angle Per Sector (1D)")
        div7  = make_axes_locatable(ax7)
        cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
        cb7   = plt.colorbar(im7, cax = cbax7)
        cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
        cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        cb7.set_label(r"$\Phi^{1D}$")
        ax8 = fig.add_subplot(gs[0, 2])
        data = post_data.enst_flux_per_sec_1d[:, try_type, :]
        im8 = ax8.imshow(post_data.enst_flux_per_sec_1d[:, try_type, :], aspect='auto', extent = (theta_k3_min, theta_k3_max, 1, sys_vars.ndata), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_xticks(angticks)
        ax8.set_xticklabels(angtickLabels)
        ax6.set_ylabel(r"$t$")
        ax6.set_xlabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector (1D)")
        ax8.set_title(r"Enstrophy Flux Per Sector (1D)")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Pi_{\mathcal{S}_\theta}^{1D}$")
        plt.suptitle(r"Spacetime 1D Data - Triad Type: {}".format(try_type))
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_SpaceTime_1D.png".format(try_type), bbox_inches="tight")
        plt.close()

        fig = plt.figure(figsize = (21, 9))
        gs  = GridSpec(1, 3)
        ax8 = fig.add_subplot(gs[0, 2])
        data = post_data.enst_flux_per_sec[:, try_type, :]
        im8 = ax8.imshow(data, aspect='auto', extent = (theta_k3_min, theta_k3_max, 1, sys_vars.ndata), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
        ax8.set_xticks(angticks)
        ax8.set_xticklabels(angtickLabels)
        ax6.set_ylabel(r"$t$")
        ax6.set_xlabel(r"$\theta$")
        ax6.set_title(r"Sync Per Sector (1D)")
        ax8.set_title(r"Enstrophy Flux Per Sector")
        div8  = make_axes_locatable(ax8)
        cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
        cb8   = plt.colorbar(im8, cax = cbax8)
        cb8.set_label(r"$\Pi_{\mathcal{S}_\theta}^{1D}$")
        plt.savefig(cmdargs.out_dir_avg_snaps + "/Type[{}]_SpaceTime_PerSector.png".format(try_type), bbox_inches="tight")
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
                    ## Create tasks for the process pool
                    if cmdargs.full:
                        if cmdargs.triad_plot_type == "sec":
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full_sec, args = (i, cmdargs.out_dir_triads, 
                                                    run_data.w[i, :, :], 
                                                    post_data.enst_spectrum[i, :, :], 
                                                    post_data.enst_flux_per_sec_1d[i, int(cmdargs.triad_type), :], 
                                                    post_data.enst_flux_per_sec_2d[i, int(cmdargs.triad_type), :, :], 
                                                    post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                    post_data.theta_k3, 
                                                    post_data.triad_R_1d[i, int(cmdargs.triad_type), :], post_data.triad_R_2d[i, int(cmdargs.triad_type), :, :], 
                                                    post_data.triad_Phi_1d[i, int(cmdargs.triad_type), :], post_data.triad_Phi_2d[i, int(cmdargs.triad_type), :, :], 
                                                    [flux_min, flux_max, flux_2d_min, flux_2d_max],
                                                    run_data.time[i], 
                                                    run_data.x, run_data.y, 
                                                    sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                        elif cmdargs.triad_plot_type == "all":
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_all, args = (i, cmdargs.out_dir_triads, 
                                                                    run_data.w[i, :, :], 
                                                                    post_data.enst_spectrum[i, :, int(sys_vars.Nx/3):], 
                                                                    post_data.enst_flux_per_sec[i, int(cmdargs.triad_type), :], 
                                                                    post_data.enst_flux_per_sec_1d[i, int(cmdargs.triad_type), :], 
                                                                    post_data.enst_flux_per_sec_2d[i, int(cmdargs.triad_type), :, :], 
                                                                    post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                    post_data.theta_k3, 
                                                                    post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_R_1d[i, int(cmdargs.triad_type), :], post_data.triad_R_2d[i, int(cmdargs.triad_type), :, :], 
                                                                    post_data.triad_Phi[i, int(cmdargs.triad_type), :], post_data.triad_Phi_1d[i, int(cmdargs.triad_type), :], post_data.triad_Phi_2d[i, int(cmdargs.triad_type), :, :], 
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
                    for t in range(num_triad_types):
                        print("TRIAD TYPE: {}".format(t), end = " ")
                        ## Create tasks for the process pool
                        if cmdargs.full:
                            if cmdargs.triad_plot_type == "sec":
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full_sec, args = (i, cmdargs.out_dir_triads, 
                                                        run_data.w[i, :, :], 
                                                        post_data.enst_spectrum[i, :, :], 
                                                        post_data.enst_flux_1d[i, t, :], 
                                                        post_data.enst_flux_per_sec_2d[i, t, :, :], 
                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                        post_data.theta_k3, 
                                                        post_data.triad_R_1d[i, t, :], post_data.triad_R_2d[i, t, :, :], 
                                                        post_data.triad_Phi_1d[i, t, :], post_data.triad_Phi_2d[i, t, :, :], 
                                                        [flux_min, flux_max, flux_2d_min, flux_2d_max],
                                                        run_data.time[i], 
                                                        run_data.x, run_data.y, 
                                                        sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                            elif cmdargs.triad_plot_type == "all":
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_all, args = (i, cmdargs.out_dir_triads, 
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
                        ## Plot the data
                        if cmdargs.full:                       
                            if cmdargs.triad_plot_type == "sec":
                                plot_sector_phase_sync_snaps_full_sec(i, cmdargs.out_dir_triads, 
                                                                        run_data.w[i, :, :], 
                                                                        post_data.enst_spectrum[i, :, :], 
                                                                        post_data.enst_flux_per_sec_1d[i, int(cmdargs.triad_type), :], 
                                                                        post_data.enst_flux_per_sec_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                        post_data.theta_k3, 
                                                                        post_data.triad_R_1d[i, int(cmdargs.triad_type), :], post_data.triad_R_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        post_data.triad_Phi_1d[i, int(cmdargs.triad_type), :], post_data.triad_Phi_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        [flux_min, flux_max, flux_2d_min, flux_2d_max], 
                                                                        run_data.time[i], 
                                                                        run_data.x, run_data.y, 
                                                                        sys_vars.Nx, sys_vars.Ny)
                            elif cmdargs.triad_plot_type == "all":
                                plot_sector_phase_sync_snaps_all(i, cmdargs.out_dir_triads, 
                                                                        run_data.w[i, :, :], 
                                                                        post_data.enst_spectrum[i, :, int(sys_vars.Nx/3):], 
                                                                        post_data.enst_flux_per_sec[i, int(cmdargs.triad_type), :], 
                                                                        post_data.enst_flux_per_sec_1d[i, int(cmdargs.triad_type), :], 
                                                                        post_data.enst_flux_per_sec_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        post_data.phases[i, :, int(sys_vars.Nx/3):], 
                                                                        post_data.theta_k3, 
                                                                        post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_R_1d[i, int(cmdargs.triad_type), :], post_data.triad_R_2d[i, int(cmdargs.triad_type), :, :], 
                                                                        post_data.triad_Phi[i, int(cmdargs.triad_type), :], post_data.triad_Phi_1d[i, int(cmdargs.triad_type), :], post_data.triad_Phi_2d[i, int(cmdargs.triad_type), :, :], 
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