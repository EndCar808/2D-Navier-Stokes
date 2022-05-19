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
    post_data = import_post_processing_data(post_file_path, sys_vars, method)

    ## Number of triad types
    num_triad_types = 7
    # -----------------------------------------
    # # --------  Plot Data
    # -----------------------------------------
    # ##------------------------ Plot enstrophy fluxes in C
    # fig = plt.figure(figsize = (16, 8))
    # gs  = GridSpec(1, 2)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(run_data.time, post_data.enst_flux_C[:])
    # ax1.set_xlabel(r"$t$")
    # ax1.set_yscale("symlog")
    # ax1.set_title(r"$\Pi_{\mathcal{C}}$: Enstrophy Flux into C")
    # ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.plot(run_data.time, post_data.enst_diss_C[:])
    # ax2.set_xlabel(r"$t$")
    # ax2.set_yscale("symlog")
    # ax2.set_title(r"$\epsilon_{\mathcal{C}}$: Enstrophy Diss in C")
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    # plt.savefig(cmdargs.out_dir + "EnstrophyFluxDiss_C.png")
    # plt.close()

    # ##------------------------ Plot enstrophy fluxs
    # fig = plt.figure(figsize = (21, 8))
    # gs  = GridSpec(1, 3)
    # ax1 = fig.add_subplot(gs[0, 0])
    # for i in range(post_data.enst_flux_spec.shape[0]):
    #     ax1.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.enst_flux_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax1.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.enst_flux_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax1.set_xlabel(r"$k$")
    # ax1.set_xscale('log')   
    # ax1.set_yscale('symlog')
    # ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax1.set_title(r"$\Pi(|\mathbf{k}|)$: Enstrophy Flux Spectrum")
    
    # ax2 = fig.add_subplot(gs[0, 1])
    # for i in range(post_data.enst_flux_spec.shape[0]):
    #     ax2.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.enst_diss_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.enst_diss_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax2.set_xlabel(r"$k$")
    # ax2.set_yscale('log')
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax2.set_title(r"$\epsilon(|\mathbf{k}|)$: Enstrophy Dissipation Spectrum")
    # ax2.set_xscale('log')

    # ax3 = fig.add_subplot(gs[0, 2])
    # for i in range(post_data.enst_flux_spec.shape[0]):
    #     ax3.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.d_enst_dt_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax3.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.d_enst_dt_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax3.set_xlabel(r"$k$")
    # ax3.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax3.set_title(r"$\frac{\mathrm{d} \mathcal{E}}{\mathrm{d} t}(|\mathbf{k}|)$: Enstrophy Rate of Change Spectrum")
    # ax3.set_xscale('log') 
    # ax3.set_yscale('symlog') 

    # plt.savefig(cmdargs.out_dir + "EnstrophySpectra.png")
    # plt.close()

    # ##------------------------ Time Averaged Enstorphy Spectra and Flux Spectra
    # fig = plt.figure(figsize = (21, 8))
    # gs  = GridSpec(1, 2)
    # ax2 = fig.add_subplot(gs[0, 0])
    # for i in range(spec_data.enst_spectrum.shape[0]):
    #     ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enst_spectrum[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(spec_data.enst_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax2.set_xlabel(r"$k$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax2.set_title(r"$\mathcal{E}(|\mathbf{k}|)$: Enstrophy Spectrum")

    # ax2 = fig.add_subplot(gs[0, 1])
    # for i in range(post_data.enst_flux_spec.shape[0]):
    #     ax2.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.enst_flux_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.enst_flux_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax2.set_xlabel(r"$k$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('symlog')
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax2.set_title(r"$\Pi(|\mathbf{k}|)$: Enstrophy Flux Spectrum")
    
    # plt.savefig(cmdargs.out_dir + "TimeAveragedEnstrophySpectra.png")
    # plt.close()


    # ##------------------------ Plot energy fluxes in C
    # fig = plt.figure(figsize = (16, 8))
    # gs  = GridSpec(1, 2)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(run_data.time, post_data.enrg_flux_C[:])
    # ax1.set_xlabel(r"$t$")
    # ax1.set_yscale("symlog")
    # ax1.set_title(r"$\Pi_{\mathcal{C}}^{\mathcal{K}}$: Energy Flux into C")
    # ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.plot(run_data.time, post_data.enrg_diss_C[:])
    # ax2.set_xlabel(r"$t$")
    # ax2.set_yscale("symlog")
    # ax2.set_title(r"$\kappa_{\mathcal{C}}$: Energy Diss in C")
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    # plt.savefig(cmdargs.out_dir + "EnergyFluxDiss_C.png")
    # plt.close()

    # ##------------------------ Plot energy fluxs
    # fig = plt.figure(figsize = (21, 8))
    # gs  = GridSpec(1, 3)
    # ax1 = fig.add_subplot(gs[0, 0])
    # for i in range(post_data.enrg_flux_spec.shape[0]):
    #     ax1.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.enrg_flux_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax1.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.enrg_flux_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax1.set_xlabel(r"$k$")
    # ax1.set_xscale('log')
    # ax1.set_yscale('symlog')
    # ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax1.set_title(r"$\Pi(|\mathbf{k}|)$: Energy Flux Spectrum")
    
    # ax2 = fig.add_subplot(gs[0, 1])
    # for i in range(post_data.enrg_flux_spec.shape[0]):
    #     ax2.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.enrg_diss_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.enrg_diss_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax2.set_xlabel(r"$k$")
    # ax2.set_yscale('log')
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax2.set_title(r"$\kappa(|\mathbf{k}|)$: Energy Dissipation Spectrum")
    # ax2.set_xscale('log')

    # ax3 = fig.add_subplot(gs[0, 2])
    # for i in range(post_data.enrg_flux_spec.shape[0]):
    #     ax3.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.d_enrg_dt_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax3.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.d_enrg_dt_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax3.set_xlabel(r"$k$")
    # ax3.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax3.set_title(r"$\frac{\mathrm{d} \mathcal{K}}{\mathrm{d} t}(|\mathbf{k}|)$: Energy Rate of Change Spectrum")
    # ax3.set_xscale('log') 
    # ax3.set_yscale('symlog') 

    # plt.savefig(cmdargs.out_dir + "EnergySpectra.png")
    # plt.close()

    # ##------------------------ Time Averaged Energy Spectra and Flux Spectra
    # fig = plt.figure(figsize = (21, 8))
    # gs  = GridSpec(1, 2)
    # ax2 = fig.add_subplot(gs[0, 0])
    # for i in range(spec_data.enrg_spectrum.shape[0]):
    #     ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enrg_spectrum[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(spec_data.enrg_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax2.set_xlabel(r"$k$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax2.set_title(r"$\mathcal{K}(|\mathbf{k}|)$: Energy Spectrum")

    # ax2 = fig.add_subplot(gs[0, 1])
    # for i in range(post_data.enrg_flux_spec.shape[0]):
    #     ax2.plot(np.arange(1, int(sys_vars.Nx/3)), post_data.enrg_flux_spec[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    # ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(post_data.enrg_flux_spec[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    # ax2.set_xlabel(r"$k$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('symlog')
    # ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax2.set_title(r"$\Pi^{\mathcal{K}}(|\mathbf{k}|)$: Energy Flux Spectrum")
    
    # plt.savefig(cmdargs.out_dir + "TimeAveragedEnergySpectra.png")
    # plt.close()





    # -----------------------------------------
    # # --------  Compare Data
    # -----------------------------------------
    ## Plot flux comparisons
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, 0, :], axis = -1))
    ax1.plot(run_data.time, post_data.enst_flux_C[:])
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"\Pi_{\mathcal{C}}")
    ax1.legend([r"$\sum_{\theta}\Pi_{\mathcal{C}_{\theta}}$: Sum over Sectors of Flux", r"$\Pi_{\mathcal{C}}$: Enstrophy Flux into $\mathcal{C}$"])
    ax1.set_yscale('symlog')
    # ax1.set_xscale('log')
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    plot = []
    for i in range(num_triad_types):
        l, = ax2.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, i, :], axis = 1))
        plot.append(l.get_color())
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\sum_{\theta}\Pi_{\mathcal{C}_{\theta}}$")
    ax2.legend([r"Type ${}$".format(t) for t in range(num_triad_types)])
    # ax2.set_yscale('symlog')
    # ax2.set_xscale('log')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, 1, :], axis = -1), color = plot[1])
    ax3.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, 2, :], axis = -1), color = plot[2])
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$\sum_{\theta}\Pi_{\mathcal{C}_{\theta}}$")
    ax3.legend([r"Type $1$", r"Type $2$"])
    ax3.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, 3, :], axis = -1), color = plot[3])
    ax4.plot(run_data.time, np.sum(post_data.enst_flux_per_sec[:, 4, :], axis = -1), color = plot[4])
    ax4.set_xlabel(r"$t$")
    ax4.set_ylabel(r"$\sum_{\theta}\Pi_{\mathcal{C}_{\theta}}$")
    ax4.legend([r"Type $3$", r"Type $4$"])
    ax4.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

    plt.savefig(cmdargs.out_dir + "EnstrophyFluxCompare.png")
    plt.close()



    # -----------------------------------------
    # # --------  Compare Data
    # -----------------------------------------