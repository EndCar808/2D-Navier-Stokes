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
from functions import tc, import_data #sim_data, import_data, import_spectra_data, import_post_processing_data
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


def u_x(w_hat, k2Inv, ky):

    u_hat_x = np.complex(0.0, 1.0) * k2Inv * ky * w_hat

    return np.fft.irfft2(u_hat_x)
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
    # ## Read in simulation parameters
    # sys_vars = sim_data(cmdargs.in_dir)

    # ## Read in solver data
    # run_data = import_data(cmdargs.in_dir, sys_vars)

    # ## Read in spectra data
    # spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

    # ## Read in post processing data
    # post_data = import_post_processing_data(post_file_path, sys_vars, method)

    # ## Number of triad types
    # num_triad_types = 7
    sys_vars, run_data, spec_data, sync_data, post_data = import_data(cmdargs.in_dir, post_file_path)
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



    # -----------------------------------------------
    # # --------  Compare Fluxes From Post & Solver
    # -----------------------------------------------
    t = -1
    # print(post_data.enrg_flux_spec[t, 1:])
    # print(post_data.enrg_flux_spec[t, 1:] / np.cumsum(spec_data.enrg_flux_spectrum[t, 1:]))
    ## Plot flux comparisons
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(post_data.enst_flux_spec[t, 1:], '.-')
    ax1.plot(np.cumsum(spec_data.enst_flux_spectrum[t, 1:]) * 0.5, '--')
    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$\Pi(k)$")
    ax1.legend([r"Post Enstorphy Flux Spec", r"Solver Enstorphy Flux Spec"])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(post_data.enrg_flux_spec[t, 1:], '.-')
    ax2.plot(np.cumsum(spec_data.enrg_flux_spectrum[t, 1:]) * 0.5, '--')
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$\Pi(k)$")
    ax2.legend([r"Post Energy Flux Spec", r"Solver Energy Flux Spec"])
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(run_data.w[t, :, :] - post_data.w[t, :, :], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax3.set_xlabel(r"$y$")
    ax3.set_ylabel(r"$x$")
    ax3.set_xlim(0.0, run_data.y[-1])
    ax3.set_ylim(0.0, run_data.x[-1])
    ax3.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax3.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_title(r"$t = {:0.5f}$".format(run_data.time[t]))
    
    ## Plot colourbar
    div3  = make_axes_locatable(ax3)
    cbax3 = div3.append_axes("right", size = "10%", pad = 0.05)
    cb3   = plt.colorbar(im3, cax = cbax3)
    cb3.set_label(r"$\omega(x, y)$")
    plt.savefig(cmdargs.out_dir + "FluxCompareSolverPost.png")
    plt.close()

    # -----------------------------------------------
    # # --------  Compare Post & Solver Data
    # -----------------------------------------------
    print(run_data.u[1, :, :, 0])
    # for i in range(20):
    #     print(post_data.u[i, :, :, 0])
    print(run_data.u[1, :, :, 0] - u_x(post_data.w_hat[1, :, :], run_data.k2Inv, run_data.ky))
    print(run_data.u[1, :, :, 0] - post_data.u[1, :, :, 0])
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 3)
    ax3 = fig.add_subplot(gs[0, 0])
    im3 = ax3.imshow(run_data.u[0, :, :, 0] - post_data.u[0, :, :, 0], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax3.set_xlabel(r"$y$")
    ax3.set_ylabel(r"$x$")
    ax3.set_xlim(0.0, run_data.y[-1])
    ax3.set_ylim(0.0, run_data.x[-1])
    ax3.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax3.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
    ## Plot colourbar
    div3  = make_axes_locatable(ax3)
    cbax3 = div3.append_axes("right", size = "10%", pad = 0.05)
    cb3   = plt.colorbar(im3, cax = cbax3)
    cb3.set_label(r"$u_x$")

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(run_data.u[0, :, :, 1] - post_data.u[0, :, :, 1], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax4.set_xlabel(r"$y$")
    ax4.set_ylabel(r"$x$")
    ax4.set_xlim(0.0, run_data.y[-1])
    ax4.set_ylim(0.0, run_data.x[-1])
    ax4.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax4.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax4.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax4.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax4.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
    ## Plot colourbar
    div4  = make_axes_locatable(ax4)
    cbax4 = div4.append_axes("right", size = "10%", pad = 0.05)
    cb4   = plt.colorbar(im4, cax = cbax4)
    cb4.set_label(r"$u_y$")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(run_data.u[int(sys_vars.ndata/2), :, :, 0] - post_data.u[int(sys_vars.ndata/2), :, :, 0], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax2.set_xlabel(r"$y$")
    ax2.set_ylabel(r"$x$")
    ax2.set_xlim(0.0, run_data.y[-1])
    ax2.set_ylim(0.0, run_data.x[-1])
    ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
    ## Plot colourbar
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_label(r"$u_x$")

    ax1 = fig.add_subplot(gs[1, 1])
    im1 = ax1.imshow(run_data.u[int(sys_vars.ndata/2), :, :, 1] - post_data.u[int(sys_vars.ndata/2), :, :, 1], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax1.set_xlabel(r"$y$")
    ax1.set_ylabel(r"$x$")
    ax1.set_xlim(0.0, run_data.y[-1])
    ax1.set_ylim(0.0, run_data.x[-1])
    ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_title(r"$t = {:0.5f}$".format(run_data.time[int(sys_vars.ndata/2)]))
    ## Plot colourbar
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$u_y$")

    ax5 = fig.add_subplot(gs[0, 2])
    im5 = ax5.imshow(run_data.u[-1, :, :, 0] - post_data.u[-1, :, :, 0], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax5.set_xlabel(r"$y$")
    ax5.set_ylabel(r"$x$")
    ax5.set_xlim(0.0, run_data.y[-1])
    ax5.set_ylim(0.0, run_data.x[-1])
    ax5.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax5.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax5.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax5.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax5.set_title(r"$t = {:0.5f}$".format(run_data.time[-1]))
    ## Plot colourbar
    div5  = make_axes_locatable(ax5)
    cbax5 = div5.append_axes("right", size = "10%", pad = 0.05)
    cb5   = plt.colorbar(im5, cax = cbax5)
    cb5.set_label(r"$u_x$")

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(run_data.u[-1, :, :, 1] - post_data.u[-1, :, :, 1], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax6.set_xlabel(r"$y$")
    ax6.set_ylabel(r"$x$")
    ax6.set_xlim(0.0, run_data.y[-1])
    ax6.set_ylim(0.0, run_data.x[-1])
    ax6.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax6.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax6.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax6.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax6.set_title(r"$t = {:0.5f}$".format(run_data.time[-1]))
    ## Plot colourbar
    div6  = make_axes_locatable(ax6)
    cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
    cb6   = plt.colorbar(im6, cax = cbax6)
    cb6.set_label(r"$u_y$")

    plt.savefig(cmdargs.out_dir + "VelocityCompare.png")
    plt.close()


    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 3)
    ax3 = fig.add_subplot(gs[0, 0])
    im3 = ax3.imshow(np.real(run_data.u_hat[0, :, :, 0] - post_data.u_hat[0, :, :, 0]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax3.set_xlabel(r"$y$")
    ax3.set_ylabel(r"$x$")
    ax3.set_xlim(0.0, run_data.y[-1])
    ax3.set_ylim(0.0, run_data.x[-1])
    ax3.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax3.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax3.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
    ## Plot colourbar
    div3  = make_axes_locatable(ax3)
    cbax3 = div3.append_axes("right", size = "10%", pad = 0.05)
    cb3   = plt.colorbar(im3, cax = cbax3)
    cb3.set_label(r"$\Re uhat_x$")

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(np.imag(run_data.u_hat[0, :, :, 1] - post_data.u_hat[0, :, :, 1]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax4.set_xlabel(r"$y$")
    ax4.set_ylabel(r"$x$")
    ax4.set_xlim(0.0, run_data.y[-1])
    ax4.set_ylim(0.0, run_data.x[-1])
    ax4.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax4.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax4.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax4.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax4.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
    ## Plot colourbar
    div4  = make_axes_locatable(ax4)
    cbax4 = div4.append_axes("right", size = "10%", pad = 0.05)
    cb4   = plt.colorbar(im4, cax = cbax4)
    cb4.set_label(r"$\Im uhat_y$")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.real(run_data.u_hat[int(sys_vars.ndata/2), :, :, 0] - post_data.u_hat[int(sys_vars.ndata/2), :, :, 0]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax2.set_xlabel(r"$y$")
    ax2.set_ylabel(r"$x$")
    ax2.set_xlim(0.0, run_data.y[-1])
    ax2.set_ylim(0.0, run_data.x[-1])
    ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
    ## Plot colourbar
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_label(r"$\Re u_x$")

    ax1 = fig.add_subplot(gs[1, 1])
    im1 = ax1.imshow(np.imag(run_data.u_hat[int(sys_vars.ndata/2), :, :, 1] - post_data.u_hat[int(sys_vars.ndata/2), :, :, 1]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax1.set_xlabel(r"$y$")
    ax1.set_ylabel(r"$x$")
    ax1.set_xlim(0.0, run_data.y[-1])
    ax1.set_ylim(0.0, run_data.x[-1])
    ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_title(r"$t = {:0.5f}$".format(run_data.time[int(sys_vars.ndata/2)]))
    ## Plot colourbar
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$\Im u_y$")

    ax5 = fig.add_subplot(gs[0, 2])
    im5 = ax5.imshow(np.real(run_data.u_hat[-1, :, :, 0] - post_data.u_hat[-1, :, :, 0]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax5.set_xlabel(r"$y$")
    ax5.set_ylabel(r"$x$")
    ax5.set_xlim(0.0, run_data.y[-1])
    ax5.set_ylim(0.0, run_data.x[-1])
    ax5.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax5.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax5.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax5.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax5.set_title(r"$t = {:0.5f}$".format(run_data.time[-1]))
    ## Plot colourbar
    div5  = make_axes_locatable(ax5)
    cbax5 = div5.append_axes("right", size = "10%", pad = 0.05)
    cb5   = plt.colorbar(im5, cax = cbax5)
    cb5.set_label(r"$\Re u_x$")

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(np.imag(run_data.u_hat[-1, :, :, 1] - post_data.u_hat[-1, :, :, 1]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax6.set_xlabel(r"$y$")
    ax6.set_ylabel(r"$x$")
    ax6.set_xlim(0.0, run_data.y[-1])
    ax6.set_ylim(0.0, run_data.x[-1])
    ax6.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax6.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax6.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax6.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax6.set_title(r"$t = {:0.5f}$".format(run_data.time[-1]))
    ## Plot colourbar
    div6  = make_axes_locatable(ax6)
    cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
    cb6   = plt.colorbar(im6, cax = cbax6)
    cb6.set_label(r"$\Im u_y$")

    plt.savefig(cmdargs.out_dir + "FourierVelocityCompare.png")
    plt.close()


    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)

    solve_nonlin = np.fft.irfft2(run_data.nonlin[int(sys_vars.ndata/2), :, :])
    post_nonlin = np.fft.irfft2(post_data.nonlin[int(sys_vars.ndata/2), :, :])
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(solve_nonlin - post_nonlin, extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax1.set_xlabel(r"$y$")
    ax1.set_ylabel(r"$x$")
    ax1.set_xlim(0.0, run_data.y[-1])
    ax1.set_ylim(0.0, run_data.x[-1])
    ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_title(r"$t = {:0.5f}$".format(run_data.time[int(sys_vars.ndata/2)]))
    ## Plot colourbar
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$Nonlinear Term$")
    solve_nonlin = np.fft.irfft2(run_data.nonlin[-1, :, :])
    post_nonlin = np.fft.irfft2(post_data.nonlin[-1, :, :])
    ax6 = fig.add_subplot(gs[0, 1])
    im6 = ax6.imshow(solve_nonlin - post_nonlin, extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax6.set_xlabel(r"$y$")
    ax6.set_ylabel(r"$x$")
    ax6.set_xlim(0.0, run_data.y[-1])
    ax6.set_ylim(0.0, run_data.x[-1])
    ax6.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax6.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax6.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax6.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax6.set_title(r"$t = {:0.5f}$".format(run_data.time[-1]))
    ## Plot colourbar
    div6  = make_axes_locatable(ax6)
    cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
    cb6   = plt.colorbar(im6, cax = cbax6)
    cb6.set_label(r"$Nonlinear Term$")

    plt.savefig(cmdargs.out_dir + "NonlinearCompare.png")
    plt.close()
    # -----------------------------------------------
    # # --------  Plot Enstorphy Dissipation Field
    # -----------------------------------------------
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(post_data.enst_diss_field[t, :, :], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max 
    ax1.set_xlabel(r"$y$")
    ax1.set_ylabel(r"$x$")
    ax1.set_xlim(0.0, run_data.y[-1])
    ax1.set_ylim(0.0, run_data.x[-1])
    ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_title(r"$t = {:0.5f}$".format(run_data.time[t]))
    
    ## Plot colourbar
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"Enstrophy Dissipation")
    plt.savefig(cmdargs.out_dir + "EnstrophyDissipationField.png")
    plt.close()

