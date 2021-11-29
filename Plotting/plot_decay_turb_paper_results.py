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
import pyfftw as fftw

from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, enstrophy_spectrum, energy_spectrum, total_energy, fftw_init_2D, fft, ifft

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
        
        def __init__(self, in_dir = None, out_dir = None):
            self.in_dir     = in_dir
            self.out_dir    = out_dir

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o")
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


    return cargs


######################
##       MAIN       ##
######################
if __name__ == '__main__':
  
    # -------------------------------------
    ## --------- Parse Commnad Line
    # -------------------------------------
    cmdargs = parse_cml(sys.argv[1:]) 
    method  = "default"
    
    # -----------------------------------------
    ## --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir, method)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars, method)

    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars, method)

    ## Read in post processing data
    post_data = import_post_processing_data(cmdargs.in_dir, sys_vars, method)
    

    # -----------------------------------------
    ## --------  Plot Data
    # -----------------------------------------
    ## System Measures plot
    fig = plt.figure(figsize = (16, 8))
    kmax = int(sys_vars.Nx/3 + 1)
    kk = np.arange(1, kmax)
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(run_data.time, run_data.tot_enrg[:] / run_data.tot_enrg[0])
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\mathcal{K}(t) / \mathcal{K}(0)$")
    ax1.set_xlim(run_data.time[0], run_data.time[-1])
    ax1.set_ylim(0.7, 1.0)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(run_data.time, run_data.tot_enst[:] / run_data.tot_enst[0])
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\mathcal{E}(t) / \mathcal{E}(0)$")
    ax2.set_xlim(run_data.time[0], run_data.time[-1])
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0.01, 11.0)
    ax2.set_ylim(1e-2, 1.01)
    plt.savefig(cmdargs.out_dir + "DecayTurb_System_Measures.png", bbox_inches = 'tight') 
    plt.close()

    ## Vorticity Field
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 3)
    if sys_vars.T == 1:
        indexes = [1, 3, 10]
    else:
        indexes = [11, 31, 110]
    for i, indx in enumerate(indexes):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(run_data.w[indx, :, :], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
        ax.set_xlabel(r"$y$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0.0, run_data.y[-1])
        ax.set_ylim(0.0, run_data.x[-1])
        ax.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
        ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
        ax.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax.set_title(r"$t = {:0.5f}$".format(run_data.time[indx]))
        div  = make_axes_locatable(ax)
        cbax = div.append_axes("right", size = "10%", pad = 0.05)
        cb   = plt.colorbar(im, cax = cbax)
        cb.set_label(r"$\omega(x, y)$")
    plt.savefig(cmdargs.out_dir + "DecayTurb_VorticityField.png", bbox_inches = 'tight') 
    plt.close()

    ## Energy Spectrum
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]:
        ax1.plot(spec_data.enrg_spectrum[i, 1:kmax])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1, kmax)
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnergySpectrum.png", bbox_inches = 'tight') 
    plt.close()

    ## Vorticity & Velocity Distribution
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]:
        bin_width   = post_data.w_pdf_ranges[i, 1] - post_data.w_pdf_ranges[i, 0]
        bin_centres = (post_data.w_pdf_ranges[i, 1:] + post_data.w_pdf_ranges[i, :-1]) * 0.5
        pdf = post_data.w_pdf_counts[i, :] / (np.sum((post_data.w_pdf_counts[i, :])) * bin_width)
        var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        ax1.plot(bin_centres, pdf)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"PDF")
    ax1.set_yscale('log')
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]])
    ax2 = fig.add_subplot(gs[0, 1])
    for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]:
        bin_width   = post_data.u_pdf_ranges[i, 1] - post_data.u_pdf_ranges[i, 0]
        bin_centres = (post_data.u_pdf_ranges[i, 1:] + post_data.u_pdf_ranges[i, :-1]) * 0.5
        pdf = post_data.u_pdf_counts[i, :] / (np.sum((post_data.u_pdf_counts[i, :])) * bin_width)
        var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        ax2.plot(bin_centres, pdf)
    ax2.set_xlabel(r"$\mathbf{u}$")
    ax2.set_ylabel(r"PDF")
    ax2.set_yscale('log') 
    ax2.set_xlim(0, 1)   
    ax2.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_VelVortPDFs_C.png", bbox_inches='tight') 
    plt.close()

    ## Flux Spectra
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.cumsum(spec_data.enrg_flux_spectrum[-1, :]))
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    ax1.set_title(r"Energy Flux Spectra")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.cumsum(spec_data.enst_flux_spectrum[-1, :]))
    ax2.set_title(r"Enstrophy Flux Spectra")
    ax2.set_xlabel(r"$|\mathbf{k}|$")
    ax2.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    plt.savefig(cmdargs.out_dir + "DecayTurb_FluxSpectra.png", bbox_inches='tight') 
    plt.close()

    ## McWilliams System Measurables
    fig = plt.figure(figsize = (10, 8))
    gs  = GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(run_data.time, run_data.tot_enrg[:])
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\mathcal{K}(t)$")
    ax1.set_xlim(run_data.time[0], run_data.time[-1])
    ax1.set_ylim(0.4, run_data.tot_enrg[0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(run_data.time, run_data.tot_enst[:])
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\mathcal{E}(t)$")
    ax2.set_xlim(run_data.time[0], run_data.time[-1])
    # ax2.set_yscale('log')
    # ax2.set_xscale('log')
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(run_data.time, np.sum(np.arange(1, spec_data.enrg_spectrum.shape[1]) * spec_data.enrg_spectrum[:, 1:], axis = -1) / run_data.tot_enrg[:])
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$\bar{k}$")
    ax3.set_xlim(run_data.time[0], run_data.time[-1])
    plt.savefig(cmdargs.out_dir + "McWilliamsPaper_System_Measures.png", bbox_inches = 'tight') 
    plt.close()


    ## McWilliams Energy Spectra
    fig       = plt.figure(figsize = (8, 10))
    gs        = GridSpec(1, 1)
    ax1       = fig.add_subplot(gs[0, 0])
    kmax      = int(sys_vars.Nx / 3 + 1)
    timesteps = len(run_data.time)
    tindx     = [0, int(timesteps * (2 / 40)), int(timesteps * (5 / 40)), int(timesteps * (10 / 40)), int(timesteps * (20 / 40)), -1]
    for i in tindx:
        ax1.plot(np.arange(1, kmax), spec_data.enrg_spectrum[i, 1:kmax])
    ax1.set_xlabel(r"$|k|$")
    ax1.set_ylabel(r"$\mathcal{K}(|k|)$")
    ax1.set_xlim(1, kmax - 1)
    ax1.set_ylim(1e-8, 1)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend([r"$t = {}$".format(run_data.time[t]) for t in tindx])
    plt.savefig(cmdargs.out_dir + "McWilliamsPaper_EnergySpectra.png", bbox_inches = 'tight') 
    plt.close()

    ## McWilliams Stream Function 
    psi_hat = np.ones((sys_vars.Nx, sys_vars.Nk)) * np.complex(0.0, 0.0)
    psi     = np.zeros((sys_vars.Nx, sys_vars.Ny))
    fig   = plt.figure(figsize = (10, 8))
    gs    = GridSpec(1, 2)
    for idx in [0, 160]:
        for i in range(sys_vars.Nx):
            for j in range(sys_vars.Nk):
                k_sqr = run_data.kx[i]**2 + run_data.ky[j]**2
                if run_data.kx[i] == 0.0 and run_data.ky[j] == 0.0:
                    psi_hat[i, j] = np.complex(0.0, 0.0)
                else:
                    psi_hat[i, j] = run_data.w_hat[idx, i, j] / k_sqr
        psi  = np.fft.irfft2(psi_hat)
        x, y = np.meshgrid(run_data.x, run_data.y)
        ax1  = fig.add_subplot(gs[0, int(idx / 160)])
        ax1.contour(x, y, psi)
        ax1.set_xlabel(r"$y$")
        ax1.set_ylabel(r"$x$")
        ax1.set_title(r"$t = {}$".format(run_data.time[idx]))
    plt.savefig(cmdargs.out_dir + "McWilliamsPaper_StreamFunctionContour.png", bbox_inches = 'tight') 
    plt.close()
    fig = plt.figure(figsize = (8, 10))
    gs  = GridSpec(1, 2, wspace = 0.4)
    for idx in [0, 160]:
        for i in range(sys_vars.Nx):
            for j in range(sys_vars.Nk):
                k_sqr = run_data.kx[i]**2 + run_data.ky[j]**2
                if run_data.kx[i] == 0.0 and run_data.ky[j] == 0.0:
                    psi_hat[i, j] = np.complex(0.0, 0.0)
                else:
                    psi_hat[i, j] = run_data.w_hat[idx, i, j] / k_sqr
        psi = np.fft.irfft2(psi_hat)
        ax1 = fig.add_subplot(gs[0, int(idx / 160)])
        im  = ax1.imshow(psi, extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
        ax1.set_xlabel(r"$y$")
        ax1.set_ylabel(r"$x$")
        ax1.set_xlim(0.0, run_data.y[-1])
        ax1.set_ylim(0.0, run_data.x[-1])
        ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
        ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
        ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax1.set_title(r"$t = {:0.5f}$".format(run_data.time[indx]))
        div  = make_axes_locatable(ax1)
        cbax = div.append_axes("right", size = "10%", pad = 0.05)
        cb   = plt.colorbar(im, cax = cbax)
        cb.set_label(r"$\omega(x, y)$")
        ax1.set_title(r"$t = {}$".format(run_data.time[idx]))
    plt.savefig(cmdargs.out_dir + "McWilliamsPaper_StreamFunction.png", bbox_inches = 'tight') 
    plt.close()

    ## McWilliams Vorticity
    fig = plt.figure(figsize = (8, 10))
    gs  = GridSpec(2, 2, wspace = 0.4, hspace = -0.1)
    ax  = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    timesteps = len(run_data.time)
    tindx     = [0, int(timesteps * (2.5 / 40)), int(timesteps * (16.5 / 40)), int(timesteps * (37.5 / 40))]
    for i, t in enumerate(tindx):
        im = ax[i].imshow(run_data.w[t, :, :], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
        ax[i].set_xlabel(r"$y$")
        ax[i].set_ylabel(r"$x$")
        ax[i].set_xlim(0.0, run_data.y[-1])
        ax[i].set_ylim(0.0, run_data.x[-1])
        ax[i].set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
        ax[i].set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax[i].set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
        ax[i].set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax[i].set_title(r"$t = {:0.5f}$".format(run_data.time[indx]))
        div  = make_axes_locatable(ax[i])
        cbax = div.append_axes("right", size = "10%", pad = 0.05)
        cb   = plt.colorbar(im, cax = cbax)
        cb.set_label(r"$\omega(x, y)$")
        ax[i].set_title(r"$t = {}$".format(run_data.time[t]))
    plt.savefig(cmdargs.out_dir + "McWilliamsPaper_Vorticity.png", bbox_inches = 'tight') 
    plt.close()