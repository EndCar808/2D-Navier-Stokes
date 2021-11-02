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

from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, enstrophy_spectrum

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
    sys_params = sim_data(cmdargs.in_dir, method)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_params, method)

    ## Read in spectra data
    spectra_data = import_spectra_data(cmdargs.in_dir, sys_params, method)

    ## Read in post processing data
    post_proc_data = import_post_processing_data(cmdargs.in_dir, sys_params, method)


    # -----------------------------------------
    ## --------  Plot Data
    # -----------------------------------------
    ## System Measures plot
    fig = plt.figure(figsize = (16, 8))
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
    if sys_params.T == 1:
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

    ## Energy Spectrum Test Initial Condition
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    kmax = int(sys_params.Nx/3 + 1)
    kk = np.arange(1, kmax)
    kk_spec = np.ones((kmax, ))
    spec_ic = np.ones((kmax, ))
    for i in range(sys_params.Nx):
        for j in range(sys_params.Nk):
            spec_indx = np.sqrt(run_data.kx[i]**2 + run_data.ky[j]**2)
            if int(spec_indx) < kmax:
                kk_spec[int(spec_indx)] = spec_indx
    print(kk[:10])
    print(kk_spec[:10])
    if sys_params.u0 == "DECAY_TURB_II":
        spec_ic = (kk**6) / ((1 + kk/60)**18) / (10**9.5)
        spec_kk = (kk_spec**6) / ((1 + (kk_spec)/60)**18) / (10**9.5)

    elif sys_params.u0 == "DECAY_TURB":
        spec_ic = (kk) / ((1 + (kk**4)/6)) / (10**1.1)
        spec_kk = (kk_spec) / ((1 + (kk_spec**4)/6)) / (10**1.1)
    ax1.plot(kk, spec_ic)
    ax1.plot(kk, spectra_data.enrg_spectrum[0, 1:kmax])
    ax1.plot(kk, spec_kk[1:])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend([r"Paper", r"IC", r"Py"])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnergySpectrum_TEST_IC.png", bbox_inches = 'tight') 
    plt.close()

    ## Enstrophy Spectrum Test Initial Condition
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    kmax = int(sys_params.Nx/3 + 1)
    kk = np.arange(1, kmax)
    spec_ic = np.ones((kmax - 1, ))
    if sys_params.u0 == "DECAY_TURB_II":
        spec_ic = (kk**6) / ((1 + kk/60)**18) / (10**9.5)
    elif sys_params.u0 == "DECAY_TURB":
        spec_ic = (kk) / ((1 + (kk**4)/6)) / (10**1)
    ax1.plot(kk, spec_ic)
    ax1.plot(kk, spectra_data.enst_spectrum[0, 1:kmax])
    spec, _  = enstrophy_spectrum(run_data.w_hat[0, :, :], run_data.kx, run_data.ky, sys_params.Nx, sys_params.Ny)
    ax1.plot(kk, spec[1:kmax])
    ax1.plot(kk, post_proc_data.enst_spectrum_1d[0, 1:kmax])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend([r"Paper", r"IC", r"Py", r"Post"])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnstrophySpectrum_TEST_IC.png", bbox_inches = 'tight') 
    plt.close()


    ## Energy Spectrum
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]:
        ax1.plot(spectra_data.enrg_spectrum[i, 1:kmax])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1, kmax)
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnergySpectrum.png", bbox_inches = 'tight') 
    plt.close()

    ## Vorticity & Velocity Distribution
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]:
        bin_width   = post_proc_data.w_pdf_ranges[i, 1] - post_proc_data.w_pdf_ranges[i, 0]
        bin_centres = (post_proc_data.w_pdf_ranges[i, 1:] + post_proc_data.w_pdf_ranges[i, :-1]) * 0.5
        pdf = post_proc_data.w_pdf_counts[i, :] / (np.sum((post_proc_data.w_pdf_counts[i, :])) * bin_width)
        var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        ax1.plot(bin_centres, pdf)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"PDF")
    ax1.set_yscale('log')
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]])
    ax2 = fig.add_subplot(gs[0, 1])
    for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]:
        bin_width   = post_proc_data.u_pdf_ranges[i, 1] - post_proc_data.u_pdf_ranges[i, 0]
        bin_centres = (post_proc_data.u_pdf_ranges[i, 1:] + post_proc_data.u_pdf_ranges[i, :-1]) * 0.5
        pdf = post_proc_data.u_pdf_counts[i, :] / (np.sum((post_proc_data.u_pdf_counts[i, :])) * bin_width)
        var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        ax2.plot(bin_centres, pdf)
    ax2.set_xlabel(r"$\mathbf{u}$")
    ax2.set_ylabel(r"PDF")
    ax2.set_yscale('log') 
    ax2.set_xlim(0, 1)   
    ax2.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_VelVortPDFs_C.png", bbox_inches='tight') 
    plt.close()

    ## Flux Spectra
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.cumsum(spectra_data.enrg_flux_spectrum[-1, :]))
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    ax1.set_title(r"Energy Flux Spectra")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.cumsum(spectra_data.enst_flux_spectrum[-1, :]))
    ax2.set_title(r"Enstrophy Flux Spectra")
    ax2.set_xlabel(r"$|\mathbf{k}|$")
    ax2.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    plt.savefig(cmdargs.out_dir + "DecayTurb_FluxSpectra.png", bbox_inches='tight') 
    plt.close()