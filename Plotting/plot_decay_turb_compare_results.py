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
import getopt
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
    ax1.set_ylim(0.8, 1.01)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(run_data.time, run_data.tot_enst[:] / run_data.tot_enst[0])
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\mathcal{E}(t) / \mathcal{E}(0)$")
    ax2.set_xlim(run_data.time[0], run_data.time[-1])
    ax2.set_yscale('log')
    ax2.set_ylim(1e-2, 1.01)
    plt.savefig(cmdargs.out_dir + "DecayTurb_System_Measures.png", bbox_inches='tight') 
    plt.close()

    ## Energy Spectrum
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]:
        ax1.plot(spectra_data.enrg_spectrum[i, :])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnergySpectrum.png", bbox_inches='tight') 
    plt.close()

    ## Vorticity Distribution
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]:
        bin_width   = post_proc_data.w_pdf_ranges[i, 1] - post_proc_data.w_pdf_ranges[i, 0]
        bin_centres = (post_proc_data.w_pdf_ranges[i, 1:] + post_proc_data.w_pdf_ranges[i, :-1]) * 0.5
        pdf = post_proc_data.w_pdf_counts[i, :] / (np.sum((post_proc_data.w_pdf_counts[i, :])) * bin_width)
        var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        ax1.plot(bin_centres / var, pdf * var)
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
        ax2.plot(bin_centres / var, pdf * var)
    ax2.set_xlabel(r"$\mathbf{u}$")
    ax2.set_ylabel(r"PDF")
    ax2.set_yscale('log') 
    ax2.set_xlim(-10, 10)   
    ax2.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_params.ndata/2), sys_params.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_VelVortPDFs.png", bbox_inches='tight') 
    plt.close()

    ## Flux Spectra
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(spectra_data.enrg_flux_spectrum[-1, :])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    ax1.set_title(r"Energy Flux Spectra")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(spectra_data.enst_flux_spectrum[-1, :])
    ax2.set_title(r"Enstrophy Flux Spectra")
    ax2.set_xlabel(r"$|\mathbf{k}|$")
    ax2.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    plt.savefig(cmdargs.out_dir + "DecayTurb_FluxSpectra.png", bbox_inches='tight') 
    plt.close()