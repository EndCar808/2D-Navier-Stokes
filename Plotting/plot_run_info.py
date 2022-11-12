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
import matplotlib as mpl
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
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, import_sys_msr, energy_spectrum, enstrophy_spectrum
from plot_functions import plot_flow_summary
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

        def __init__(self, in_dir = None, out_dir = None, in_file = None, plotting = False, video = False, par = False):
            self.in_dir         = in_dir
            self.in_file        = out_dir
            self.plotting       = plotting
            self.video          = video
            self.parallel       = par
            self.num_threads    = 5


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:", ["video", "par", "plot"])
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

        if opt in ['-p']:
            ## Read input directory
            cargs.num_threads = int(arg)

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--video']:
            ## Read in plotting indicator
            cargs.video = True

        elif opt in ['--par']:
            ## Read in plotting indicator
            cargs.parallel = True


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

    ## Read in system measures
    sys_msr_data = import_sys_msr(cmdargs.in_dir, sys_vars)

    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

    snaps_output_dir = cmdargs.out_dir + "RUN_SNAPS/"
    if os.path.isdir(snaps_output_dir) != True:
        print("Making folder:" + tc.C + " RUN_SNAPS/" + tc.Rst)
        os.mkdir(snaps_output_dir)
    vid_snaps_output_dir = cmdargs.out_dir + "VID_SNAPS/"
    if os.path.isdir(vid_snaps_output_dir) != True:
        print("Making folder:" + tc.C + " VID_SNAPS/" + tc.Rst)
        os.mkdir(vid_snaps_output_dir)
    # -----------------------------------------
    # # --------  Plot Data
    # -----------------------------------------
    ##------------------------------- Plot vorticity
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2, hspace = 0.3) 

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = []
    for i in range(2):
        for j in range(2):
            ax1.append(fig.add_subplot(gs[i, j]))
    for j, i in enumerate([0, 50, 100, -1]):
        im1 = ax1[j].imshow(run_data.w[i, :], extent = (sys_msr_data.y[0], sys_msr_data.y[-1], sys_msr_data.x[-1], sys_msr_data.x[0]), cmap = "jet") #, vmin = w_min, vmax = w_max 
        ax1[j].set_xlabel(r"$y$")
        ax1[j].set_ylabel(r"$x$")
        ax1[j].set_xlim(0.0, sys_msr_data.y[-1])
        ax1[j].set_ylim(0.0, sys_msr_data.x[-1])
        ax1[j].set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, sys_msr_data.y[-1]])
        ax1[j].set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax1[j].set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, sys_msr_data.x[-1]])
        ax1[j].set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax1[j].set_title(r"$t = {:0.5f}$".format(sys_msr_data.time[i]))
        ## Plot colourbar
        div1  = make_axes_locatable(ax1[j])
        cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
        cb1   = plt.colorbar(im1, cax = cbax1)
        cb1.set_label(r"$\omega(x, y)$")
    plt.savefig(snaps_output_dir + "Vorticity.png")
    plt.close()

    ##---------- Energy Enstrophy
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(2, 3, hspace = 0.35)
    ## Plot the energy dissipation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sys_msr_data.time, sys_msr_data.enrg_diss)
    ax1.set_xlabel(r"$t$")
    ax1.set_title(r"Energy Dissipation")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the enstrophy dissipation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sys_msr_data.time, sys_msr_data.enst_diss)
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Enstrophy Dissipation")
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the relative energy
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(sys_msr_data.time, sys_msr_data.tot_enrg)
    ax1.set_xlabel(r"$t$")
    ax1.set_title(r"Total Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the relative helicity
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(sys_msr_data.time, sys_msr_data.tot_enst)
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Total Enstrophy")
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ## Plot the relative helicity
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(sys_msr_data.time, sys_msr_data.tot_forc)
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Total Forcing Input")
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.plot(sys_msr_data.time, sys_msr_data.tot_palin)
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Total Palinstrophy")
    ax2.set_yscale('symlog')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    plt.savefig(snaps_output_dir + "System_Measures.png")
    plt.close()


    ##---------- Mean Flow
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sys_msr_data.mean_flow_x, label = "$$")
    ax1.set_ylabel(r"$\bar{u}(y)$")
    ax1.set_xlabel(r"$y$")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax1.set_title(r"Mean Flow in x Direction")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sys_msr_data.x, sys_msr_data.mean_flow_y, label = "$$")
    ax2.set_xlabel(r"$x$")
    ax2.set_xlabel(r"$\bar{v}(y)$")
    ax2.set_title(r"Mean Flow in y Directon")
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    plt.savefig(snaps_output_dir + "MeanFlow.png")
    plt.close()

    ##---------- Spectra
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(spec_data.enrg_spectrum[-1, :], label = "$$")
    ax1.set_xlabel(r"$k$")
    ax1.set_title(r"Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(spec_data.enst_spectrum[-1, :], label = "$$")
    ax2.set_xlabel(r"$k$")
    ax2.set_title(r"Energy")
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.savefig(snaps_output_dir + "Spectra.png")
    plt.close()



    ##---------- Flux Spectra
    fig = plt.figure(figsize = (32, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enrg_flux_spectrum[-1, 1:int(sys_vars.Nx//3)], label = "$$")
    ax1.set_xlabel(r"$k$")
    ax1.set_title(r"Energy")
    ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax1.set_yscale('log')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enst_flux_spectrum[-1, 1:int(sys_vars.Nx//3)], label = "$$")
    ax2.set_xlabel(r"$k$")
    ax2.set_title(r"Energy")
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_yscale('log')
    plt.savefig(snaps_output_dir + "Flux_Spectra.png")
    plt.close()


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

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(spec_data.enst_flux_spectrum.shape[0]):
        ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enst_flux_spectrum[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(spec_data.enst_flux_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\Pi(|\mathbf{k}|)$: Enstrophy Flux Spectrum")
    
    plt.savefig(snaps_output_dir + "TimeAveragedEnstrophySpectra.png")
    plt.close()

    ##------------------------ Time Averaged Energy Spectra and Flux Spectra
    fig = plt.figure(figsize = (21, 8))
    gs  = GridSpec(1, 2)
    ax2 = fig.add_subplot(gs[0, 0])
    for i in range(spec_data.enst_spectrum.shape[0]):
        ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enrg_spectrum[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(spec_data.enrg_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\mathcal{K}(|\mathbf{k}|)$: Energy Spectrum")

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(spec_data.enrg_flux_spectrum.shape[0]):
        ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enrg_flux_spectrum[i, 1:int(sys_vars.Nx/3)], 'r', alpha = 0.15)
    ax2.plot(np.arange(1, int(sys_vars.Nx/3)), np.mean(spec_data.enrg_flux_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0), 'k')
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\Pi(|\mathbf{k}|)$: Energy Flux Spectrum")
    
    plt.savefig(snaps_output_dir + "TimeAveragedEnergySpectra.png")
    plt.close()


    ##------------------------ Time Averaged Spectra to Measure the Spectra Scaling Exponent
    k_range = np.arange(1, int(sys_vars.Nx/3))
    inert_range = np.arange(9, 31)
    mean_enrg_spec = np.mean(spec_data.enrg_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
    mean_enst_spec = np.mean(spec_data.enst_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
    p_enrg = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enrg_spec[inert_range]), 1)
    p_enst = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enst_spec[inert_range]), 1)

    fig = plt.figure(figsize = (21, 8))
    gs  = GridSpec(1, 2)
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.plot(k_range, mean_enrg_spec, 'k')
    ax2.plot(k_range[inert_range], np.exp(p_enrg[1]) * k_range[inert_range]**p_enrg[0], '--', color='orangered',label="$E(k) \propto k^{:.2f}$".format(p_enrg[0])) ## \Rightarrow \propto$ k^{-(3 + \qi)} \Rightarrow \qi = {:.2f} , np.absolute(np.absolute(p_enrg[0]) - 3))
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\mathcal{K}(|\mathbf{k}|)$: Energy Spectrum")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(k_range, mean_enst_spec, 'k')
    ax2.plot(k_range[inert_range], np.exp(p_enst[1]) * k_range[inert_range]**p_enst[0], '--', color='orangered',label="$E(k) \propto k^{:.2f}$".format(p_enst[0])) ## \propto$ k^{-(1 + \qi)} \Rightarrow , \qi = {:.2f} np.absolute(np.absolute(p_enst[0]) - 3))
    ax2.set_xlabel(r"$k$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.set_title(r"$\mathcal{E}(|\mathbf{k}|)$: Enstrophy Spectrum")
    
    plt.savefig(snaps_output_dir + "SpectraScaling.png")
    plt.close()




    if cmdargs.video:

        wmin = np.amin(run_data.w / np.sqrt(np.mean(run_data.w**2)))
        wmax = np.amax(run_data.w / np.sqrt(np.mean(run_data.w**2)))

        sys_vars.forc_k = 2.0

        ## Get max and min system measures 
        emax  = np.amax(sys_msr_data.tot_enrg[:] )
        enmax = np.amax(sys_msr_data.tot_enst[:] / sys_vars.forc_k**2 )
        pmax  = np.amax(sys_msr_data.enst_diss[:] / sys_vars.forc_k**2 )
        # print(emax, enmax, pmax)
        emin  = np.amin(sys_msr_data.tot_enrg[:] )
        enmin = np.amin(sys_msr_data.tot_enst[:] / sys_vars.forc_k**2 )
        pmin  = np.amin(sys_msr_data.enst_diss[:] / sys_vars.forc_k**2 )
        m_max = np.amax([emax, enmax, pmax])
        m_min = np.amin([emin, enmin, pmin])
        # m_max = np.amax([emax, enmax])
        # m_min = np.amin([emin, enmin])
        ## Get the min and max spectra values
        enrg_spec_max = np.amax(spec_data.enrg_spectrum[spec_data.enrg_spectrum != 0.0])
        enrg_spec_min = np.amin(spec_data.enrg_spectrum[spec_data.enrg_spectrum != 0.0])
        enst_spec_max = np.amax(spec_data.enst_spectrum[spec_data.enst_spectrum != 0.0])
        enst_spec_min = np.amin(spec_data.enst_spectrum[spec_data.enrg_spectrum != 0.0])


        ## Start timer
        start = TIME.perf_counter()
        print("\n" + tc.Y + "Printing Snaps..." + tc.Rst)
        
        ## Print full summary snaps = base + spectra
        print("\n" + tc.Y + "Number of SNAPS:" + tc.C + " {}\n".format(sys_vars.ndata) + tc.Rst)
        if cmdargs.parallel:
            ## No. of processes
            proc_lim = cmdargs.num_threads

            ## Create tasks for the process pool
            groups_args = [(mprocs.Process(target = plot_flow_summary, args = (vid_snaps_output_dir, i, run_data.w[i, :, :] / np.sqrt(np.mean(run_data.w[:, :, :]**2)), wmin, wmax, m_min, m_max, enrg_spec_min, enrg_spec_max, enst_spec_min, enst_spec_max, sys_vars.forc_k, sys_msr_data.x, sys_msr_data.y, sys_msr_data.time, sys_vars.Nx, sys_vars.Ny, sys_msr_data.kx, sys_msr_data.ky, spec_data.enrg_spectrum[i, :], spec_data.enst_spectrum[i, :], sys_msr_data.tot_enrg, sys_msr_data.tot_enst[:] / (15.5**2), sys_msr_data.enst_diss / (15.5**2))) for i in range(run_data.w.shape[0]))] * proc_lim

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
            # Loop over snapshots
            for i in range(sys_vars.ndata):
                plot_flow_summary(vid_snaps_output_dir, i, run_data.w[i, :, :] / np.sqrt(np.mean(run_data.w[:, :, :]**2)), wmin, wmax, m_min, m_max, enrg_spec_min, enrg_spec_max, enst_spec_min, enst_spec_max, sys_vars.forc_k, sys_msr_data.x, sys_msr_data.y, sys_msr_data.time, sys_vars.Nx, sys_vars.Ny, sys_msr_data.kx, sys_msr_data.ky, spec_data.enrg_spectrum[i, :], spec_data.enst_spectrum[i, :], sys_msr_data.tot_enrg, sys_msr_data.tot_enst[:] / (15.5**2), sys_msr_data.enst_diss / (15.5**2))
        

        framesPerSec = 15
        inputFile    = vid_snaps_output_dir + "SNAP_%05d.png"
        videoName    = vid_snaps_output_dir + "2D_FULL_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
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
        run("cd {};".format(vid_snaps_output_dir) + "rm {};".format("./*.png") + "cd -;", shell = True)