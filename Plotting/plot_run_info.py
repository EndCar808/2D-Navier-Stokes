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
import concurrent.futures as cf
import time as TIME
from subprocess import Popen, PIPE, run
from matplotlib.pyplot import cm
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, import_sys_msr, energy_spectrum, enstrophy_spectrum
from functions import compute_pdf
from plot_functions import plot_flow_summary, plot_flow_summary_stream, plot_phase_snaps_stream, plot_phase_snaps
from plot_functions import plot_vort, plot_time_averaged_spectra_both
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
            self.summ           = False 
            self.field          = False 
            self.stream         = False 


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:", ["video", "par", "plot", "stream", "summ", "field"])
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

        elif opt in ['--stream']:
            ## Read in plotting indicator
            cargs.stream = True

        elif opt in ['--summ']:
            ## Read in plotting indicator
            cargs.summ = True

        elif opt in ['--field']:
            ## Read in plotting indicator
            cargs.field = True


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

    snaps_output_dir = cmdargs.out_dir + "RUN_INFO_SNAPS/"
    if os.path.isdir(snaps_output_dir) != True:
        print("Making folder:" + tc.C + " RUN_INFO_SNAPS/" + tc.Rst)
        os.mkdir(snaps_output_dir)
    summ_vid_snaps_output_dir = cmdargs.out_dir + "SUM_VID_SNAPS/"
    if os.path.isdir(summ_vid_snaps_output_dir) != True:
        print("Making folder:" + tc.C + " SUM_VID_SNAPS/" + tc.Rst)
        os.mkdir(summ_vid_snaps_output_dir)
    phase_vid_snaps_output_dir = cmdargs.out_dir + "PHASE_VID_SNAPS/"
    if os.path.isdir(phase_vid_snaps_output_dir) != True:
        print("Making folder:" + tc.C + " PHASE_VID_SNAPS/" + tc.Rst)
        os.mkdir(phase_vid_snaps_output_dir)
    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    if not cmdargs.stream:
        ## Read in solver data
        run_data = import_data(cmdargs.in_dir, sys_vars)

        ## Read in system measures
        sys_msr = import_sys_msr(cmdargs.in_dir, sys_vars)

        ## Read in spectra data
        spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

        ## Read in post processing data
        post_data = import_post_processing_data(post_file_path, sys_vars, method)

        # -----------------------------------------
        # # --------  Plot Data
        # -----------------------------------------

        ##-------------------------
        ## Plot vorticity   
        ##-------------------------
        ##------------------------------- Plot vorticity
        snaps_indx = [0, sys_vars.ndata//4, sys_vars.ndata//2, -1]
        plot_vort(snaps_output_dir, run_data.w, sys_msr.x, sys_msr.y, sys_msr.time, snaps_indx)

        ##---------- Energy Enstrophy
        fig = plt.figure(figsize = (32, 8))
        gs  = GridSpec(2, 3, hspace = 0.35)
        ## Plot the energy dissipation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(sys_msr.tot_time, sys_msr.enrg_diss)
        ax1.set_xlabel(r"$t$")
        ax1.set_title(r"Energy Dissipation")
        ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ## Plot the enstrophy dissipation
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(sys_msr.tot_time, sys_msr.enst_diss)
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Enstrophy Dissipation")
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ## Plot the relative energy
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(sys_msr.tot_time, sys_msr.tot_enrg)
        ax1.set_xlabel(r"$t$")
        ax1.set_title(r"Total Energy")
        ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ## Plot the relative helicity
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(sys_msr.tot_time, sys_msr.tot_enst)
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Total Enstrophy")
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ## Plot the relative helicity
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(sys_msr.tot_time, sys_msr.tot_enrg_forc, label=r"Energy Forcing Input")
        ax2.plot(sys_msr.tot_time, sys_msr.tot_enst_forc, label=r"Enstrophy Forcing Input")
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Total Forcing Input")
        ax2.legend()
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2 = fig.add_subplot(gs[1, 2])
        ax2.plot(sys_msr.tot_time, sys_msr.tot_palin)
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Total Palinstrophy")
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        plt.savefig(snaps_output_dir + "System_Measures.png")
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


        #------------------------------------------
        # Plot Time Averaged Spectra
        #------------------------------------------
        plot_time_averaged_spectra_both(snaps_output_dir, spec_data.enrg_spectrum, spec_data.enrg_flux_spectrum, sys_vars.Nx//3, spect_type="Energy")
        plot_time_averaged_spectra_both(snaps_output_dir, spec_data.enst_spectrum, spec_data.enst_flux_spectrum, sys_vars.Nx//3, spect_type="Enstrophy")


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


        # -----------------------------------------
        # # --------  Plot Longitudinal Increment PDFs
        # -----------------------------------------
        ## Create Figure
        fig = plt.figure(figsize = (16, 8))
        gs  = GridSpec(1, 2) 

        if post_data.vel_long_incr_ranges.shape[0] == 2:
            plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \pi$"]
        else:
            plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \frac{2\pi}{N}$", r"$r = \frac{4\pi}{N}$", r"$r = \frac{16\pi}{N}$",  r"$r = \pi$"]

        ## Longitudinal PDFs
        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(post_data.vel_long_incr_ranges.shape[0]):
            bin_centres, pdf = compute_pdf(post_data.vel_long_incr_ranges[i, :], post_data.vel_long_incr_counts[i, :], normalized = True)
            ax1.plot(bin_centres, pdf, label = plot_legend[i])
        ax1.set_xlabel(r"$\delta_r \mathbf{u}_{\parallel} / \langle (\delta_r \mathbf{u}_{\parallel})^2 \rangle^{1/2}$")
        ax1.set_ylabel(r"PDF")
        ax1.set_yscale('log')
        ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax1.set_title("Velocity Longitudinal Increments")
        ax1.legend()

        ## Transverse PDFs
        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(post_data.vort_long_incr_ranges.shape[0]):
            bin_centres, pdf = compute_pdf(post_data.vort_long_incr_ranges[i, :], post_data.vort_long_incr_counts[i, :], normalized = True)        
            ax2.plot(bin_centres, pdf, label = plot_legend[i])
        ax2.set_xlabel(r"$\delta_r \omega_{\parallel} / \langle (\delta_r \omega_{\parallel})^2 \rangle^{1/2}$")
        ax2.set_ylabel(r"PDF")
        ax2.set_yscale('log')
        ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax2.set_title("Vorticity Longitudinal Incrments")
        ax2.legend()

        plt.suptitle(r"Longitudinal Increment PDFs")
        
        plt.savefig(snaps_output_dir + "/Longitudinal_Incrmenents_PDFs.png")
        plt.close()


        # -----------------------------------------
        # # --------  Plot Transverse Increment PDFs
        # -----------------------------------------
        ## Create Figure
        fig = plt.figure(figsize = (16, 8))
        gs  = GridSpec(1, 2) 

        if post_data.vel_trans_incr_ranges.shape[0] == 2:
            plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \pi$"]
        else:
            plot_legend = [r"$r = \frac{\pi}{N}$", r"$r = \frac{2\pi}{N}$", r"$r = \frac{4\pi}{N}$", r"$r = \frac{16\pi}{N}$",  r"$r = \pi$"]

        ## Transverse PDFs
        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(post_data.vel_trans_incr_ranges.shape[0]):
            bin_centres, pdf = compute_pdf(post_data.vel_trans_incr_ranges[i, :], post_data.vel_trans_incr_counts[i, :], normalized = True)
            ax1.plot(bin_centres, pdf, label = plot_legend[i])
        ax1.set_xlabel(r"$\delta_r \mathbf{u}_{\perp} / \langle (\delta_r \mathbf{u}_{\perp})^2 \rangle^{1/2}$")
        ax1.set_ylabel(r"PDF")
        ax1.set_yscale('log')
        ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax1.set_title("Velocity Transverse Increments")
        ax1.legend()

        ## Transverse PDFs
        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(post_data.vort_trans_incr_ranges.shape[0]):
            bin_centres, pdf = compute_pdf(post_data.vort_trans_incr_ranges[i, :], post_data.vort_trans_incr_counts[i, :], normalized = True)        
            ax2.plot(bin_centres, pdf, label = plot_legend[i])
        ax2.set_xlabel(r"$\delta_r \omega_{\perp} / \langle (\delta_r \omega_{\perp})^2 \rangle^{1/2}$")
        ax2.set_ylabel(r"PDF")
        ax2.set_yscale('log')
        ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax2.set_title("Vorticity Transverse Incrments")
        ax2.legend()

        plt.suptitle(r"Transverse Increment PDFs")
        
        plt.savefig(snaps_output_dir + "/Transverse_Incrmenents_PDFs.png")
        plt.close()

        # -----------------------------------------
        # # --------  Compare Structure Funcs
        # -----------------------------------------
        fig = plt.figure(figsize = (16, 8))
        gs  = GridSpec(1, 2)
        r = np.arange(1, np.minimum(sys_vars.Nx, sys_vars.Ny) / 2 + 1)
        L = np.minimum(sys_vars.Nx, sys_vars.Ny) / 2

        ## Velocity
        ax1 = fig.add_subplot(gs[0, 0])
        p, = ax1.plot(r / L, 3.0 / 2.0 * (r / L), linestyle = '--', label = r"$\frac{3}{2} \epsilon_l r$")
        ax1.plot(r / L, post_data.mxd_vel_str_func[:], linestyle = '-.', color = p.get_color(), label = r"Mixed; $3\left\langle\delta u_{\parallel}(r)\left[\delta u_{\perp}(r)\right]^2\right\rangle$")
        ax1.plot(r / L, post_data.vel_long_str_func[2, :], linestyle = ':', color = p.get_color(), label = r"Third Order; $\left\langle\left[\delta u_{\|}(r)\right]^3\right\rangle$")
        ax1.set_xlabel(r"$r / L$")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax1.set_title("Holds For Inverse Cascade")
        ax1.legend()

        ## Vorticity
        ax2 = fig.add_subplot(gs[0, 1])
        p, = ax2.plot(r / L, -2.0 * r / L, linestyle = '--', label = r"$-2 \eta_I r$")
        ax2.plot(r / L, post_data.mxd_vel_str_func[:], linestyle = '-.', color = p.get_color(), label = r"Mixed; $\left\langle\delta u_{\|}(r)[\delta \omega(r)]^2\right\rangle$")
        p, = ax2.plot(r / L, 1.0 / 8.0 * (r / L) ** 3, linestyle = '--', label = r"$\frac{1}{8} \eta_I r^3$")
        ax2.plot(r / L, post_data.vel_long_str_func[2, :], '-.', color = p.get_color(), label = r"Third Order; $\left\langle\left[\delta u_{\|}(r)\right]^3\right\rangle$")
        ax2.set_xlabel(r"$r / L$")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(color = 'k', linewidth = .5, linestyle = ':')
        ax2.set_title("Holds For Direct Cascade")
        ax2.legend()

        plt.suptitle(r"Compare Structure Funcitons")
        
        plt.savefig(snaps_output_dir + "/Compare_Structure_Functions.png")
        plt.close()


    if cmdargs.video:

        if cmdargs.summ:
            if cmdargs.stream:
                if cmdargs.parallel:
                    ## Create process Pool executer
                    executor = cf.ProcessPoolExecutor(cmdargs.num_threads)

                    ## Submit jobs to the executor
                    futures = [executor.submit(plot_flow_summary_stream, cmdargs.in_dir, summ_vid_snaps_output_dir, i, sys_vars.Nx, sys_vars.Ny) for i in range(sys_vars.ndata)]

                    ## Wait until these jobs are finished
                    cf.wait(futures)
                    for f in futures:
                        print(f.result())
                else:
                    for i in range(sys_vars.ndata):
                        plot_flow_summary_stream(cmdargs.in_dir, summ_vid_snaps_output_dir, i, sys_vars.Nx, sys_vars.Ny)

            else:
                wmin = np.amin(run_data.w / np.sqrt(np.mean(run_data.w**2)))
                wmax = np.amax(run_data.w / np.sqrt(np.mean(run_data.w**2)))

                sys_vars.forc_k = 2.0

                ## Get max and min system measures 
                emax  = np.amax(sys_msr.tot_enrg[:])
                enmax = np.amax(sys_msr.tot_enst[:] / sys_vars.forc_k**2 )
                pmax  = np.amax(sys_msr.enst_diss[:] / sys_vars.forc_k**2 )
                # print(emax, enmax, pmax)
                emin  = np.amin(sys_msr.tot_enrg[:])
                enmin = np.amin(sys_msr.tot_enst[:] / sys_vars.forc_k**2 )
                pmin  = np.amin(sys_msr.enst_diss[:] / sys_vars.forc_k**2 )
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
                    groups_args = [(mprocs.Process(target = plot_flow_summary, args = (summ_vid_snaps_output_dir, i, run_data.w[i, :, :] / np.sqrt(np.mean(run_data.w[:, :, :]**2)), wmin, wmax, m_min, m_max, enrg_spec_min, enrg_spec_max, enst_spec_min, enst_spec_max, sys_vars.forc_k, sys_msr.x, sys_msr.y, sys_msr.time, sys_vars.Nx, sys_vars.Ny, sys_msr.kx, sys_msr.ky, spec_data.enrg_spectrum[i, :], spec_data.enst_spectrum[i, :], sys_msr.tot_enrg, sys_msr.tot_enst[:] / (15.5**2), sys_msr.enst_diss / (15.5**2))) for i in range(run_data.w.shape[0]))] * proc_lim

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
                        plot_flow_summary(summ_vid_snaps_output_dir, i, run_data.w[i, :, :] / np.sqrt(np.mean(run_data.w[:, :, :]**2)), wmin, wmax, m_min, m_max, enrg_spec_min, enrg_spec_max, enst_spec_min, enst_spec_max, sys_vars.forc_k, sys_msr.x, sys_msr.y, sys_msr.time, sys_vars.Nx, sys_vars.Ny, sys_msr.kx, sys_msr.ky, spec_data.enrg_spectrum[i, :], spec_data.enst_spectrum[i, :], sys_msr.tot_enrg, sys_msr.tot_enst[:] / (15.5**2), sys_msr.enst_diss / (15.5**2))
            

            framesPerSec = 15
            inputFile    = summ_vid_snaps_output_dir + "SNAP_%05d.png"
            videoName    = summ_vid_snaps_output_dir + "2D_FULL_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
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
            run("cd {};".format(summ_vid_snaps_output_dir) + "rm {};".format("./*.png") + "cd -;", shell = True)

        if cmdargs.field:
            # if cmdargs.stream:
            if cmdargs.parallel:
                ## Create process Pool executer
                executor = cf.ProcessPoolExecutor(cmdargs.num_threads)

                ## Submit jobs to the executor
                futures = [executor.submit(plot_phase_snaps_stream, cmdargs.in_dir, phase_vid_snaps_output_dir, cmdargs.in_file, i, sys_vars.Nx, sys_vars.Ny) for i in range(sys_vars.ndata)]

                ## Wait until these jobs are finished
                cf.wait(futures)
                for f in futures:
                    print(f.result())
            else:
                for i in range(sys_vars.ndata):
                    plot_phase_snaps_stream(cmdargs.in_dir, phase_vid_snaps_output_dir, cmdargs.in_file, i, sys_vars.Nx, sys_vars.Ny)


            framesPerSec = 15
            inputFile    = phase_vid_snaps_output_dir + "Phase_SNAP_%05d.png"
            videoName    = phase_vid_snaps_output_dir + "2D_FULL_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
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
            run("cd {};".format(phase_vid_snaps_output_dir) + "rm {};".format("./*.png") + "cd -;", shell = True)
