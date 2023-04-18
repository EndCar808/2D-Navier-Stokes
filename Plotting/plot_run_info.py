#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to compare solver results with decaying turbulence papers
#        Solver data

#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py as h5
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
import concurrent.futures as cf
from subprocess import Popen, PIPE, run
from matplotlib.pyplot import cm
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data
from functions import get_flux_spectrum
from plot_functions import plot_flow_summary, plot_flow_summary_stream, plot_phase_snaps_stream, plot_phase_snaps
from plot_functions import plot_vort
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
            self.in_dir   = in_dir
            self.in_file  = out_dir
            self.plotting = plotting
            self.stream   = False
            self.summ     = False
            self.video    = False
            self.field    = False
            self.parallel = False


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
        print()
    phase_vid_snaps_output_dir = cmdargs.out_dir + "PHASE_VID_SNAPS/"
    if os.path.isdir(phase_vid_snaps_output_dir) != True:
        print("Making folder:" + tc.C + " PHASE_VID_SNAPS/" + tc.Rst)
        os.mkdir(phase_vid_snaps_output_dir)
    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    # Read in 2s spectrum limits from post
    with h5.File(post_file_path) as post_file:
        min_enrg_lim = 1e10
        min_enst_lim = 1e10
        max_enrg_lim = 0.0
        max_enst_lim = 0.0
        for i in range(sys_vars.ndata):
            group_name = "Snap_{:05d}".format(i)
            enrg_spec = post_file[group_name]["FullFieldEnergySpectrum"][:]
            enst_spec = post_file[group_name]["FullFieldEnstrophySpectrum"][:]
            min_enrg  = np.amin(np.delete(enrg_spec.flatten(), np.where(enrg_spec.flatten() == -50.0)))
            max_enrg  = np.amax(np.delete(enrg_spec.flatten(), np.where(enrg_spec.flatten() == -50.0)))
            min_enst  = np.amin(np.delete(enst_spec.flatten(), np.where(enst_spec.flatten() == -50.0)))
            max_enst  = np.amax(np.delete(enst_spec.flatten(), np.where(enst_spec.flatten() == -50.0)))
            min_enrg_lim = np.minimum(min_enrg_lim, min_enrg)
            min_enst_lim = np.minimum(min_enst_lim, min_enst)
            max_enrg_lim = np.maximum(max_enrg_lim, max_enrg)
            max_enst_lim = np.maximum(max_enst_lim, max_enst)
        spec_lims = np.array([min_enrg_lim, max_enrg_lim, min_enst_lim, max_enst_lim])
        spec_lims[spec_lims[:] == 0.0] = 1e-12

    if not cmdargs.stream:
        ## Read in solver data
        run_data = import_data(cmdargs.in_dir, sys_vars)

        ## Read in spectra data
        spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

        with h5.File(cmdargs.in_dir + "SystemMeasures_HDF_Data.h5", "r") as main_file:
            tot_enrg = main_file["TotalEnergy"][:]
            eddy_turn = 2.0 * np.pi / np.sqrt(np.mean(tot_enrg))
        ##-------------------------
        ## Plot vorticity   
        ##-------------------------
        ##------------------------------- Plot vorticity
        snaps_indx = [0, sys_vars.ndata//4, sys_vars.ndata//2, -1]
        plot_vort(snaps_output_dir, run_data.w, run_data.x, run_data.y, run_data.time, snaps_indx)


        # -----------------------------------------
        # # --------  Plot Data
        # -----------------------------------------
        ##---------- Energy Enstrophy
        fig = plt.figure(figsize = (32, 8))
        gs  = GridSpec(2, 3)
        ## Plot the energy dissipation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(run_data.time, run_data.enrg_diss)
        ax1.set_xlabel(r"$t$")
        ax1.set_title(r"Energy Dissipation")
        ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(run_data.time, run_data.enst_diss)
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Enstrophy Dissipation")
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(run_data.time, run_data.tot_enrg)
        ax1.set_xlabel(r"$t$")
        ax1.set_title(r"Total Energy")
        ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(run_data.time, run_data.tot_enst)
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Total Enstrophy")
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(run_data.time, run_data.tot_forc, label=r"Forcing Input")
        ax2.set_xlabel(r"$t$")
        ax2.set_title(r"Total Forcing Input")
        ax2.legend()
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2 = fig.add_subplot(gs[1, 2])
        ax2.plot(run_data.time, run_data.tot_palin)
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
        ax1.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enrg_spectrum[-1, 1:int(sys_vars.Nx//3)], label = "$$")
        ax1.set_xlabel(r"$k$")
        ax1.set_title(r"Energy")
        ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(np.arange(1, int(sys_vars.Nx/3)), spec_data.enst_spectrum[-1, 1:int(sys_vars.Nx//3)], label = "$$")
        ax2.set_xlabel(r"$k$")
        ax2.set_title(r"Energy")
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2.set_yscale('log')
        ax2.set_xscale('log')

        plt.savefig(snaps_output_dir + "Spectra.png")
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
        ax2.set_yscale('symlog')
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2.set_title(r"$\Pi(|\mathbf{k}|)$: Enstrophy Flux Spectrum")
        
        plt.savefig(snaps_output_dir + "TimeAveragedEnstrophySpectra.png")
        plt.close()


        #------------------------------------------
        # Spectra Scaling Exponent
        #------------------------------------------
        k_range        = np.arange(1, int(sys_vars.Nx/3))
        inert_range    = np.arange(9, (sys_vars.Nx//3)//2)
        mean_enrg_spec = np.mean(spec_data.enrg_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
        p_enrg         = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enrg_spec[inert_range]), 1)
        mean_enst_spec = np.mean(spec_data.enst_spectrum[:, 1:int(sys_vars.Nx/3)], axis = 0)
        p_enst         = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enst_spec[inert_range]), 1)

        fig = plt.figure(figsize = (21, 8))
        gs  = GridSpec(1, 2)
        ax2 = fig.add_subplot(gs[0, 0])
        ax2.plot(k_range, mean_enrg_spec, 'k')
        ax2.plot(k_range[inert_range], np.exp(p_enrg[1]) * k_range[inert_range]**p_enrg[0], '--', color='orangered',label=r"$E(k) \propto k^{:.2f}$;".format(p_enrg[0]) + r" $\xi = {:.2f}$".format(np.absolute(p_enrg[0]) - 3)) ## \Rightarrow \propto$ k^{-(3 + \qi)} \Rightarrow \qi = {:.2f} , np.absolute(np.absolute(p_enrg[0]) - 3))
        ax2.set_xlabel(r"$k$")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2.set_title(r"Energy Spectrum: $\mathcal{K}(|\mathbf{k}|) \sim k^{-(3 + \xi)}$")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(k_range, mean_enst_spec, 'k')
        ax2.plot(k_range[inert_range], np.exp(p_enst[1]) * k_range[inert_range]**p_enst[0], '--', color='orangered',label=r"$E(k) \propto k^{:.2f}$;".format(p_enst[0]) + r" $\xi = {:.2f}$".format(np.absolute(p_enst[0]) - 1)) ## \propto$ k^{-(1 + \qi)} \Rightarrow , \qi = {:.2f} np.absolute(np.absolute(p_enst[0]) - 3))
        ax2.set_xlabel(r"$k$")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
        ax2.set_title(r"Enstrophy Spectrum: $\mathcal{E}(|\mathbf{k}|) \sim k^{-(1 + \xi)}$")        
        plt.savefig(snaps_output_dir + "SpectraScaling.png")
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
                emax  = np.amax(run_data.tot_enrg[:])
                enmax = np.amax(run_data.tot_enst[:] / sys_vars.forc_k**2 )
                pmax  = np.amax(run_data.enst_diss[:] / sys_vars.forc_k**2 )
                # print(emax, enmax, pmax)
                emin  = np.amin(run_data.tot_enrg[:])
                enmin = np.amin(run_data.tot_enst[:] / sys_vars.forc_k**2 )
                pmin  = np.amin(run_data.enst_diss[:] / sys_vars.forc_k**2 )
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
                    groups_args = [(mprocs.Process(target = plot_flow_summary, args = (summ_vid_snaps_output_dir, i, run_data.w[i, :, :] / np.sqrt(np.mean(run_data.w[:, :, :]**2)), wmin, wmax, m_min, m_max, enrg_spec_min, enrg_spec_max, enst_spec_min, enst_spec_max, sys_vars.forc_k, run_data.x, run_data.y, run_data.time, sys_vars.Nx, sys_vars.Ny, run_data.kx, run_data.ky, spec_data.enrg_spectrum[i, :], spec_data.enst_spectrum[i, :], run_data.tot_enrg, run_data.tot_enst[:] / (15.5**2), run_data.enst_diss / (15.5**2))) for i in range(run_data.w.shape[0]))] * proc_lim

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
                        plot_flow_summary(summ_vid_snaps_output_dir, i, run_data.w[i, :, :] / np.sqrt(np.mean(run_data.w[:, :, :]**2)), wmin, wmax, m_min, m_max, enrg_spec_min, enrg_spec_max, enst_spec_min, enst_spec_max, sys_vars.forc_k, run_data.x, run_data.y, run_data.time, sys_vars.Nx, sys_vars.Ny, run_data.kx, run_data.ky, spec_data.enrg_spectrum[i, :], spec_data.enst_spectrum[i, :], run_data.tot_enrg, run_data.tot_enst[:] / (15.5**2), run_data.enst_diss / (15.5**2))
            

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
                futures = [executor.submit(plot_phase_snaps_stream, cmdargs.in_dir, phase_vid_snaps_output_dir, cmdargs.in_file, i, sys_vars.Nx, sys_vars.Ny, spec_lims) for i in range(sys_vars.ndata)]

                ## Wait until these jobs are finished
                cf.wait(futures)
                for f in futures:
                    print(f.result())
            else:
                for i in range(sys_vars.ndata):
                    plot_phase_snaps_stream(cmdargs.in_dir, phase_vid_snaps_output_dir, cmdargs.in_file, i, sys_vars.Nx, sys_vars.Ny, spec_lims)


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





