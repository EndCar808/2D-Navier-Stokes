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
from subprocess import Popen, PIPE, run
from matplotlib.pyplot import cm
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data
from plot_functions import plot_sector_phase_sync_snaps, plot_sector_phase_sync_snaps_full, plot_sector_phase_sync_snaps_full_sec
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
        flux_min = np.amin(post_data.enst_flux_per_sec[:, :, :])
        flux_max = np.amax(post_data.enst_flux_per_sec[:, :, :])
    # -----------------------------------------
    # # --------  Make Output Directories
    # -----------------------------------------
    ## Make output directory for snaps
    cmdargs.out_dir_phases     = cmdargs.out_dir + "PHASE_SYNC_SNAPS/"
    cmdargs.out_dir_triads     = cmdargs.out_dir + "TRIAD_PHASE_SYNC_SNAPS/"
    if os.path.isdir(cmdargs.out_dir_phases) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_phases)
    if os.path.isdir(cmdargs.out_dir_triads) != True:
        print("Making folder:" + tc.C + " TRIAD_PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_triads)
    print("Phases Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_phases) + tc.Rst)
    print("Triads Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_triads) + tc.Rst)
    # -----------------------------------------
    # # --------  Plot Data
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
                    groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

                else:
                    groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_phases, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.phase_R[i, :], post_data.phase_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

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
                        plot_sector_phase_sync_snaps_full(i, cmdargs.out_dir_phases, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.phase_R[i, :], post_data.phase_Phi[i, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                    else:
                        plot_sector_phase_sync_snaps(i, cmdargs.out_dir_phases, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.phase_R[i, :], post_data.phase_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

        if cmdargs.triads:

            if cmdargs.parallel:
                ## No. of processes
                proc_lim = cmdargs.num_procs

                if cmdargs.triad_type != "all":
                    print("TRIAD TYPE: {}".format(int(cmdargs.triad_type)), end = " ")
                    ## Create tasks for the process pool
                    if cmdargs.full:
                        if cmdargs.triad_plot_type == "sec":
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full_sec, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, int(cmdargs.triad_type), :], post_data.enst_flux_per_sec_across_sec[i, int(cmdargs.triad_type), :, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_R_across_sec[i, int(cmdargs.triad_type), :, :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], post_data.triad_Phi_across_sec[i, int(cmdargs.triad_type), :, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                        else:
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, int(cmdargs.triad_type), :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                    else:
                        groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

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
                    num_triad_types = post_data.triad_R_across_sec.shape[1]
                    for t in range(num_triad_types):
                        print("TRIAD TYPE: {}".format(t), end = " ")
                        ## Create tasks for the process pool
                        if cmdargs.full:
                            if cmdargs.triad_plot_type == "sec":
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full_sec, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, t, :], post_data.enst_flux_per_sec_across_sec[i, t, :, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, t, :], post_data.triad_R_across_sec[i, t, :, :], post_data.triad_Phi[i, t, :], post_data.triad_Phi_across_sec[i, t, :, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                            else:
                                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps_full, args = (i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, t, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim
                        else:
                            groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, t, :], post_data.triad_Phi[i, t, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

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
                        framesPerSec = 30
                        inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
                        videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{}]_KFRAC[{}]_TYPE[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect, post_data.kmax_frac, t)
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
                        run("rm {}".format(cmdargs.out_dir_triads) + "*.png", shell = True)
            else:   
                if cmdargs.triad_type != "all":
                    ## Loop through simulation and plot data
                    for i in range(sys_vars.ndata):
                        ## Plot the data
                        if cmdargs.full:  
                            if cmdargs.full:
                                if cmdargs.triad_plot_type == "sec":
                                    plot_sector_phase_sync_snaps_full_sec(i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, int(cmdargs.triad_type), :], post_data.enst_flux_per_sec_across_sec[i, int(cmdargs.triad_type), :, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_R_across_sec[i, int(cmdargs.triad_type), :, :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], post_data.triad_Phi_across_sec[i, int(cmdargs.triad_type), :, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                                else:
                                    plot_sector_phase_sync_snaps_full(i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                        else:
                            plot_sector_phase_sync_snaps(i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, int(cmdargs.triad_type), :], post_data.triad_Phi[i, int(cmdargs.triad_type), :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)
                else:
                    num_triad_types = post_data.triad_R_across_sec.shape[1]
                    for t in range(num_triad_types):
                        print("TRIAD TYPE: {}".format(t), end = " ")
                        ## Loop through simulation and plot data
                        for i in range(sys_vars.ndata):
                            ## Plot the data
                            if cmdargs.full: 
                                if cmdargs.triad_plot_type == "sec":
                                    plot_sector_phase_sync_snaps_full_sec(i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, t, :], post_data.enst_flux_per_sec_across_sec[i, t, :, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, t, :], post_data.triad_R_across_sec[i, t, :, :], post_data.triad_Phi[i, t, :], post_data.triad_Phi_across_sec[i, t, :, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                                else:
                                    plot_sector_phase_sync_snaps_full(i, cmdargs.out_dir_triads, run_data.w[i, :, :], post_data.enst_spectrum[i, :, :], post_data.enst_flux_per_sec[i, 0, :], post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, t, :], post_data.triad_Phi[i, t, :], flux_min, flux_max, run_data.time[i], run_data.x, run_data.y, sys_vars.Nx, sys_vars.Ny)
                            else:
                                plot_sector_phase_sync_snaps(i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, t, :], post_data.triad_Phi[i, t, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

                        ## Video variables
                        framesPerSec = 30
                        inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
                        videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{}]_KFRAC[{}]_TYPE[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect, post_data.kmax_frac, t)
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
                        run("rm {}".format(cmdargs.out_dir_triads) + "*.png", shell = True)

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
            framesPerSec = 30
            inputFile    = cmdargs.out_dir_phases + "Phase_Sync_SNAP_%05d.png"
            videoName    = cmdargs.out_dir_phases + "PhaseSync_N[{},{}]_u0[{}]_NSECT[{}]_KFRAC[{}]_TAG[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect, post_data.kmax_frac, cmdargs.tag)
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
            run("rm {}".format(cmdargs.out_dir_phases) + "*.png", shell = True)

        if cmdargs.triads and cmdargs.triad_type != "all":

            ## Video variables
            framesPerSec = 30
            inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
            videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{}]_KFRAC[{}]_TYPE[{}]_TAG[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect, post_data.kmax_frac, int(cmdargs.triad_type), cmdargs.tag)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
            [runCodeOutput, runCodeErr] = process.communicate()
            print(runCodeOutput)
            print(runCodeErr)
            process.wait()

            ## Remove the generated snaps after video is created
            run("rm {}".format(cmdargs.out_dir_triads) + "*.png", shell = True)

        ## Start timer
        end = TIME.perf_counter()
        print("Movie Time:" + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)