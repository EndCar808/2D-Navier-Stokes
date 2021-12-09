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

        def __init__(self, in_dir = None, out_dir = None, parallel = False, plotting = False, video = False, triads = False, phases = False, ord_triads = False):
            self.in_dir             = in_dir
            self.out_dir_phases     = out_dir
            self.out_dir_triads     = out_dir
            self.out_dir_ord_triads = out_dir
            self.parallel           = parallel
            self.plotting           = plotting
            self.video              = video 
            self.phases             = phases
            self.triads             = triads
            self.ord_triads         = ord_triads


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o", ["par", "plot", "vid", "phase", "triads" , "ord_triads"])
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

        elif opt in ['--par']:
            ## Read in parallel indicator
            cargs.parallel = True

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--vid']:
            ## Read in spectra file
            cargs.video = True

        elif opt in ['--phase']:
            ## Read in plotting indicator
            cargs.phases = True

        elif opt in ['--triads']:
            ## Read in spectra file
            cargs.triads = True

        elif opt in ['--ord_triads']:
            ## Read in spectra file
            cargs.ord_triads = True

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
    method  = "default"

    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir, method)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars, method)

    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars, method)

    ## Read in post processing data
    post_data = import_post_processing_data(cmdargs.in_dir, sys_vars, method)

    ## Make output directory for snaps
    cmdargs.out_dir_phases     = cmdargs.out_dir + "PHASE_SYNC_SNAPS/"
    cmdargs.out_dir_triads     = cmdargs.out_dir + "TRIAD_PHASE_SYNC_SNAPS/"
    cmdargs.out_dir_ord_triads = cmdargs.out_dir + "ORDER_TRIAD_PHASE_SYNC_SNAPS/"
    if os.path.isdir(cmdargs.out_dir_phases) != True:
        print("Making folder:" + tc.C + " PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_phases)
    if os.path.isdir(cmdargs.out_dir_triads) != True:
        print("Making folder:" + tc.C + " TRIAD_PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_triads)
    if os.path.isdir(cmdargs.out_dir_ord_triads) != True:
        print("Making folder:" + tc.C + " ORDER_TRIAD_PHASE_SYNC_SNAPS/" + tc.Rst)
        os.mkdir(cmdargs.out_dir_ord_triads)
    print("Phases Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_phases) + tc.Rst)
    print("Triads Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_triads) + tc.Rst)
    print("Ordered Triads Output Folder: "+ tc.C + "{}".format(cmdargs.out_dir_ord_triads) + tc.Rst)
    # -----------------------------------------
    # # --------  Plot Data
    # -----------------------------------------
    # ## Define a wedge increments
    # Nwedge = 40
    # dtheta = np.pi / Nwedge
    # theta  = np.arange(-np.pi / 2., np.pi / 2., dtheta)

    # ## Define the Radius
    # r = int(sys_vars.Nx / 3)

    # ## Define the wavenumbers
    # ky = np.arange(0, r + 1)
    # kx = np.arange(-r + 1, r + 1)

    if cmdargs.plotting:

        ## Start timer
        start = TIME.perf_counter()
        print("\n" + tc.Y + "Printing Snaps..." + tc.Rst + "Total Snaps to Print: [" + tc.C + "{}".format(sys_vars.ndata) + tc.Rst + "]")

        if cmdargs.phases:

            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 10

                ## Create tasks for the process pool
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
                    plot_sector_phase_sync_snaps(i, cmdargs.out_dir_phases, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.phase_R[i, :], post_data.phase_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

        if cmdargs.triads:

            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 10

                ## Create tasks for the process pool
                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, :], post_data.triad_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

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
                    plot_sector_phase_sync_snaps(i, cmdargs.out_dir_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, :], post_data.triad_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

        if cmdargs.ord_triads:

            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 10

                ## Create tasks for the process pool
                groups_args = [(mprocs.Process(target = plot_sector_phase_sync_snaps, args = (i, cmdargs.out_dir_ord_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.triad_R[i, :], post_data.triad_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)) for i in range(sys_vars.ndata))] * proc_lim

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
                    plot_sector_phase_sync_snaps(i, cmdargs.out_dir_ord_triads, post_data.phases[i, :, int(sys_vars.Nx/3 - 1):], post_data.theta, post_data.ord_triad_R[i, :], post_data.ord_triad_Phi[i, :], run_data.time[i], sys_vars.Nx, sys_vars.Ny)

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
            videoName    = cmdargs.out_dir_phases + "PhaseSync_N[{},{}]_u0[{}]_NSECT[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect)
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

        if cmdargs.triads:

            ## Video variables
            framesPerSec = 30
            inputFile    = cmdargs.out_dir_triads + "Phase_Sync_SNAP_%05d.png"
            videoName    = cmdargs.out_dir_triads + "TriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect)
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

        if cmdargs.ord_triads:

            ## Video variables
            framesPerSec = 30
            inputFile    = cmdargs.out_dir_ord_triads + "Phase_Sync_SNAP_%05d.png"
            videoName    = cmdargs.out_dir_ord_triads + "OrderedTriadPhaseSync_N[{},{}]_u0[{}]_NSECT[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0, post_data.num_sect)
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

        ## Start timer
        end = TIME.perf_counter()
        print("Movie Time:" + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)