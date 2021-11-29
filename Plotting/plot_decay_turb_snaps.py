#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
import getopt
from itertools import zip_longest
import multiprocessing as mprocs
import time as TIME
from subprocess import Popen, PIPE
from numba import njit
import pyfftw as fftw

from functions import tc, sim_data, import_data, import_spectra_data
from plot_functions import plot_decay_snaps, plot_decay_snaps_2


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
        
        def __init__(self, in_dir = None, out_dir = None, main_file = None, spec_file = None, base_snap = False, full_snap = False, parallel = False, plotting = False, video = False):
            self.spec_file  = spec_file
            self.main_file  = main_file
            self.in_dir     = in_dir
            self.out_dir    = out_dir
            self.base_snap  = base_snap
            self.full_snap  = full_snap
            self.parallel   = parallel
            self.plotting   = plotting
            self.video      = video 

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:m:s:", ["full_snap", "base_snap", "par", "plot", "vid"])
    except:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:
        
        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("Input Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

        elif opt in ['-o']:
            ## Read in output directory
            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['-m']:
            ## Read in main file
            cargs.main_file = str(arg)
            print("Main File: " + tc.C + "{}".format(cargs.main_file) + tc.Rst)

        elif opt in ['-s']:
            ## Read in spectra file
            cargs.spec_file = str(arg)
            print("Spectra File: " + tc.C + "{}".format(cargs.spec_file) + tc.Rst)

        elif opt in ['--base_snap']:
            ## Read in summary snaps indicator
            cargs.base_snap = True

            ## Make summary snaps output directory 
            cargs.out_dir = cargs.in_dir + "DECAY_SNAPS/"
            if os.path.isdir(cargs.out_dir) != True:
                print("Making folder:" + tc.C + "DECAY_SNAPS/" + tc.Rst)
                os.mkdir(cargs.out_dir)
            print("Output Folder: "+ tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['--full_snap']:
            ## Read in summary snaps indicator
            cargs.full_snap = True

            ## Make summary snaps output directory 
            cargs.out_dir = cargs.in_dir + "DECAY_SNAPS/"
            if os.path.isdir(cargs.out_dir) != True:
                print("Making folder:" + tc.C + "DECAY_SNAPS/" + tc.Rst)
                os.mkdir(cargs.out_dir)
            print("Output Folder: "+ tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['--par']:
            ## Read in parallel indicator
            cargs.parallel = True

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--vid']:
            ## Read in spectra file
            cargs.video = True

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
    if cmdargs.spec_file == None:
        spectra_data = import_spectra_data(cmdargs.in_dir, sys_params, method)
    else:
        spectra_data = import_spectra_data(cmdargs.spec_file, sys_params, method)

    ## Get max and min vorticity
    wmin = np.amin(run_data.w[:, :, :])
    wmax = np.amax(run_data.w[:, :, :])
    ## Get max and min system measures 
    emax  = np.amax(run_data.tot_enrg[:] / run_data.tot_enrg[0])
    enmax = np.amax(run_data.tot_enst[:] / run_data.tot_enst[0])
    pmax  = np.amax(run_data.tot_palin[:] / run_data.tot_palin[0])
    # print(emax, enmax, pmax)
    m_max = np.amax([emax, enmax, pmax])
    emin  = np.amin(run_data.tot_enrg[:] / run_data.tot_enrg[0])
    enmin = np.amin(run_data.tot_enst[:] / run_data.tot_enst[0])
    pmin  = np.amin(run_data.tot_palin[:] / run_data.tot_palin[0])
    m_min = np.amin([emin, enmin, pmin])
    # print(emin, enmin, pmin)

    # -----------------------------------------
    ## ------ Plot Snaps
    # -----------------------------------------  
    if cmdargs.plotting:

        ## Start timer
        start = TIME.perf_counter()
        print("\n" + tc.Y + "Printing Snaps..." + tc.Rst)
        
        ## Print full summary snaps = base + spectra
        if cmdargs.full_snap:
            print("\n" + tc.Y + "Printing Full Snaps..." + tc.Rst)
            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 10

                ## Create tasks for the process pool
                groups_args = [(mprocs.Process(target = plot_decay_snaps_2, args = (cmdargs.out_dir, i, run_data.w[i, :, :], wmin, wmax, m_min, m_max, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Ny, run_data.kx, run_data.ky, spectra_data.enrg_spectrum[:i, :], spectra_data.enst_spectrum[:i, :], run_data.tot_enrg, run_data.tot_enst, run_data.tot_palin)) for i in range(run_data.w.shape[0]))] * proc_lim

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
                for i in range(sys_params.ndata):
                    plot_decay_snaps_2(cmdargs.out_dir, i, run_data.w[i, :, :], wmin, wmax, m_min, m_max, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Ny, run_data.kx, run_data.ky, spectra_data.enrg_spectrum[:i, :], spectra_data.enst_spectrum[:i, :], run_data.tot_enrg, run_data.tot_enst, run_data.tot_palin, sys_params.u0)
                # i = sys_params.ndata - 1
                # plot_decay_snaps_2(cmdargs.out_dir, i, run_data.w[i, :, :], wmin, wmax, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Ny, run_data.kx, run_data.ky, spectra_data.enrg_spectrum[:i, :], spectra_data.enst_spectrum[:i, :], run_data.tot_enrg, run_data.tot_enst, run_data.tot_palin)

        ## Print base summary snaps
        if cmdargs.base_snap:
            print("\n" + tc.Y + "Printing Base Snaps..." + tc.Rst)
            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 10

                ## Create tasks for the process pool
                groups_args = [(mprocs.Process(target = plot_decay_snaps, args = (cmdargs.out_dir, i, run_data.w[i, :, :], wmin, wmax, m_min, m_max, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Ny, run_data.kx, run_data.ky, run_data.tot_enrg, run_data.tot_enst, run_data.tot_palin)) for i in range(run_data.w.shape[0]))] * proc_lim

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
                for i in range(sys_params.ndata):
                    plot_decay_snaps(cmdargs.out_dir, i, run_data.w[i, :, :], wmin, wmax, m_min, m_max, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Ny, run_data.kx, run_data.ky, run_data.tot_enrg, run_data.tot_enst, run_data.tot_palin)
                # i = sys_params.ndata - 1
                # plot_decay_snaps(cmdargs.out_dir, i, run_data.w[i, :, :], wmin, wmax, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Ny, run_data.kx, run_data.ky, run_data.tot_enrg, run_data.tot_enst, run_data.tot_palin)

        ## End timer
        end = TIME.perf_counter()
        plot_time = end - start
        print("\n" + tc.Y + "Finished Plotting..." + tc.Rst)
        print("\n\nPlotting Time: " + tc.C + "{:5.8f}s\n\n".format(plot_time) + tc.Rst)



    #------------------------------------
    # ----- Make Video
    #-------------------------------------
    if cmdargs.video:

        ## Start timer
        start = TIME.perf_counter()

        if cmdargs.full_snap:
            framesPerSec = 30
            inputFile    = cmdargs.out_dir + "Decay2_SNAP_%05d.png"
            videoName    = cmdargs.out_dir + "2D_FULL_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_params.Nx, sys_params.Ny, sys_params.u0)
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

        if cmdargs.base_snap:
            framesPerSec = 30
            inputFile    = cmdargs.out_dir + "Decay_SNAP_%05d.png"
            videoName    = cmdargs.out_dir + "2D_BASE_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_params.Nx, sys_params.Ny, sys_params.u0)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            ## Start timer
            start = TIME.perf_counter()

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

    ## Print summary of timmings to screen
    if cmdargs.plotting:
        print("\n\nPlotting Time: " + tc.C + " {:5.8f}s\n\n".format(plot_time) + tc.Rst)
    if cmdargs.video:
        print("Movie Time: " + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)
