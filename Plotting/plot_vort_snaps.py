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
from plot_functions import plot_phase_snaps, plot_summary_snaps


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
        
        def __init__(self, in_dir = None, out_dir = None, phase_dir = None, main_file = None, spec_file = None, summ_snap = False, phase_snap = False, parallel = False, plotting = False, video = False):
            self.spec_file  = spec_file
            self.main_file  = main_file
            self.in_dir     = in_dir
            self.out_dir    = out_dir
            self.phase_dir  = phase_dir
            self.summ_snap  = summ_snap
            self.phase_snap = phase_snap
            self.parallel   = parallel
            self.plotting   = plotting
            self.video      = video 

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:m:s:", ["s_snap", "p_snap", "par", "plot", "vid"])
        print(opts, args)
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

        elif opt in ['--s_snap']:
            ## Read in summary snaps indicator
            cargs.summ_snap = True

            ## Make summary snaps output directory 
            cargs.out_dir = cargs.in_dir + "SNAPS/"
            if os.path.isdir(cargs.out_dir) != True:
                print("Making folder:" + tc.C + " SNAPS/" + tc.Rst)
                os.mkdir(cargs.out_dir)
            print("Output Folder: "+ tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['--p_snap']:
            ## Read in phase_snaps indicator
            cargs.phase_snap = True

            ## Make phase snaps output directory
            cargs.phase_dir = cargs.in_dir + "PHASE_SNAPS/"
            if os.path.isdir(cargs.phase_dir) != True:
                print("Making folder:" + tc.C + " PHASE_SNAPS/" + tc.Rst)
                os.mkdir(cargs.phase_dir)
            print("Phases Output Folder: "+ tc.C + "{}".format(cargs.phase_dir) + tc.Rst)

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

@njit
def FullFieldShifted(w_h, kx, ky):

    ## Allocate memory
    kmax     = int(w_h.shape[0] / 3)
    w_h_full = np.ones((int(2 * kmax - 1), int(2 * kmax - 1))) * np.complex(0., 0.)

    for i in range(w_h.shape[0]):
        if abs(kx[i]) < kmax:
            for j in range(w_h.shape[1]):
                if abs(ky[j]) < kmax:
                    if np.sqrt(kx[i]**2 + ky[j]**2) < kmax:
                        if ky[j] == 0: 
                            w_h_full[kmax - 1 + kx[i], kmax - 1 + ky[j]] = w_h[i, j]
                        else:
                            w_h_full[kmax - 1 + kx[i], kmax - 1 + ky[j]] = w_h[i, j]
                            w_h_full[kmax - 1 - kx[i], kmax - 1 - ky[j]] = np.conjugate(w_h[i, j])

    return w_h_full

@njit
def ZeroCentred(w_h):

    ## Get dims
    Nk = w_h.shape[1]
    
    ## Make kx > 0
    tmp1 = np.concatenate((np.flipud(np.conjugate(w_h[:Nk, -2:0:-1])), np.flipud(w_h[:Nk, :])), axis = 1)

    ## Make ky < 0
    tmp2 = np.concatenate((np.flipud(np.conjugate(w_h[Nk:, -2:0:-1])), np.flipud(w_h[Nk:, :])), axis = 1)

    return np.concatenate((tmp1, tmp2), axis = 0)


def transform_w(w):

    ## Initialize the Fourier space vorticity
    w_hat = np.ones((w.shape[0], w.shape[1], int(w.shape[1] / 2 + 1))) * np.complex(0.0, 0.0)

    ## Allocate array
    real = fftw.zeros_aligned((w.shape[1], w.shape[2]), dtype = 'float64')

    ## Create the FFTW transform
    fft2_r2c = fftw.builders.rfft2(real)

    for i in range(w.shape[0]):
        w_hat[i, :, :] = fft2_r2c(w[i, :, :])

    return w_hat

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

    wmin = np.amin(run_data.w)
    wmax = np.amax(run_data.w)

    # -----------------------------------------
    ## ------ Plot Snaps
    # -----------------------------------------
    ## Start timer
    start = TIME.perf_counter()
    print("\n" + tc.Y + "Printing Snaps..." + tc.Rst)
    
    ## Print main summary snaps
    if cmdargs.summ_snap and cmdargs.plotting:
        print("\n" + tc.Y + "Printing Summary Snaps..." + tc.Rst)
        if cmdargs.parallel:
            ## No. of processes
            proc_lim = 10

            ## Create tasks for the process pool
            groups_args = [(mprocs.Process(target = plot_summary_snaps, args = (cmdargs.out_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, wmin, wmax, run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.enrg_diss[:i], run_data.enst_diss[:i], run_data.enrg_flux_sbst[:i], run_data.enrg_diss_sbst[:i], run_data.enst_flux_sbst[:i], run_data.enst_diss_sbst[:i], run_data.time, sys_params.Nx, sys_params.Ny)) for i in range(run_data.w.shape[0]))] * proc_lim

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
            ## Loop over snapshots
            for i in range(sys_params.ndata):
                plot_summary_snaps(cmdargs.out_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, wmin, wmax, run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.enrg_diss[:i], run_data.enst_diss[:i], run_data.enrg_flux_sbst[:i], run_data.enrg_diss_sbst[:i], run_data.enst_flux_sbst[:i], run_data.enst_diss_sbst[:i], run_data.time, sys_params.Nx, sys_params.Ny)
    print(cmdargs.phase_snap)
    ## Print phase summary snaps
    if cmdargs.phase_snap and cmdargs.plotting:
        print("\n" + tc.Y + "Printing Phase Snaps..." + tc.Rst)
        if cmdargs.parallel:
            ## No. of processes
            proc_lim = 10

            ## Create tasks for the process pool
            groups_args = [(mprocs.Process(target = plot_phase_snaps, args = (cmdargs.phase_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], wmax, wmin, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Nx, run_data.k2Inv, run_data.kx, run_data.ky)) for i in range(run_data.w.shape[0]))] * proc_lim

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
            ## Loop over snahpshots
            for i in range(sys_params.ndata):
                plot_phase_snaps(cmdargs.phase_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], wmax, wmin, run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Nx, run_data.k2Inv, run_data.kx, run_data.ky)

    ## End timer
    end = TIME.perf_counter()
    plot_time = end - start
    print("\n" + tc.Y + "Finished Plotting..." + tc.Rst)
    print("\n\nPlotting Time: " + tc.C + "{:5.8f}s\n\n".format(plot_time) + tc.Rst)
        
    #------------------------------------
    # ----- Make Video
    #-------------------------------------
    if cmdargs.video:
        if cmdargs.summ_snap:
            framesPerSec = 15
            inputFile    = cmdargs.out_dir + "SNAP_%05d.png"
            videoName    = cmdargs.out_dir + "2D_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_params.Nx, sys_params.Ny, sys_params.u0)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            ## Start timer
            start = TIME.perf_counter()

            process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
            [runCodeOutput, runCodeErr] = process.communicate()
            print(runCodeOutput)
            print(runCodeErr)
            process.wait()

        if cmdargs.phase_snap:
            framesPerSec = 15
            inputFile    = cmdargs.phase_dir + "Phase_SNAP_%05d.png"
            videoName    = cmdargs.phase_dir + "2D_NavierStokes_N[{},{}]_u0[{}]_Phases.mp4".format(sys_params.Nx, sys_params.Ny, sys_params.u0)
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
        print("Video Location:" + tc.C + videoName + tc.Rst + "\n")

        ## Start timer
        end = TIME.perf_counter()

    ## Print summary of timmings to screen
    print("\n\nPlotting Time:" + tc.C + " {:5.8f}s\n\n".format(plot_time) + tc.Rst)
    print("Movie Time:" + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)
    












    # kx = np.append(np.arange(0, sys_params.Nk), np.linspace(-sys_params.Ny//2 + 1, -1, sys_params.Ny//2 - 1))
    # ky = np.append(np.arange(0, sys_params.Nk), np.linspace(-sys_params.Ny//2 + 1, -1, sys_params.Ny//2 - 1))
    # print()
    # print(kx)
    # print(ky)
    # print()
    # print(np.fft.ifftshift(kx))

    # kx = kx[:, np.newaxis] * np.ones((sys_params.Nx, sys_params.Ny))
    # print(kx)
    # axes = tuple(range(kx.ndim))
    # print(kx.ndim)
    # print(axes)

    # for dim in kx.shape:
    #     print("dim: {}".format(dim))
    # shift = [-(dim // 2 + 1) for dim in kx.shape]
    # print(shift)

    # shift = -(x.shape[axes] // 2)
    # shift = [-(x.shape[ax] // 2) for ax in axes]
    # print()
    # print(np.flipud(fft_ishift_freq(kx)))

    # print()
    # print(np.roll(kx, shift = [-2, -2], axis = (0, 1)))
    # t = 1
    # ffw_h = FullField(run_data.w_hat[t, :, :])
    # print(sys_params.Nx, sys_params.Ny)
    # print(ffw_h.shape)
    # for i in range(ffw_h.shape[0]):
    #     for j in range(ffw_h.shape[1]):
    #         # print("wh[{}, {}]: {:0.5f} {:0.5f}I ".format(i, j, np.real(ffw_h[i, j]), np.imag(ffw_h[i, j])), end = "")
    #         print("wh[{}, {}]: {:0.5f} ".format(i, j, np.angle(ffw_h[i, j])), end = "")
    #     print()
    # print()
    # ffsw_h = FullFieldShifted(run_data.w_hat[t, :, :], run_data.kx, run_data.ky)
    # # for i in range(ffsw_h.shape[0]):
    # #     for j in range(ffsw_h.shape[1]):
    # #         print("wh[{}, {}]: {:0.5f} {:0.5f}I ".format(i, j, np.real(ffsw_h[i, j]), np.imag(ffsw_h[i, j])), end = "")
    # #     print()
    # # print()
    # print(ffw_h.shape)
    # print(ffsw_h.shape)
    # print(np.allclose(ffw_h[abs(run_data.kx) < int(sys_params.Nx/3), abs(run_data.ky) < int(sys_params.Nx/3)], ffsw_h))