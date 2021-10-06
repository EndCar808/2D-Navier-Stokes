#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##
######################
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
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

    phase_snap = True
    summ_snap  = True
    parallel   = True
    plotting   = False
    video      = True


    # -------------------------------------
    ## --------- Directories
    # -------------------------------------
    method = "default"
    ## Read in the data directory provided at CML
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> You must provide directory to data files.")
        sys.exit()
    elif len(sys.argv) == 2:
        ## Get input folder from CML
        input_dir = sys.argv[1]
        print("Input Folder: " + tc.C + "{}".format(input_dir) + tc.Rst)
    elif len(sys.argv) == 3 :
        ## Get file output mode input files from CML
        input_dir = sys.argv[1]
        spectra_dir = sys.argv[2]
        print("Main File: " + tc.C + "{}".format(input_dir) + tc.Rst)
        print("Spectra File: " + tc.C + "{}".format(spectra_dir) + tc.Rst)
        method = "file"
    else: 
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()


    ## Check if output folder exists -> if not make it
    if method != "file":
        output_dir = input_dir + "SNAPS/"
        if os.path.isdir(output_dir) != True:
            print("Making folder:" + tc.C + " SNAPS/" + tc.Rst)
            os.mkdir(output_dir)
        print("Output Folder: "+ tc.C + "{}".format(output_dir) + tc.Rst)
        phases_output_dir = input_dir + "PHASE_SNAPS/"
        if os.path.isdir(phases_output_dir) != True:
            print("Making folder:" + tc.C + " PHASE_SNAPS/" + tc.Rst)
            os.mkdir(phases_output_dir)
        print("Phases Output Folder: "+ tc.C + "{}".format(phases_output_dir) + tc.Rst)


    # -----------------------------------------
    ## --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_params = sim_data(input_dir, method)

    ## Read in solver data
    run_data = import_data(input_dir, sys_params, method)

    ## Read in spectra data
    if method == "file":
        spectra_data = import_spectra_data(spectra_dir, sys_params, method)
    else:
        spectra_data = import_spectra_data(input_dir, sys_params, method)


    # -----------------------------------------
    ## ------ Plot Snaps
    # -----------------------------------------
    ## Start timer
    start = TIME.perf_counter()
    print("\n" + tc.Y + "Printing Snaps..." + tc.Rst)
    
    ## Print main summary snaps
    if summ_snap and plotting:
        print("\n" + tc.Y + "Printing Summary Snaps..." + tc.Rst)
        if parallel:
            ## No. of processes
            proc_lim = 10

            ## Create tasks for the process pool
            groups_args = [(mprocs.Process(target = plot_summary_snaps, args = (output_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.enrg_diss[:i], run_data.enst_diss[:i], run_data.enrg_flux_sbst[:i], run_data.enrg_diss_sbst[:i], run_data.enst_flux_sbst[:i], run_data.enst_diss_sbst[:i], run_data.time, sys_params.Nx, sys_params.Ny)) for i in range(run_data.w.shape[0]))] * proc_lim

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
                plot_summary_snaps(output_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.enrg_diss[:i], run_data.enst_diss[:i], run_data.enrg_flux_sbst[:i], run_data.enrg_diss_sbst[:i], run_data.enst_flux_sbst[:i], run_data.enst_diss_sbst[:i], run_data.time, sys_params.Nx, sys_params.Ny)

    ## Print phase summary snaps
    if phase_snap and plotting:
        print("\n" + tc.Y + "Printing Phase Snaps..." + tc.Rst)
        if parallel:
            ## No. of processes
            proc_lim = 10

            ## Create tasks for the process pool
            groups_args = [(mprocs.Process(target = plot_phase_snaps, args = (phases_output_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Nx, run_data.k2Inv, run_data.kx, run_data.ky)) for i in range(run_data.w.shape[0]))] * proc_lim

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
                plot_phase_snaps(phases_output_dir, i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Nx, run_data.k2Inv, run_data.kx, run_data.ky)

    ## End timer
    end = TIME.perf_counter()
    plot_time = end - start
    print("\n" + tc.Y + "Finished Plotting..." + tc.Rst)
    print("\n\nPlotting Time: " + tc.C + "{:5.8f}s\n\n".format(plot_time) + tc.Rst)
        
    #------------------------------------
    # ----- Make Video
    #-------------------------------------
    if video:
        if summ_snap:
            framesPerSec = 15
            inputFile    = output_dir + "SNAP_%05d.png"
            videoName    = output_dir + "2D_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_params.Nx, sys_params.Ny, sys_params.u0)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            ## Start timer
            start = TIME.perf_counter()

            process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
            [runCodeOutput, runCodeErr] = process.communicate()
            print(runCodeOutput)
            print(runCodeErr)
            process.wait()

        if phase_snap:
            framesPerSec = 15
            inputFile    = phases_output_dir + "Phase_SNAP_%05d.png"
            videoName    = phases_output_dir + "2D_NavierStokes_N[{},{}]_u0[{}]_Phases.mp4".format(sys_params.Nx, sys_params.Ny, sys_params.u0)
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