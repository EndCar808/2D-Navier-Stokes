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

from functions import tc, sim_data, import_data, import_spectra_data, ZeroCentredField, import_post_processing_data
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
        
        def __init__(self, in_dir = None, out_dir = None, phase_dir = None, main_file = None, spec_file = None, summ_snap = False, phase_snap = False, parallel = False, plotting = False, video = False, use_post = False, post_file = None):
            self.spec_file  = spec_file
            self.main_file  = main_file
            self.post_file  = post_file
            self.in_dir     = in_dir
            self.out_dir    = out_dir
            self.phase_dir  = phase_dir
            self.summ_snap  = summ_snap
            self.phase_snap = phase_snap
            self.parallel   = parallel
            self.plotting   = plotting
            self.video      = video 
            self.use_post   = use_post

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:m:s:f:", ["s_snap", "p_snap", "par", "plot", "vid", "use_post"])
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

        elif opt in ['--use_post']:
            ## Read in use of post processing indicator
            cargs.use_post = True

        elif opt in ['-f']:
            ## Read post processing file
            cargs.post_file = str(arg)

            print("Post Processing File: "+ tc.C + "{}".format(cargs.post_file) + tc.Rst)

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


def FullFieldMask(Nx, Ny, k_max):

    if Nx == 2 * k_max - 1 or Ny == 2 * k_max - 1:
        kleftlim  = (Nx + 1) / 2
        krightlim = (Ny + 1) / 2
    else:
        kleftlim = Nx / 2
        krightlim = Ny / 2
    ## Initialize indices
    mask_array = np.full((Nx, Ny), False, dtype = bool)
    x = np.linspace(-kleftlim + 1, krightlim, Nx)
    y = np.linspace(-kleftlim + 1, krightlim, Ny)

    ## Set up meshgrid
    x, y = np.meshgrid(x, y)
    mask = np.sqrt(x**2 + y**2)

    ## Loop through array and set True/False mask
    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j] < k_max:
                mask_array[i, j] = True
            else:
                mask_array[i, j] = False

    return mask_array

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

def run_plot_cmd_par(groups_args):

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

def run_video_cmd(cmd):

    process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
    [runCodeOutput, runCodeErr] = process.communicate()
    print(runCodeOutput)
    print(runCodeErr)
    process.wait()

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
    sys_vars = sim_data(cmdargs.in_dir, method)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars, method)

    ## Read in spectra data
    if cmdargs.spec_file == None and os.path.isfile(cmdargs.in_dir + "Spectra_HDF_Data.h5") == True:
        spec_data = import_spectra_data(cmdargs.in_dir, sys_vars, method)
    elif os.path.isdir(cmdargs.in_dir + "Spectra_HDF_Data.h5") != True:
        pass
    else:
        spec_data = import_spectra_data(cmdargs.spec_file, sys_vars, method)

    # spec_data = import_spectra_data(spectra_file, sys_vars, method)

    # -----------------------------------------
    ## --------  Data Prep
    # -----------------------------------------
    ## Compute real space vorticity if it does not exist in data 
    if run_data.no_w:
        print("\nPreparing real space vorticity...", end = " ")
        for i in range(sys_vars.ndata):
            run_data.w[i, :, :] = np.fft.irfft2(run_data.w_hat[i, :, :])
        print("Finished!")

    ## Read in post data if indicated/it exists 
    if cmdargs.use_post:
        ## Read in post processing data
        post_data = import_post_processing_data(cmdargs.in_dir + cmdargs.post_file, sys_vars, "alt")

        if cmdargs.phase_snap:
            phases        = post_data.phases
            enrg_spectrum = post_data.enrg_spectrum
            enst_spectrum = post_data.enst_spectrum
            
            ## Spectrum limits for plotting
            min_enrg = np.amin(np.delete(enrg_spectrum.flatten(), np.where(enrg_spectrum.flatten() == -50.0)))
            max_enrg = np.amax(np.delete(enrg_spectrum.flatten(), np.where(enrg_spectrum.flatten() == -50.0)))
            min_enst = np.amin(np.delete(enst_spectrum.flatten(), np.where(enst_spectrum.flatten() == -50.0)))
            max_enst = np.amax(np.delete(enst_spectrum.flatten(), np.where(enst_spectrum.flatten() == -50.0)))
            spec_limits = np.array([min_enrg, max_enrg, min_enst, max_enst])
            spec_limits[spec_limits[:] == 0.0] = 1e-12  ## If the limits are zero set to non-zero
            print(spec_limits)

    ## If not using post processing data and phase snaps are needed precompute data
    if cmdargs.phase_snap and not cmdargs.use_post:  
        ## Allocate full field data
        dim = 2 * int(sys_vars.Nx / 3) - 1
        phases         = np.zeros((sys_vars.ndata, dim, dim)) 
        enrg_spectrum  = np.zeros((sys_vars.ndata, dim, dim)) 
        enst_spectrum  = np.zeros((sys_vars.ndata, dim, dim))
        indx = FullFieldMask(sys_vars.Nx, sys_vars.Ny, int(sys_vars.Nx / 3))
        print(indx.shape)
        print(run_data.w_hat[10, :, :].shape)
        for i in range(sys_vars.ndata):
            w_h = ZeroCentredField(run_data.w_hat[i, :, :])
            print(w_h.shape)
            print(w_h[indx].reshape((dim, dim)).shape)
            print(phases[i, :, :].shape)
            phases[i, :, :]        = np.mod(np.angle(w_h[indx]), 2.0 * np.pi)
            enst_spectrum[i, :, :] = np.absolute(w_h[indx]) ** 2
            enrg_spectrum[i, :, :] = np.absolute(w_h[indx]) ** 2
        
        ## Spectrum limits for plotting
        spec_limits = np.array([np.amin(enrg_spectrum), np.amax(enrg_spectrum), np.amin(enst_spectrum), np.amax(enst_spectrum)])
        spec_limits[spec_limits == 0.0] = 1e-6
        print(spec_limits)

    ## Get Max data for plotting
    wmin = np.amin(run_data.w[:, :, :])
    wmax = np.amax(run_data.w[:, :, :])

    # -----------------------------------------
    ## ------ Plot Snaps
    # -----------------------------------------
    if cmdargs.plotting:
        
        ## Start timer
        start = TIME.perf_counter()
        print("\n" + tc.Y + "Printing Snaps..." + tc.Rst + "Total Snaps to Print: [" + tc.C + "{}".format(sys_vars.ndata) + tc.Rst + "]")
        
        ## Print main summary snaps
        if cmdargs.summ_snap:
            print("\n" + tc.Y + "Printing Summary Snaps..." + tc.Rst)
            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 20

                ## Create tasks for the process pool
                cmdlist = [(mprocs.Process(target = plot_summary_snaps, args = (cmdargs.out_dir, i, 
                                                                                run_data.w[i, :, :], 
                                                                                run_data.x, run_data.y, 
                                                                                wmin, wmax, 
                                                                                run_data.kx, run_data.ky, 
                                                                                int(sys_vars.Nx / 3), 
                                                                                run_data.tot_enrg[:i], 
                                                                                run_data.tot_enst[:i], 
                                                                                run_data.tot_palin[:i], 
                                                                                spec_data.enrg_spectrum[i, :], 
                                                                                spec_data.enst_spectrum[i, :], 
                                                                                run_data.enrg_diss[:i], run_data.enst_diss[:i], 
                                                                                run_data.enrg_flux_sbst[:i], run_data.enrg_diss_sbst[:i], 
                                                                                run_data.enst_flux_sbst[:i], run_data.enst_diss_sbst[:i], 
                                                                                run_data.time, 
                                                                                sys_vars.Nx, sys_vars.Ny)) for i in range(run_data.w.shape[0]))] * proc_lim

                ## Run plotting command 
                run_plot_cmd_par(cmdlist)
            else:
                ## Loop over snapshots
                for i in range(1, sys_vars.ndata):
                    plot_summary_snaps(cmdargs.out_dir, i, 
                                        run_data.w[i, :, :], 
                                        run_data.x, run_data.y, 
                                        wmin, wmax, 
                                        run_data.kx, run_data.ky, 
                                        int(sys_vars.Nx / 3), 
                                        run_data.tot_enrg[:i], 
                                        run_data.tot_enst[:i], 
                                        run_data.tot_palin[:i], 
                                        spec_data.enrg_spectrum[i, :], 
                                        spec_data.enst_spectrum[i, :], 
                                        run_data.enrg_diss[:i], run_data.enst_diss[:i], 
                                        run_data.enrg_flux_sbst[:i], run_data.enrg_diss_sbst[:i], 
                                        run_data.enst_flux_sbst[:i], run_data.enst_diss_sbst[:i], 
                                        run_data.time, 
                                        sys_vars.Nx, sys_vars.Ny)

        ## Print phase summary snaps
        if cmdargs.phase_snap:
            print("\n" + tc.Y + "Printing Phase Snaps..." + tc.Rst)
            if cmdargs.parallel:
                ## No. of processes
                proc_lim = 20

                ## Create tasks for the process pool
                cmdlist = [(mprocs.Process(target = plot_phase_snaps, args = (cmdargs.phase_dir, i, 
                                                                              run_data.w_hat[i, :, :], 
                                                                              phases[i, :, :], 
                                                                              enrg_spectrum[i, :, :], 
                                                                              enst_spectrum[i, :, :], 
                                                                              spec_limits, 
                                                                              wmin, wmax, 
                                                                              run_data.x, run_data.y, 
                                                                              run_data.time, 
                                                                              sys_vars.Nx, sys_vars.Nx, 
                                                                              run_data.kx, run_data.ky)) for i in range(run_data.w.shape[0]))] * proc_lim

                ## Run plotting command 
                run_plot_cmd_par(cmdlist)
            else:
                ## Loop over snahpshots
                for i in range(sys_vars.ndata):
                    plot_phase_snaps(cmdargs.phase_dir, i, 
                                     run_data.w[i, :, :], 
                                     phases[i, :, :], 
                                     enrg_spectrum[i, :, :], 
                                     enst_spectrum[i, :, :], 
                                     spec_limits, 
                                     wmin, wmax, 
                                     run_data.x, run_data.y, 
                                     run_data.time, 
                                     sys_vars.Nx, sys_vars.Nx, 
                                     run_data.kx, run_data.ky)

        ## End timer
        end = TIME.perf_counter()
        plot_time = end - start
        print("\n" + tc.Y + "Finished Plotting..." + tc.Rst)
        print("\n\nPlotting Time: " + tc.C + "{:5.8f}s\n\n".format(plot_time) + tc.Rst)
        
    #------------------------------------
    # ----- Make Video
    #------------------------------------
    if cmdargs.video:

        ## Start timer
        start = TIME.perf_counter()

        if cmdargs.summ_snap:
            framesPerSec = 60
            inputFile    = cmdargs.out_dir + "SNAP_%05d.png"
            videoName    = cmdargs.out_dir + "2D_NavierStokes_N[{},{}]_u0[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)


            ## Run make video command
            run_video_cmd(cmd)

            ## Prin summary of timmings to screen
            print("\n" + tc.Y + "Finished making video..." + tc.Rst)
            print("Video Location: " + tc.C + videoName + tc.Rst + "\n")

        if cmdargs.phase_snap:
            framesPerSec = 60
            inputFile    = cmdargs.phase_dir + "Phase_SNAP_%05d.png"
            if cmdargs.use_post:
                videoName    = cmdargs.phase_dir + "2D_POSTDATA_NavierStokes_N[{},{}]_u0[{}]_Phases_new.mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
            else:
                videoName    = cmdargs.phase_dir + "2D_NavierStokes_N[{},{}]_u0[{}]_Phases.mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
            cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
            # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

            ## Start timer
            start = TIME.perf_counter()

            ## Run make video command
            run_video_cmd(cmd)

            ## Prin summary of timmings to screen
            print("\n" + tc.Y + "Finished making video..." + tc.Rst)
            print("Video Location: " + tc.C + videoName + tc.Rst + "\n")

            

        ## Start timer
        end = TIME.perf_counter()



    ## Print summary of timmings to screen
    if cmdargs.plotting:
        print("\n\nPlotting Time:" + tc.C + " {:5.8f}s\n\n".format(plot_time) + tc.Rst)
    if cmdargs.video:
        print("Movie Time:" + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)