#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##
######################
import matplotlib as mpl
# mpl.use('PDF') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
from itertools import zip_longest
import multiprocessing as mprocs
import time as TIME
from subprocess import Popen, PIPE
from numba import njit
import pyfftw as fftw




###############################
##       FUNCTION DEFS       ##
###############################
def plot_snaps(i, w, x, y, t, w_min, w_max, abs_err, err_max, err_min, time_t, tot_en, tot_ens, tot_pal, t_0, t_end):
  

    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(3, 2, hspace = 0.4, wspace = 0.4)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0:2, 0])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu", vmin = w_min, vmax = w_max)
    ax1.set_xlabel(r"$y$")
    ax1.set_ylabel(r"$x$")
    ax1.set_xlim(0.0, y[-1])
    ax1.set_ylim(0.0, x[-1])
    ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_title(r"$t = {:0.5f}$".format(t))

    ## Plot colourbar
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$\omega(x, y)$")

    ##-------------------------
    ## Plot Error   
    ##-------------------------
    ax2 = fig.add_subplot(gs[0:2, 1])
    im2 = ax2.imshow(abs_err, extent = (y[0], y[-1], x[-1], x[0]), cmap = "jet", vmin = err_min, vmax = err_max) ##
    ax2.set_xlabel(r"$y$")
    ax2.set_ylabel(r"$x$")
    ax2.set_xlim(0.0, y[-1])
    ax2.set_ylim(0.0, x[-1])
    ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
    ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
    ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax2.set_title(r"$t = {:0.5f}$".format(t))

    ## Plot colourbar
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_label(r"$|\omega_n - \omega_e|$")
    cb2.formatter.set_powerlimits((0, 0))

    ##-------------------------
    ## Plot System Variables   
    ##-------------------------
    ax3 = fig.add_subplot(gs[2, 0:2])
    ax3.plot(time_t, tot_en)
    ax3.plot(time_t, tot_ens)
    ax3.plot(time_t, tot_pal)
    ax3.set_xlabel(r"$t$")
    ax3.set_xlim(t_0, t_end)
    ax3.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax3.set_yscale("log")
    ax3.legend([r"Total Energy", r"Total Enstrophy", r"Total Palinstrophy"])

    ## Save figure
    plt.savefig(output_dir + "SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()


def TaylorGreen(x, y, t):

    ## Vorticty in real space
    w = 2. * KAPPA * np.cos(KAPPA * x) * np.cos(KAPPA * y[:, np.newaxis]) * np.exp(- 2 * KAPPA**2 * nu * t)

    return w
######################
##       MAIN       ##
######################
if __name__ == '__main__':


    #--------------------------------
    ## --------- System Parameters
    #--------------------------------
    Nx = 128
    Ny = 128
    int_iters = 4767
    u0 = "TAYLOR_GREEN"
    kymax = int(2 * Ny / 3)
    kxmax = int(2 * Nx / 3)

    nu    = 1.
    KAPPA = 1.

    #-------------------------------
    ## --------- Directories
    #-------------------------------
    input_dir  = "../Data/Test/" # /work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/2D_NavierStokes/
    output_dir = "../Data/Test/SNAPS_N[{},{}]_ITERS[{}]_u0[{}]/".format(Nx, Ny, int_iters, u0) # /work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/2D_NavierStokes/

    if os.path.isdir(output_dir) != True:
        print("Making folder: SNAPS_N[{},{}]_ITERS[{}]_u0[{}]/".format(Nx, Ny, int_iters, u0))
        os.mkdir(output_dir)
    #------------------------------------
    # -------- Open File & Read In data
    #------------------------------------
    with h5py.File(input_dir + "Test_N[{},{}]_ITERS[{}].h5".format(Nx, Ny, int_iters), 'r') as file:
        ## Get the number of data saves
        num_saves = len([g for g in list(file.keys()) if 'Iter' in g])

        ## Allocate arrays
        w       = np.zeros((num_saves, Nx, Ny))
        tg_soln = np.zeros((num_saves, Nx, Ny))
        w_hat   = np.ones((num_saves, Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)
        time    = np.zeros((num_saves, ))
        Real    = 0
        Fourier = 0

        # Read in the vorticity
        for i, group in enumerate(file.keys()):
            if "Iter" in group:
                if 'w' in list(file[group].keys()):
                    w[i, :, :] = file[group]["w"][:, :]
                    Real = 1
                if 'w_hat' in list(file[group].keys()):
                    w_hat[i, :, :] = file[group]["w_hat"][:, :]
                    Fourier = 1
                if 'TGSoln' in list(file[group].keys()):
                    tg_soln[i, :, :] = file[group]["TGSoln"][:, :]
                    Fourier = 1
                time[i] = file[group].attrs["TimeValue"]
            else:
                continue

        # Define min and max for plotiting
        w_min = np.amin(w)
        w_max = np.amax(w)
   
        ## Read in the space arrays
        if 'kx' in list(file.keys()):
            kx = file["kx"][:]
        if 'ky' in list(file.keys()):
            ky = file["ky"][:]
        if 'x' in list(file.keys()):
            x  = file["x"][:]
        if 'y' in list(file.keys()):
            y  = file["y"][:]
        ## Read system measures
        if 'TotalEnergy' in list(file.keys()):
            tot_energy = file['TotalEnergy'][:]
        if 'TotalEnstrophy' in list(file.keys()):
            tot_enstr = file['TotalEnstrophy'][:]
        if 'TotalPalinstrophy' in list(file.keys()):
            tot_palin = file['TotalPalinstrophy'][:]

    ## Precomute error for plotting later
    abs_err = np.zeros((num_saves, Nx, Ny))
    for i in range(num_saves):
        abs_err[i, :, :] = np.absolute(w[i, :, :] - TaylorGreen(x, y, time[i]))
    ## Get max and min error
    err_max = np.amax(abs_err)
    err_min = np.amin(abs_err)
    print(err_max)
    print(err_min)

    # t = 10
    # for i in range(5):
    #     for j in range(5):
    #         print("Er[{}, {}]: {} ".format(i, j, abs_err[t, i, j]), end ="")
    #     print()
    # print()
    # for i in range(5):
    #     for j in range(5):
    #         print("CEr[{}, {}]: {} ".format(i, j, w[t, i, j] - tg_soln[t, i, j]), end ="")
    #     print()

    #-----------------------
    # ------ Plot Snaps
    #-----------------------
    # Start timer
    # start = TIME.perf_counter()

    # ## No. of processes
    # proc_lim = 1

    # ## Create tasks for the process pool
    # groups_args = [(mprocs.Process(target = plot_snaps, args = (i, w[i, :, :], x, y, time[i], w_min, w_max, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1])) for i in range(w.shape[0]))] * proc_lim

    # ## Loop of grouped iterable
    # for procs in zip_longest(*groups_args): 
    #     pipes     = []
    #     processes = []
    #     for p in filter(None, procs):
    #         recv, send = mprocs.Pipe(False)
    #         processes.append(p)
    #         pipes.append(recv)
    #         p.start()

    #     for process in processes:
    #         process.join()

    # ## End timer
    # end = TIME.perf_counter()
    # plot_time = end - start
    # print("\n\nPlotting Time: {:5.8f}s\n\n".format(plot_time))

    
    # for i in range(num_saves):
    #     plot_snaps(i, w[i, :, :], x, y, time[i], w_min, w_max, abs_err[i, :, :], err_min, err_max, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1])


    # #-----------------------
    # # ----- Make Video
    # #-----------------------
    framesPerSec = 15
    inputFile    = output_dir + "SNAP_%05d.png"
    videoName    = output_dir + "2D_NavierStokes_N[{},{}]_ITERS[{}]_u0[{}].mp4".format(Nx, Ny, int_iters, u0)
    cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
    # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

    ## Start timer
    start = TIME.perf_counter()

    process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
    [runCodeOutput, runCodeErr] = process.communicate()
    print(runCodeOutput)
    print(runCodeErr)
    process.wait()


    print("Finished making video...")
    print("Video Location...")
    print("\n" + videoName + "\n")

    ## Start timer
    end = TIME.perf_counter()

    print("\n\nPlotting Time: {:5.8f}s\n\n".format(plot_time))
    print("Movie Time: {:5.8f}s\n\n".format(end - start))