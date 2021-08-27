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
def plot_snaps(i, w, w_hat, x, y, t, w_min, w_max, kx, ky, kx_max, time_t, tot_en, tot_ens, tot_pal, t_0, t_end, Nx, Ny):
    
    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(3, 2, hspace = 0.4, wspace = 0.4)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0:2, 0:1])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu") #, vmin = w_min, vmax = w_max
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

    #-------------------------
    # Plot Energy Spectrum   
    #-------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    engy_spec, engy_spec_sum = energy_spectrum(w_hat, kx, ky, Nx, Ny)
    krange = np.arange(0, engy_spec.shape[0])
    ax2.plot(engy_spec[1:kx_max] / engy_spec_sum)  # kx[1:kx_max],
    # kpower = (1 / (kx[11:61] ** 4))
    # ax2.plot(kx[11:61], kpower , linestyle = ':', linewidth = 0.5, color = 'black')
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$E(k) / \sum E(k)$")
    ax2.set_title(r"Energy Spectrum")
    # ax2.set_ylim(1e-8, 1)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    # ax2.text(x = kx[61],  y = kpower[-1], s = r"$k^{-4}$")

    #--------------------------
    # Plot Enstrophy Spectrum   
    #--------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    enstr_spec, enstr_spec_sum = enstrophy_spectrum(w_hat, Nx, Ny)
    krange = np.arange(0, enstr_spec.shape[0])
    # ax3.plot(enstr_spec[1:kx_max] / enstr_spec_sum)  # kx[1:kx_max], 
    # kpower = (1 / (np.power(kx[11:61], 5 / 3)))
    # ax3.plot(kx[11:61], kpower, linestyle = ':', linewidth = 0.5, color = 'black')
    ax3.set_xlabel(r"$k$")
    ax3.set_ylabel(r"$S(k) / \sum S(k)$")
    ax3.set_title(r"Enstrophy Spectrum")
    # ax3.set_ylim(1e-5, 1)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    # ax3.text(x = kx[61],  y = kpower[-1], s = r"$k^{-5/3}$")


    #--------------------------
    # Plot System Measures   
    #--------------------------
    ax4 = fig.add_subplot(gs[2, 0:])
    ax4.plot(time_t, tot_en)
    ax4.plot(time_t, tot_ens)
    ax4.plot(time_t, tot_pal)
    ax4.set_xlabel(r"$t$")
    ax4.set_xlim(t_0, t_end)
    ax4.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax4.set_yscale("log")
    ax4.legend([r"Total Energy", r"Total Enstrophy", r"Total Palinstrophy"])

    ## Save figure
    plt.savefig(output_dir + "SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()

    
@njit
def energy_spectrum(w_h, kx, ky, Nx, Ny):

    ## Spectrum size
    spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)))

    ## Velocity arrays
    energy_spec = np.zeros(spec_size)

    ## Find u_hat
    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):
            ## Compute prefactor
            k_sqr = np.complex(0.0, 1.0) / (kx[i] ** 2 + ky[j] ** 2 + 1e-50)

            ## Compute Fourier velocities
            u_hat = ky[j] * k_sqr * w_h[i, j]
            v_hat = -kx[i] * k_sqr * w_h[i, j]

            ## Compute the mode
            spec_indx = int(np.sqrt(kx[i] * kx[i] + ky[j] * ky[j]))

            ## Update spectrum sum for current mode
            energy_spec[spec_indx] += np.absolute(u_hat * np.conjugate(u_hat)) + np.absolute(v_hat * np.conjugate(v_hat))


    return energy_spec, np.sum(energy_spec)

@njit
def enstrophy_spectrum(w_h, Nx, Ny):

      ## Spectrum size
    spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)) + 1)

    ## Velocity arrays
    enstrophy_spec = np.zeros(spec_size)

    # Find u_hat
    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):
            ## Compute the mode
            spec_indx = int(np.sqrt(kx[i] * kx[i] + ky[j] * ky[j]))
            
            ## Update the spectrum sum for the current mode
            enstrophy_spec[spec_indx] = np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))


    return enstrophy_spec, np.sum(enstrophy_spec)

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


    #--------------------------------
    ## --------- System Parameters
    #--------------------------------
    Nx = 128
    Ny = 128
    int_iters = 934123
    u0 = "DECAY_TURB"
    kymax = int(2 * Ny / 3)
    kxmax = int(2 * Nx / 3)

    cfl = 1.73

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
    with h5py.File(input_dir + "HDF_N[{},{}]_ITERS[{}]_CFL[{}].h5".format(Nx, Ny, int_iters, cfl), 'r') as file:
        ## Get the number of data saves
        num_saves = len([g for g in list(file.keys()) if 'Iter' in g])

        ## Allocate arrays
        w       = np.zeros((num_saves, Nx, Ny))
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

    #-----------------------
    ## ------ Transform w
    #-----------------------
    if Real == 1 and Fourier == 0:
        w_hat = transform_w(w)

    # w_hat_py = transform_w(w)
    # for i in range(5):
    #     for j in range(5):
    #         print("wh[{}]: {} {}I \twhpy[{}]: {} {}I".format(i * w_hat.shape[0] + j, np.real(w_hat[10, i, j]), np.imag(w_hat[10, i, j]),  i * w_hat.shape[0] + j, np.real(w_hat_py[10, i, j]), np.imag(w_hat_py[10, i, j])))
    # print(np.allclose(w_hat, w_hat_py))
    # print(w[0, :, :])
    # print(w_hat[0, :, :])
    # enstr_spec, enstr_spec_sum = enstrophy_spectrum(w_hat[0, :], kx_max)
    # energy_spec, energy_spec_sum = energy_spectrum(w_hat[0, :, :], kx, ky, kxmax)
    # print(energy_spec)


    # i = 0
    # plot_snaps(i, w[i, :, :], w_hat[i, :, :], x, y, time[i], w_min, w_max, kx, ky, kxmax, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1], Nx, Ny)
    # i = 1000
    # plot_snaps(i, w[i, :, :], w_hat[i, :, :], x, y, time[i], w_min, w_max, kx, ky, kxmax, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1], Nx, Ny)
    # i = num_saves - 1
    # plot_snaps(i, w[i, :, :], w_hat[i, :, :], x, y, time[i], w_min, w_max, kx, ky, kxmax, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1], Nx, Ny)



    for i in range(num_saves):
        plot_snaps(i, w[i, :, :], w_hat[i, :, :], x, y, time[i], w_min, w_max, kx, ky, kxmax, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1], Nx, Ny)


    # -----------------------
    ## ------ Plot Snaps
    # -----------------------
    ## Start timer
    # start = TIME.perf_counter()

    # ## No. of processes
    # proc_lim = 1

    # ## Create tasks for the process pool
    # groups_args = [(mprocs.Process(target = plot_snaps, args = (i, w[i, :, :], w_hat[i, :, :], x, y, time[i], w_min, w_max, kx, ky, kxmax, time[:i], tot_energy[:i], tot_enstr[:i], tot_palin[:i], time[0], time[-1], Nx, Ny)) for i in range(w.shape[0]))] * proc_lim

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
        


    #-----------------------
    # ----- Make Video
    #-----------------------
    # framesPerSec = 15
    # inputFile    = output_dir + "SNAP_%05d.png"
    # videoName    = output_dir + "2D_NavierStokes_N[{},{}]_ITERS[{}]_u0[{}].mp4".format(Nx, Ny, int_iters, u0)
    # cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)
    # # cmd = "ffmpeg -r {} -f image2 -s 1280×720 -i {} -vcodec libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)

    # ## Start timer
    # start = TIME.perf_counter()

    # process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
    # [runCodeOutput, runCodeErr] = process.communicate()
    # print(runCodeOutput)
    # print(runCodeErr)
    # process.wait()


    # print("Finished making video...")
    # print("Video Location...")
    # print("\n" + videoName + "\n")

    # ## Start timer
    # end = TIME.perf_counter()

    # print("\n\nPlotting Time: {:5.8f}s\n\n".format(plot_time))
    # print("Movie Time: {:5.8f}s\n\n".format(end - start))
    # 