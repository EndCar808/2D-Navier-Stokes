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


from functions import SimData, ImportData, ImportSpectraData

#################################
## Colour Printing to Terminal ##
#################################
class tc:
    H    = '\033[95m'
    B    = '\033[94m'
    C    = '\033[96m'
    G    = '\033[92m'
    Y    = '\033[93m'
    R    = '\033[91m'
    Rst  = '\033[0m'
    Bold = '\033[1m'
    Underline = '\033[4m'


###############################
##       FUNCTION DEFS       ##
###############################
def plot_snaps(i, w, w_hat, x, y, w_min, w_max, kx, ky, kx_max, tot_en, tot_ens, tot_pal, enrg_spec, enst_spec, time, Nx, Ny):
    
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
    ax1.set_title(r"$t = {:0.5f}$".format(time[i]))
    
    ## Plot colourbar
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$\omega(x, y)$")

    #-------------------------
    # Plot Energy Spectrum   
    #-------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # engy_spec, engy_spec_sum = energy_spectrum(w_hat, kx, ky, Nx, Ny)
    # krange = np.arange(0, engy_spec.shape[0])
    # ax2.plot(engy_spec[1:kx_max] / engy_spec_sum)  # kx[1:kx_max],
    # kpower = (1 / (kx[11:61] ** 4))
    # ax2.plot(kx[11:61], kpower , linestyle = ':', linewidth = 0.5, color = 'black')
    ax2.plot(enrg_spec)
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
    # enstr_spec, enstr_spec_sum = enstrophy_spectrum(w_hat, Nx, Ny)
    # krange = np.arange(0, enstr_spec.shape[0])
    # ax3.plot(enstr_spec[1:kx_max] / enstr_spec_sum)  # kx[1:kx_max], 
    # kpower = (1 / (np.power(kx[11:61], 5 / 3)))
    # ax3.plot(kx[11:61], kpower, linestyle = ':', linewidth = 0.5, color = 'black')
    ax3.plot(enst_spec)
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
    ax4.plot(time[:i], tot_en)
    ax4.plot(time[:i], tot_ens)
    ax4.plot(time[:i], tot_pal)
    ax4.set_xlabel(r"$t$")
    ax4.set_xlim(time[0], time[-1])
    ax4.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax4.set_yscale("log")
    ax4.legend([r"Total Energy", r"Total Enstrophy", r"Total Palinstrophy"])

    ## Save figure
    plt.savefig(output_dir + "SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()

    
@njit
def energy_spectrum(w_h, kx, ky, Nx, Ny):

    ## Spectrum size
    spec_size = Nx / 2

    ## Velocity arrays
    energy_spec = np.zeros(spec_size)

    ## Find u_hat
    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            if kx[i] == 0.0 & ky[i] == 0.0:
                u_hat = np.complex(0.0 + 0.0)
                v_hat = np.complex(0.0 + 0.0)
            else:
                ## Compute prefactor
                k_sqr = np.complex(0.0, 1.0) / (kx[i] ** 2 + ky[j] ** 2)

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

    #-------------------------------
    ## --------- Directories
    #-------------------------------
    ## Read in the data directory provided at CML
    if len(sys.argv) != 2:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> You must provide directory to data.")
        sys.exit()
    else:
        input_dir  = sys.argv[1]
    print("Input Folder: " + tc.C + "{}".format(input_dir) + tc.Rst)

    ## Check if output folder exists -> if not make it
    output_dir = input_dir + "SNAPS/"
    if os.path.isdir(output_dir) != True:
        print("Making folder:" + tc.C + " SNAPS/" + tc.Rst)
        os.mkdir(output_dir)
    print("Output Folder: "+ tc.C + "{}".format(output_dir) + tc.Rst)


    #------------------------------------
    # --------  Read In data
    #------------------------------------
    ## Read in simulation parameters
    sys_params = SimData(input_dir)

    ## Read in solver data
    run_data = ImportData(input_dir, sys_params)

    ## Read in spectra data
    spectra_data = ImportSpectraData(input_dir, sys_params)
    



    # -----------------------
    ## ------ Plot Snaps
    # -----------------------
    ## Start timer
    start = TIME.perf_counter()
    print("\n" + tc.Y + "Printing Snaps..." + tc.Rst)
    
    # i = 0
    # plot_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)
    # i = 1000
    # plot_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)
    # i = num_saves - 1
    # plot_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)



    for i in range(sys_params.ndata):
        plot_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)

    # ## No. of processes
    # proc_lim = 1

    # ## Create tasks for the process pool
    # groups_args = [(mprocs.Process(target = plot_snaps, args = (i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)) for i in range(run_data.w.shape[0]))] * proc_lim

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

    ## End timer
    end = TIME.perf_counter()
    plot_time = end - start
    print("\n\nPlotting Time: {:5.8f}s\n\n".format(plot_time))
        


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