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


from functions import tc, sim_data, import_data, import_spectra_data


# ------------------------------------
## --------  Colormap
# ------------------------------------
## Colourmap
colours = [[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0.25, 0], [1, 1, 1]]   #located @ 0, pi/2, pi, 3pi/2 and 2pi
# my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', colours, N = kmax)                            # set N to inertial range
my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', colours)                            # set N to inertial range

# ## Phases Colourmap
# myhsv   = cm.hsv(np.arange(255))
# norm    = mpl.colors.Normalize(vmin = 0.0, vmax = 2.0*np.pi)
# my_mhsv = mpl.colors.LinearSegmentedColormap.from_list('my_map', myhsv) # set N to inertial range
# phase_colors = cm.ScalarMappable(norm = norm, cmap = my_mhsv)               # map the values to rgba tuple



###############################
##       FUNCTION DEFS       ##
###############################
def plot_summary_snaps(i, w, w_hat, x, y, w_min, w_max, kx, ky, kx_max, tot_en, tot_ens, tot_pal, enrg_spec, enst_spec, time, Nx, Ny):
    
    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(3, 2, hspace = 0.4, wspace = 0.3)

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
    kindx = int(Nx / 3 + 1)
    ax2 = fig.add_subplot(gs[0, 1])
    # kpower = (1 / (kx[11:61] ** 4))
    # ax2.plot(kx[11:61], kpower , linestyle = ':', linewidth = 0.5, color = 'black')
    ax2.plot(enrg_spec[:kindx] / np.sum(enrg_spec[:kindx]))
    ax2.set_xlabel(r"$|k|$")
    ax2.set_ylabel(r"$\mathcal{K}(|k|) / \sum \mathcal{K}(|k|)$")
    ax2.set_title(r"Energy Spectrum")
    # ax2.set_ylim(1e-8, 1)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    # ax2.text(x = kx[61],  y = kpower[-1], s = r"$k^{-4}$")

    #--------------------------
    # Plot Enstrophy Spectrum   
    #--------------------------
    ax3 = fig.add_subplot(gs[1, 1]) 
    # kpower = (1 / (np.power(kx[11:61], 5 / 3)))
    # ax3.plot(kx[11:61], kpower, linestyle = ':', linewidth = 0.5, color = 'black')
    ax3.plot(enst_spec[:kindx] / np.sum(enst_spec[:kindx]))
    ax3.set_xlabel(r"$|k|$")
    ax3.set_ylabel(r"$\mathcal{E}(|k|) / \sum \mathcal{E}(|k|)$")
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


def plot_phase_snaps(i, w, w_h, x, y, time, Nx, Ny, k2Inv):

    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2, hspace = 0.35, wspace = -0.45)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0, 0])
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

    #--------------------------
    # Plot Phases   
    #--------------------------
    ## Adjust data
    data = np.mod(np.angle(np.fft.ifftshift(FullField(w_h))), 2. * np.pi)
    ## Generate colour map
    my_hsv = cm.hsv
    my_hsv.set_bad(color = "white")

    ax2  = fig.add_subplot(gs[0, 1])
    im2  = ax2.imshow(data, extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi)
    ax2.set_xlabel(r"$k_y$")
    ax2.set_ylabel(r"$k_x$")
    ax2.set_title(r"Phases")
    ax2.set_xlim(-Ny / 3, Ny / 3)
    ax2.set_ylim(-Nx / 3, Nx / 3)

    ## Plot colourbar
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_label(r"$\phi_{\mathbf{k}}$")
    cb2.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
    cb2.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

    #--------------------------
    # Plot 2D Enstrophy Spectra   
    #--------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(np.absolute(np.fft.ifftshift(FullField(w_h))) ** 2, extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm())
    ax3.set_xlabel(r"$k_y$")
    ax3.set_ylabel(r"$k_x$")
    ax3.set_title("Enstrophy Spectrum")
    ax3.set_xlim(-Ny / 3, Ny / 3)
    ax3.set_ylim(-Nx / 3, Nx / 3)

    ## Plot colourbar
    div3  = make_axes_locatable(ax3)
    cbax3 = div3.append_axes("right", size = "10%", pad = 0.05)
    cb3   = plt.colorbar(im3, cax = cbax3)
    cb3.set_label(r"$\mathcal{E}(\hat{\omega}_\mathbf{k})$")

    ##-------------------------
    ## Plot 2D Energy Spectra  
    ##-------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.absolute(np.fft.ifftshift(FullField(w_h * k2Inv.astype('complex128')))) ** 2, extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm())
    ax4.set_xlabel(r"$k_y$")
    ax4.set_ylabel(r"$k_x$")
    ax4.set_title("Energy Spectrum")
    ax4.set_xlim(-Ny / 3, Ny / 3)
    ax4.set_ylim(-Nx / 3, Nx / 3)

    ## Plot colourbar
    div4  = make_axes_locatable(ax4)
    cbax4 = div4.append_axes("right", size = "10%", pad = 0.05)
    cb4   = plt.colorbar(im4, cax = cbax4)
    cb4.set_label(r"$\mathcal{K}(\hat{\omega}_\mathbf{k})$")


    ## Save figure
    plt.savefig(phases_output_dir + "Phase_SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()



@njit
def FullField(w_h):

    return np.concatenate((w_h, np.conjugate(w_h[:, -2:0:-1])), axis = 1)


@njit
def ZeroCentredField(w_h):

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


    # -------------------------------
    ## --------- Directories
    # -------------------------------
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


    # ------------------------------------
    ## --------  Read In data
    # ------------------------------------
    ## Read in simulation parameters
    sys_params = sim_data(input_dir, method)

    ## Read in solver data
    run_data = import_data(input_dir, sys_params, method)

    ## Read in spectra data
    if method == "file":
        spectra_data = import_spectra_data(spectra_dir, sys_params, method)
    else:
        spectra_data = import_spectra_data(input_dir, sys_params, method)

    # -----------------------
    ## ------ Plot Snaps
    # -----------------------
    ## Start timer
    start = TIME.perf_counter()
    print("\n" + tc.Y + "Printing Snaps..." + tc.Rst)
    
    i = 1
    plot_summary_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)
    i = sys_params.ndata - 1
    plot_summary_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)
    # i = num_saves - 1
    # plot_summary_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)

    # for i in range(sys_params.ndata):
    #     plot_summary_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)




    # for i in range(sys_params.ndata):
    #     plot_phase_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Nx, run_data.k2Inv)

    # i = 6
    # plot_phase_snaps(i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, run_data.time, sys_params.Nx, sys_params.Nx, run_data.k2Inv)



    # ## No. of processes
    # proc_lim = 1

    # ## Create tasks for the process pool
    # groups_args = [(mprocs.Process(target = plot_summary_snaps, args = (i, run_data.w[i, :, :], run_data.w_hat[i, :, :], run_data.x, run_data.y, np.amin(run_data.w), np.amax(run_data.w), run_data.kx, run_data.ky, int(sys_params.Nx / 3), run_data.tot_enrg[:i], run_data.tot_enst[:i], run_data.tot_palin[:i], spectra_data.enrg_spectrum[i, :], spectra_data.enst_spectrum[i, :], run_data.time, sys_params.Nx, sys_params.Ny)) for i in range(run_data.w.shape[0]))] * proc_lim

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