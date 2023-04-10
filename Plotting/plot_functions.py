#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Plotting functions for solver data


#######################
##  Library Imports  ##
#######################
import numpy as np
import sys
import os
import h5py as h5
import matplotlib as mpl
if mpl.__version__ > '2':    
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from numba import njit

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


##########################################
##       SOLVER SUMMARY FUNCTIONS       ##
##########################################
def plot_vort(outdir, w, x, y, time, snaps, file_type=".png", fig_size=(16, 8)):

       fig = plt.figure(figsize = fig_size)
       gs  = GridSpec(2, 2, hspace = 0.3, wspace=0.15) 

       ax1 = []
       for i in range(2):
              for j in range(2):
                     ax1.append(fig.add_subplot(gs[i, j]))
       indx_list = snaps
       for j, i in enumerate(indx_list):
              im1 = ax1[j].imshow(w[i, :], extent = (y[0], y[-1], x[-1], x[0]), cmap = "jet") #, vmin = w_min, vmax = w_max 
              ax1[j].set_xlabel(r"$y$")
              ax1[j].set_ylabel(r"$x$")
              ax1[j].set_xlim(0.0, y[-1])
              ax1[j].set_ylim(0.0, x[-1])
              ax1[j].set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
              ax1[j].set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
              ax1[j].set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
              ax1[j].set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
              ax1[j].set_title(r"$t = {:0.5f}$".format(time[i]))
              ## Plot colourbar
              div1  = make_axes_locatable(ax1[j])
              cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
              cb1   = plt.colorbar(im1, cax = cbax1)
              cb1.set_label(r"$\omega(x, y)$")

       plt.savefig(outdir + "Vorticity" + file_type, bbox_inches='tight')
       plt.close()


def plot_summary_snaps(out_dir, i, w, x, y, w_min, w_max, kx, ky, kx_max, tot_en, tot_ens, tot_pal, enrg_spec, enst_spec, enrg_diss, enst_diss, enrg_flux_sb, enrg_diss_sb, enst_flux_sb, enst_diss_sb, time, Nx, Ny):

    """
    Plots summary snaps for each iteration of the simulation. Plot: vorticity, energy and enstrophy spectra, dissipation, flux and totals.
    """
    
    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 9))
    gs  = GridSpec(3, 2, hspace = 0.6, wspace = 0.3)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0:2, 0:1])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu") # , vmin = w_min, vmax = w_max 
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
    ax2.plot(enrg_spec[:kindx]) #  / np.sum(enrg_spec[:kindx])
    ax2.set_xlabel(r"$|\mathbf{k}|$")
    ax2.set_ylabel(r"$\mathcal{K}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
    ax2.set_title(r"Energy Spectrum")
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    #--------------------------
    # Plot Enstrophy Spectrum   
    #--------------------------
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax3.plot(enst_spec[:kindx]) # / np.sum(enst_spec[:kindx])
    ax3.set_xlabel(r"$|\mathbf{k}|$")
    ax3.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)") #/ \sum \mathcal{E}(|k|)$
    ax3.set_title(r"Enstrophy Spectrum")
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)


    #--------------------------
    # Plot System Measures   
    #-------------------------- 
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time[:i], tot_en / tot_en[0])
    ax4.plot(time[:i], tot_ens / tot_ens[0])
    ax4.plot(time[:i], tot_pal / tot_pal[0])
    ax4.set_xlabel(r"$t$")
    ax4.set_xlim(time[0], time[-1])
    ax4.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax4.set_yscale("log")
    ax4.legend([r"Total Energy", r"Total Enstrophy", r"Total Palinstrophy"])

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time[:i], enrg_diss)
    ax5.plot(time[:i], enst_diss)
    ax5.set_xlabel(r"$t$")
    ax5.set_xlim(time[0], time[-1])
    ax5.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax5.set_yscale("log")
    ax5.legend([r"Energy Diss", r"Enstrophy Diss"])

    # #--------------------------
    # # Plot Energy Flux 
    # #-------------------------- 
    # ax6 = fig.add_subplot(gs[3, 0])
    # ax6.plot(time[:i], enrg_flux_sb)
    # ax6.plot(time[:i], enrg_diss_sb)
    # ax6.set_xlabel(r"$t$")
    # ax6.set_xlim(time[0], time[-1])
    # ax6.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax6.legend([r"Energy Flux", r"Energy Diss"])

    # #--------------------------
    # # Plot Energy Flux 
    # #-------------------------- 
    # ax7 = fig.add_subplot(gs[3, 1])
    # ax7.plot(time[:i], enst_flux_sb)
    # ax7.plot(time[:i], enst_diss_sb)
    # ax7.set_xlabel(r"$t$")
    # ax7.set_xlim(time[0], time[-1])
    # ax7.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    # ax7.legend([r"Enstrophy Flux", r"Enstrophy Diss"])

    ## Save figure
    plt.savefig(out_dir + "SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()

def plot_flow_summary_stream(input_dir, output_dir, i, Nx, Ny):

       ## Open Main data file to get vorticity
       with h5.File(input_dir + "Main_HDF_Data.h5", "r") as main_file:
              ## Group name
              group_name = "Iter_{:05d}".format(i)
              if "w" in list(main_file[group_name].keys()):
                     w = main_file[group_name]["w"][:, :]
              else:
                     w_hat = main_file[group_name]["w_hat"][:, :]
                     w = np.fft.irfft2(w_hat) * Nx * Nx

              tot_en      = main_file["TotalEnergy"][:]
              tot_ens     = main_file["TotalEnstrophy"][:]
              tot_en_diss = main_file["EnergyDissipation"][:]
              x = main_file["x"][:]
              y = main_file["y"][:]
              time = main_file["Time"][:]
              kx = main_file["kx"][:]
              ky = main_file["ky"][:]

       ## Get spectra data
       with h5.File(input_dir + "Spectra_HDF_Data.h5") as spectra_file:
              group_name = "Iter_{:05d}".format(i)
              enrg_spec = spectra_file[group_name]["EnergySpectrum"][:]
              enst_spec = spectra_file[group_name]["EnstrophySpectrum"][:]

       ## Call plot flow summary function
       plot_flow_summary(output_dir, i, w, np.amin(w), np.amax(w), None, None, np.amin(enrg_spec), np.amax(enrg_spec), np.amin(enst_spec), np.amax(enst_spec), None, x, y, time, Nx, Ny, kx, ky, enrg_spec, enst_spec, tot_en, tot_ens, tot_en_diss)


def plot_phase_snaps_stream(input_dir, output_dir, post_file, i, Nx, Ny, spec_lims):

       ## Open Main data file to get vorticity
       with h5.File(input_dir + "Main_HDF_Data.h5", "r") as main_file:
              x = main_file["x"][:]
              y = main_file["y"][:]
              time = main_file["Time"][:]
              kx = main_file["kx"][:]
              ky = main_file["ky"][:]
              ## Group name
              group_name = "Iter_{:05d}".format(i)
              if "w" in list(main_file[group_name].keys()):
                     w = main_file[group_name]["w"][:, :]
              else:
                     w_hat = main_file[group_name]["w_hat"][:, :]
                     w = np.fft.irfft2(w_hat) * Nx * Nx
              
       ## Get spectra data
       with h5.File(input_dir + post_file) as post_file:
              group_name = "Snap_{:05d}".format(i)
              enrg_spec = post_file[group_name]["FullFieldEnergySpectrum"][:]
              enst_spec = post_file[group_name]["FullFieldEnstrophySpectrum"][:]
              # min_enrg  = np.amin(np.delete(enrg_spec.flatten(), np.where(enrg_spec.flatten() == -50.0)))
              # max_enrg  = np.amax(np.delete(enrg_spec.flatten(), np.where(enrg_spec.flatten() == -50.0)))
              # min_enst  = np.amin(np.delete(enst_spec.flatten(), np.where(enst_spec.flatten() == -50.0)))
              # max_enst  = np.amax(np.delete(enst_spec.flatten(), np.where(enst_spec.flatten() == -50.0)))
              # spec_lims = np.array([min_enrg, max_enrg, min_enst, max_enst])
              # spec_lims[spec_lims[:] == 0.0] = 1e-12
              phases    = post_file[group_name]["FullFieldPhases"][:]

       ## Call plot flow summary function
       plot_phase_snaps(output_dir, i, w, phases, enrg_spec, enst_spec, spec_lims, None, None, x, y, time, Nx, Ny, kx, ky)


def plot_phase_snaps(out_dir, i, w, phases, enrg_spec, enst_spec, spec_lims, w_min, w_max, x, y, time, Nx, Ny, kx, ky):

    """
    Plots summary snap of the solver data for the current iteration. Plots vorticity, full zero centre phases and spectra.

    i     : int
           - The current snap
    w     : ndarray, float64
           - The Real space vorticity Array
    w_h   : ndarray, complex128
           - The Fourier space vorticity array
    x     : array, float64
           - Array of the collocation points in the first direction
    y     : array, float64
           - Array of the collocation points in the second direction
    time  : array, float64
           - Array of the snapshot times
    Nx    : int
           - Size of the first dimension
    Ny    : int
           - Size of the second dimension
    k2Inv : ndarray, float64
           - Array containing the inverse of |k|^2
    kx    : array, int
           - The wavenumbers in the first direction
    ky    : array, int
           - The wavenumbers in the first direction
    """

    ## Print Update
    print("SNAP: {}".format(i))

    ## Generate colour maps
    my_hsv = cm.jet
    my_hsv.set_under(color = "white")
    my_magma = mpl.colors.ListedColormap(cm.magma.colors[::-1])
    my_magma.set_under(color = "white")

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2, hspace = 0.35, wspace = -0.45)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu", vmin=-6, vmax=6) #, vmin = w_min, vmax = w_max 
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
    ax2  = fig.add_subplot(gs[0, 1])
    im2  = ax2.imshow(phases, extent = (-Ny / 3 + 1, Ny / 3, -Nx / 3 + 1, Nx / 3), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi)
    ax2.set_xlabel(r"$k_y$")
    ax2.set_ylabel(r"$k_x$")
    ax2.set_title(r"Phases")
    # ax2.set_xlim(-Ny / 3, Ny / 3)
    # ax2.set_ylim(-Nx / 3, Nx / 3)

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
    ax3  = fig.add_subplot(gs[1, 0])
    im3  = ax3.imshow(enst_spec, extent = (-Ny / 3 + 1, Ny / 3, -Nx / 3 + 1, Nx / 3), cmap = my_magma, norm = mpl.colors.LogNorm(vmin = spec_lims[3], vmax = spec_lims[2])) # extent = (-Ny / 3 + 1, Ny / 3, -Nx / 3 + 1, Nx / 3), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), 
    ax3.set_xlabel(r"$k_y$")
    ax3.set_ylabel(r"$k_x$")
    ax3.set_title("Enstrophy Spectrum")
    ## Plot colourbar
    div3  = make_axes_locatable(ax3)
    cbax3 = div3.append_axes("right", size = "10%", pad = 0.05)
    cb3   = plt.colorbar(im3, cax = cbax3)
    cb3.set_label(r"$\mathcal{E}(\hat{\omega}_\mathbf{k})$")

    ##-------------------------
    ## Plot 2D Energy Spectra  
    ##-------------------------
    ax4  = fig.add_subplot(gs[1, 1])
    im4  = ax4.imshow(enrg_spec, extent = (-Ny / 3 + 1, Ny / 3, -Nx / 3 + 1, Nx / 3), cmap = my_magma, norm = mpl.colors.LogNorm(vmin = spec_lims[1], vmax = spec_lims[0])) # , cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), 
    ax4.set_xlabel(r"$k_y$")
    ax4.set_ylabel(r"$k_x$")
    ax4.set_title("Energy Spectrum")
    # ax4.set_xlim(-Ny / 3, Ny / 3)
    # ax4.set_ylim(-Nx / 3, Nx / 3)

    ## Plot colourbar
    div4  = make_axes_locatable(ax4)
    cbax4 = div4.append_axes("right", size = "10%", pad = 0.05)
    cb4   = plt.colorbar(im4, cax = cbax4)
    cb4.set_label(r"$\mathcal{K}(\hat{\omega}_\mathbf{k})$")


    ## Save figure
    plt.savefig(out_dir + "Phase_SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()

def plot_flow_summary(out_dir, i, w, w_min, w_max, measure_min, measure_max, x, y, time, Nx, Ny, kx, ky, tot_en, tot_ens, tot_pal):

    """
    Plots summary snap of the solver data for the current iteration. Plots vorticity, full zero centre phases and spectra.

    i     : int
           - The current snap
    w     : ndarray, float64
           - The Real space vorticity Array
    x     : array, float64
           - Array of the collocation points in the first direction
    y     : array, float64
           - Array of the collocation points in the second direction
    time  : array, float64
           - Array of the snapshot times
    Nx    : int
           - Size of the first dimension
    Ny    : int
           - Size of the second dimension
    kx    : array, int
           - The wavenumbers in the first direction
    ky    : array, int
           - The wavenumbers in the first direction
    """

    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2) 

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "jet") # vmin = w_min, vmax = w_max 
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
    # Plot System Measures   
    #-------------------------- 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time[:i], tot_en[:i] / tot_en[0])
    ax2.plot(time[:i], tot_ens[:i] / tot_ens[0])
    ax2.plot(time[:i], tot_pal[:i] / tot_pal[0])
    ax2.set_xlabel(r"$t$")
    ax2.set_xlim(time[0], time[-1])
    ax2.set_ylim(measure_min, measure_max)
    # ax2.set_yscale("log")
    ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax2.legend([r"$\mathcal{K}(t) / \mathcal{K}(0)$", r"$\mathcal{E}(t) / \mathcal{E}(0)$", r"$\mathcal{P}(t) / \mathcal{P}(0)$"])

    ## Save figure
    plt.savefig(out_dir + "Decay_SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()


def plot_decay_snaps_2(out_dir, i, w, w_min, w_max, measure_min, measure_max, x, y, time, Nx, Ny, kx, ky, enrg_spec, enst_spec, tot_en, tot_ens, tot_pal, u0):

       """
       Plots summary snap of the solver data for the current iteration. Plots vorticity, full zero centre phases and spectra.

       i     : int
              - The current snap
       w     : ndarray, float64
              - The Real space vorticity Array
       x     : array, float64
              - Array of the collocation points in the first direction
       y     : array, float64
              - Array of the collocation points in the second direction
       time  : array, float64
              - Array of the snapshot times
       Nx    : int
              - Size of the first dimension
       Ny    : int
              - Size of the second dimension
       kx    : array, int
              - The wavenumbers in the first direction
       ky    : array, int
              - The wavenumbers in the first direction
       """

       ## Print Update
       print("SNAP: {}".format(i))

       ## Create Figure
       fig = plt.figure(figsize = (16, 8))
       gs  = GridSpec(2, 2, hspace = 0.3) 

       ##-------------------------
       ## Plot vorticity   
       ##-------------------------
       ax1 = fig.add_subplot(gs[0, 0])
       im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "jet") #, vmin = w_min, vmax = w_max 
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
       # Plot System Measures   
       #-------------------------- 
       ax2 = fig.add_subplot(gs[0, 1])
       ax2.plot(time[:i], tot_en[:i] / tot_en[0])
       ax2.plot(time[:i], tot_ens[:i] / tot_ens[0])
       ax2.plot(time[:i], tot_pal[:i] / tot_pal[0])
       ax2.set_xlabel(r"$t$")
       ax2.set_xlim(time[0], time[-1])
       ax2.set_ylim(measure_min, measure_max)
       # ax2.set_yscale("log")
       ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
       ax2.legend([r"$\mathcal{K}(t) / \mathcal{K}(0)$", r"$\mathcal{E}(t) / \mathcal{E}(0)$", r"$\mathcal{P}(t) / \mathcal{P}(0)$"])

       #-------------------------
       # Plot Energy Spectrum   
       #-------------------------
       ax3 = fig.add_subplot(gs[1, 0])
       kindx = int(Nx / 3 + 1)
       kk = np.arange(1, kindx)
       line_color = []
       if u0 == "DECAY_TURB_II":
              p1, = ax3.plot(kk, (kk**6 /(1 + kk/60.)**18 )/ 10**7.5)
       elif u0 == "DECAY_TURB":
              p1, = ax3.plot(kk, (kk /(1 + (kk**4)/6.))/ 10**0.25) 
       elif u0 == "DECAY_TURB_ALT":
              p1, = ax3.plot(kk, (kk /(1 + (kk/6.)**4)/ 10**0.25))              
       line_color.append(p1.get_color())
       for j in range(enrg_spec.shape[0]):
              spec_enrg = enrg_spec[j, 1:kindx]  #  / np.sum(enrg_spec[j, 1:kindx])
              if j == 0:
                     p2, = ax3.plot(kk, spec_enrg, color = 'y')
                     line_color.append(p2.get_color())
              elif j == enrg_spec.shape[0] - 1:
                     p3, = ax3.plot(kk, spec_enrg, color = 'b')
                     line_color.append(p3.get_color())
              # else:                     
              #        ax3.plot(kk, spec, linestyle = ':', color = 'k', linewidth = 0.5, alpha = 0.02)
       ax3.set_xlabel(r"$|k|$")
       ax3.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$") # / \sum \mathcal{K}(|k|)
       ax3.set_title(r"Energy Spectrum")
       # ax3.set_ylim(1e-10, 10)
       ax3.set_yscale('log')
       ax3.set_xscale('log')
       ax3.legend([r"Paper", r"$E(0)$", r"$E(t)$"])
       leg = ax3.get_legend()
       for j, handle in enumerate(leg.legendHandles):
              handle.set_color(line_color[j])

       #--------------------------
       # Plot Enstrophy Spectrum   
       #--------------------------
       ax4 = fig.add_subplot(gs[1, 1]) 
       line_color = []
       if u0 == "DECAY_TURB_II":
              p1, = ax4.plot(kk, (kk**8 /(1 + kk/60.)**18 )/ 10**7.5)
       elif u0 == "DECAY_TURB":
              p1, = ax4.plot(kk, ((kk**3) /(1 + (kk**4)/6.))/ 10**2.5)
       elif u0 == "DECAY_TURB_ALT":
              p1, = ax4.plot(kk**3, (kk /(1 + (kk/6.)**4)/ 10**0.25))  
       line_color.append(p1.get_color())
       for j in range(enst_spec.shape[0]):
              spec_enst = enst_spec[j, 1:kindx] #  / np.sum(enst_spec[j, 1:kindx])
              if j == 0:
                     p2, = ax4.plot(kk, spec_enst, color = 'y')
                     line_color.append(p2.get_color())
              elif j == enst_spec.shape[0] - 1:
                     p3, = ax4.plot(kk, spec_enst,  color = 'b')
                     line_color.append(p3.get_color())
              # else:                     
              #        ax4.plot(kk, spec_enst, linestyle = ':', color = 'k', linewidth = 0.5, alpha = 0.02)
       ax4.set_xlabel(r"$|k|$")
       ax4.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")  #/ \sum \mathcal{E}(|k|)
       ax4.set_title(r"Enstrophy Spectrum")
       # ax4.set_ylim(1e-10, 10)
       ax4.set_yscale('log')
       ax4.set_xscale('log')
       ax4.legend([r"Paper", r"$E(0)$", r"$E(t)$"])
       leg = ax4.get_legend()
       for j, handle in enumerate(leg.legendHandles):
              handle.set_color(line_color[j])

       ## Save figure
       plt.savefig(out_dir + "Decay2_SNAP_{:05d}.png".format(i), bbox_inches='tight') 
       plt.close()


def plot_sector_phase_sync_snaps(i, out_dir, phases, theta_k3, R, Phi, t, Nx, Ny):

       """
       Plots the phases and average phase and sync per sector of the phases

       i       : int
               - The current snap
       out_dir : str
               - Path to the output folder
       phases  : ndarray, float64
               - 2D array of the Fourier phases, only ky > section 
       theta   : array, float64
               - 1D Array containing the angles of the sector boundaries
       R       : array, float64
               - Array of the phase sync parameter for each sector
       Phi     : array, float64
               - Array of the average phase for each sector
       t       : float64
               - The current time 
       Nx      : int
               - Size of the first dimension
       Ny      : int
               - Size of the second dimension
       """

       ## Print Update
       print("SNAP: {}".format(i))

       ## Set up figure
       fig = plt.figure(figsize = (16, 9))
       gs  = GridSpec(2, 2)
       
       ## Generate colour map
       my_hsv = cm.jet
       my_hsv.set_under(color = "white")

       #--------------------------
       # Plot Phases  
       #--------------------------
       ax1 = fig.add_subplot(gs[0, 0:2])
       # im1 = ax1.imshow(phases, extent = (0, int(Ny / 3), int(-Nx / 3 + 1), int(Nx / 3)), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi)
       # ang = np.arange(-np.pi/2, np.pi/2 + np.pi / 100, np.pi / 100)
       # angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
       # angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       # rmax_ext = int(Nx / 3 + 20)
       # for tick, label in zip(angticks, angtickLabels):
       #        if abs(tick) == np.pi/2:
       #               ax1.text(x = (rmax_ext - 7.5) * np.cos(tick) + 0.1, y = (rmax_ext - 7.5) * np.sin(tick), s = label) ## shift the +-pi/2 to the right 
       #        elif tick <= 0:
       #               ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) - 0.25, s = label) ## shift the bottom quadrant ticks down
       #        else:
       #               ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) - 0.25, s = label)
       #        ax1.plot([(rmax_ext - 15) * np.cos(tick), (rmax_ext - 11) * np.cos(tick)], [(rmax_ext - 15) * np.sin(tick), (rmax_ext - 11) * np.sin(tick)], color = 'k', linestyle = '-', linewidth = 0.5)
       # ax1.plot((rmax_ext - 15) * np.cos(ang), (rmax_ext - 15) * np.sin(ang), color = 'k', linestyle = '--', linewidth = 0.5)
       # ax1.set_xlabel(r"$k_y$")
       # ax1.set_ylabel(r"$k_x$")
       # ax1.set_ylim(-rmax_ext, rmax_ext)
       # ax1.set_xlim(0, rmax_ext)
       im1 = ax1.imshow(np.fliplr(np.transpose(phases)), extent = (int(-Nx / 3 + 1), int(Nx / 3), int(Ny / 3), 0), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi) 
       ang = np.arange(0, np.pi + np.pi / 100, np.pi / 100)
       angticks      = [0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2, 5*np.pi/8, 6*np.pi/8.0, 7*np.pi/8, np.pi]
       angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       angtickLabels.reverse()
       rmax_ext = int(Nx / 3 + 20)
       for tick, label in zip(angticks, angtickLabels):
              if tick == 0:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) + 5.0, s = label) ## shift the +-pi/2 to the right 
              elif tick > np.pi/2:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick) - 5.0, y = (rmax_ext - 7.5) * np.sin(tick) + 4.0, s = label) ## shift the bottom quadrant ticks down
              else:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick), s = label) 
              ax1.plot([(rmax_ext - 15) * np.cos(tick), (rmax_ext - 11) * np.cos(tick)], [(rmax_ext - 15) * np.sin(tick), (rmax_ext - 11) * np.sin(tick)], color = 'k', linestyle = '-', linewidth = 0.5)
       ax1.plot((rmax_ext - 15) * np.cos(ang), (rmax_ext - 15) * np.sin(ang), color = 'k', linestyle = '--', linewidth = 0.5)
       ax1.set_xlabel(r"$k_x$")
       ax1.set_ylabel(r"$k_y$")
       ax1.set_xlim(-rmax_ext, rmax_ext)
       ax1.set_ylim(rmax_ext, 0)
       ax1.set_title(r"Fourier Phases")
       div1  = make_axes_locatable(ax1)
       cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
       cb1   = plt.colorbar(im1, cax = cbax1)
       cb1.set_label(r"$\phi_{\mathbf{k}}$")
       cb1.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
       cb1.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

       #--------------------------
       # Plot Phase Sync Per Sector  
       #--------------------------
       angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
       angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       ax2 = fig.add_subplot(gs[1, 0])
       ax2.plot(theta, R)
       ax2.set_xlim(-np.pi/2, np.pi/2)
       ax2.set_xticks(angticks)
       ax2.set_xticklabels(angtickLabels)
       ax2.set_ylim(0, 1.)
       ax2.set_xlabel(r"$\theta$")
       ax2.set_ylabel(r"$\mathcal{R}$")
       ax2.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax2.set_title(r"Phase Synchronization Per Sector")

       #--------------------------
       # Plot Avg Phase Per Sector  
       #--------------------------
       ax3 = fig.add_subplot(gs[1, 1])
       ax3.plot(theta, Phi, '.-')
       ax3.set_xlim(-np.pi/2, np.pi/2)
       ax3.set_xticks(angticks)
       ax3.set_xticklabels(angtickLabels)
       ax3.set_xlabel(r"$\theta$")
       ax3.set_ylabel(r"$\Phi$")
       ax3.set_ylim(-np.pi, np.pi)
       ax3.set_yticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       ax3.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       # ax3.set_yticks([0.0, np.pi/2.0, np.pi, 3.*np.pi/2., 2.*np.pi])
       # ax3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
       ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax3.set_title(r"Averager Phase Per Sector")

       ## Add title and save fig
       plt.suptitle(r"$t = {:.5f}$".format(t))
       plt.savefig(out_dir + "/Phase_Sync_SNAP_{:05d}.png".format(i), bbox_inches = 'tight')
       plt.close()


def plot_sector_phase_sync_snaps_full(i, out_dir, w, enst_spec, enst_flux, phases, theta, R, Phi, flux_min, flux_max, t, x, y, Nx, Ny):

       """
       Plots the phases and average phase and sync per sector of the phases

       i       : int
               - The current snap
       out_dir : str
               - Path to the output folder
       phases  : ndarray, float64
               - 2D array of the Fourier phases, only ky > section 
       theta   : array, float64
               - 1D Array containing the angles of the sector boundaries
       R       : array, float64
               - Array of the phase sync parameter for each sector
       Phi     : array, float64
               - Array of the average phase for each sector
       t       : float64
               - The current time 
       Nx      : int
               - Size of the first dimension
       Ny      : int
               - Size of the second dimension
       """

       ## Print Update
       print("SNAP: {}".format(i))

       ## Set up figure
       fig = plt.figure(figsize = (16, 9))
       gs  = GridSpec(2, 3)
       
       ## Generate colour map
       my_hsv = cm.jet
       my_hsv.set_under(color = "white")
       my_hot = cm.hot
       my_hot.set_under(color = "white")

       #--------------------------
       # Plot Phases  
       #--------------------------
       ax1 = fig.add_subplot(gs[0, 0:2])
       im1 = ax1.imshow(np.fliplr(np.transpose(phases)), extent = (int(-Nx / 3 + 1), int(Nx / 3), int(Ny / 3), 0), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi) 
       ang = np.arange(0, np.pi + np.pi / 100, np.pi / 100)
       angticks      = [0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2, 5*np.pi/8, 6*np.pi/8.0, 7*np.pi/8, np.pi]
       angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       angtickLabels.reverse()
       rmax_ext = int(Nx / 3 + 20)
       for tick, label in zip(angticks, angtickLabels):
              if tick == 0:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) + 5.0, s = label) ## shift the +-pi/2 to the right 
              elif tick > np.pi/2:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick) - 5.0, y = (rmax_ext - 7.5) * np.sin(tick) + 4.0, s = label) ## shift the bottom quadrant ticks down
              else:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick), s = label) 
              ax1.plot([(rmax_ext - 15) * np.cos(tick), (rmax_ext - 11) * np.cos(tick)], [(rmax_ext - 15) * np.sin(tick), (rmax_ext - 11) * np.sin(tick)], color = 'k', linestyle = '-', linewidth = 0.5)
       ax1.plot((rmax_ext - 15) * np.cos(ang), (rmax_ext - 15) * np.sin(ang), color = 'k', linestyle = '--', linewidth = 0.5)
       ax1.set_xlabel(r"$k_x$")
       ax1.set_ylabel(r"$k_y$")
       ax1.set_xlim(-rmax_ext, rmax_ext)
       ax1.set_ylim(rmax_ext, 0)
       ax1.set_title(r"Fourier Phases")
       div1  = make_axes_locatable(ax1)
       cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
       cb1   = plt.colorbar(im1, cax = cbax1)
       cb1.set_label(r"$\phi_{\mathbf{k}}$")
       cb1.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
       cb1.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

       #--------------------------
       # Plot Vorticity  
       #--------------------------
       ax2 = fig.add_subplot(gs[0, 2])
       im2 = ax2.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "jet") #, vmin = w_min, vmax = w_max 
       ax2.set_xlabel(r"$y$")
       ax2.set_ylabel(r"$x$")
       ax2.set_xlim(0.0, y[-1])
       ax2.set_ylim(0.0, x[-1])
       ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
       ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
       ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
       ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
       div2  = make_axes_locatable(ax2)
       cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
       cb2   = plt.colorbar(im2, cax = cbax2)
       cb2.set_label(r"$\omega(x, y)$")
       ax2.set_title(r"Vorticity")

       #--------------------------
       # Plot Order Parameters
       #--------------------------
       angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
       angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       ax3 = fig.add_subplot(gs[1, 0])
       div3   = make_axes_locatable(ax3)
       axtop3 = div3.append_axes("top", size = "100%", pad = 0.2)
       axtop3.plot(theta, Phi, '.-', color = "orange")
       axtop3.set_xlim(-np.pi/2, np.pi/2)
       axtop3.set_xticks(angticks)
       axtop3.set_xticklabels([])
       axtop3.set_ylim(-np.pi-0.2, np.pi+0.2)
       axtop3.set_yticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       axtop3.set_yticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       axtop3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       axtop3.set_title(r"Order Parameters")
       axtop3.set_ylabel(r"$\Phi$")
       ax3.plot(theta, R)
       ax3.set_xticks(angticks)
       ax3.set_xlim(-np.pi/2, np.pi/2)
       ax3.set_xticklabels(angtickLabels)
       ax3.set_ylim(0 - 0.05, 1 + 0.05)
       ax3.set_xlabel(r"$\theta$")
       ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax3.set_ylabel(r"$\mathcal{R}$")

       #--------------------------------
       # Plot Enstrophy Flux Per Sector  
       #--------------------------------
       ax4 = fig.add_subplot(gs[1, 1])
       ax4.plot(theta, enst_flux, '.-')
       ax4.set_xlim(-np.pi/2, np.pi/2)
       ax4.set_xticks(angticks)
       ax4.set_xticklabels(angtickLabels)
       ax4.set_xlabel(r"$\theta$")
       ax4.set_ylabel(r"$\Pi_{\mathcal{C}}$")
       ax4.set_yscale('symlog')
       ax4.set_ylim(flux_min, flux_max)
       ax4.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax4.set_title(r"Enstrophy Flux Per Sector")

       #--------------------------------
       # Plot Enstrophy Flux Per Sector  
       #--------------------------------
       ax5  = fig.add_subplot(gs[1, 2])
       im5  = ax5.imshow(enst_spec, extent = (-Ny / 3 + 1, Ny / 3, -Nx / 3 + 1, Nx / 3), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm()) # cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm() 
       ax5.set_xlabel(r"$k_y$")
       ax5.set_ylabel(r"$k_x$")
       ax5.set_title("Enstrophy Spectrum")
       div5  = make_axes_locatable(ax5)
       cbax5 = div5.append_axes("right", size = "10%", pad = 0.05)
       cb5   = plt.colorbar(im5, cax = cbax5)
       cb5.set_label(r"$\mathcal{E}(\hat{\omega}_\mathbf{k})$")

       ## Add title and save fig
       plt.suptitle(r"$t = {:.5f}$".format(t))
       plt.savefig(out_dir + "/Phase_Sync_SNAP_{:05d}.png".format(i), bbox_inches = 'tight')
       plt.close()


def plot_sector_phase_sync_snaps_full_sec(i, out_dir, w, enst_spec, enst_flux, enst_flux_a_sec, phases, theta_k3, R, R_a_sec, Phi, Phi_a_sec, flux_lims, t, x, y, Nx, Ny):

       """
       Plots the phases and average phase and sync per sector of the phases

       i       : int
               - The current snap
       out_dir : str
               - Path to the output folder
       phases  : ndarray, float64
               - 2D array of the Fourier phases, only ky > section 
       theta_k3   : array, float64
               - 1D Array containing the angles of the sector boundaries
       R       : array, float64
               - Array of the phase sync parameter for each sector
       Phi     : array, float64
               - Array of the average phase for each sector
       t       : float64
               - The current time 
       Nx      : int
               - Size of the first dimension
       Ny      : int
               - Size of the second dimension
       """

       ## Print Update
       print("SNAP: {}".format(i))

       ## Set up figure
       fig = plt.figure(figsize = (20, 13))
       gs  = GridSpec(3, 3, hspace = 0.3)
       
       ## Generate colour map
       my_hsv = cm.jet
       my_hsv.set_under(color = "white")
       my_hot = cm.hot
       my_hot.set_under(color = "white")
       # my_magma = cm.magma
       # my_magma.set_under(color = "white")
       my_magma = mpl.colors.ListedColormap(cm.magma.colors[::-1])
       my_magma.set_under(color = "white")

       ## Create appropriate ticks and ticklabels for the sector angles
       dtheta_k3 = theta_k3[1] - theta_k3[0]
       angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
       angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       # angtickLabels_alpha = [r"$\theta - \frac{\pi}{2}$", r"$\theta - \frac{\pi}{3}$", r"$\theta - \frac{\pi}{4}$", r"$\theta - \frac{\pi}{6}$", r"$\theta + \frac{\pi}{6}$", r"$\theta + \frac{\pi}{4}$", r"$\theta + \frac{\pi}{3}$", r"$\theta + \frac{\pi}{2}$"]
       angtickLabels_alpha = np.linspace(0, len(theta_k3) + 1, num = len(angtickLabels), endpoint = False, dtype = "int64").tolist()
       theta_k3_min     = -np.pi/2 - dtheta_k3 / 2
       theta_k3_max     = np.pi/2 + dtheta_k3 /2
       alpha_angticks      = angticks
       alpha_angticklabels = angtickLabels 
       alpha_min = theta_k3_min
       alpha_max = theta_k3_max

       #--------------------------
       # Plot Phases  
       #--------------------------
       ax1 = fig.add_subplot(gs[0, 0:2])
       im1 = ax1.imshow(np.fliplr(np.transpose(phases)), extent = (int(-Nx / 3), int(Nx / 3), int(Ny / 3), 0), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi) 
       ang = np.arange(0, np.pi + np.pi / 100, np.pi / 100)
       kspace_angticks      = [0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2, 5*np.pi/8, 6*np.pi/8.0, 7*np.pi/8, np.pi]
       kspace_angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       kspace_angtickLabels.reverse()
       rmax_ext = int(Nx / 3 + 20)
       for tick, label in zip(kspace_angticks, kspace_angtickLabels):
              if tick == 0:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) + 5.0, s = label) ## shift the +-pi/2 to the right 
              elif tick > np.pi/2:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick) - 5.0, y = (rmax_ext - 7.5) * np.sin(tick) + 4.0, s = label) ## shift the bottom quadrant ticks down
              else:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick), s = label) 
              ax1.plot([(rmax_ext - 15) * np.cos(tick), (rmax_ext - 11) * np.cos(tick)], [(rmax_ext - 15) * np.sin(tick), (rmax_ext - 11) * np.sin(tick)], color = 'k', linestyle = '-', linewidth = 0.5)
       ax1.plot((rmax_ext - 15) * np.cos(ang), (rmax_ext - 15) * np.sin(ang), color = 'k', linestyle = '--', linewidth = 0.5)
       ax1.set_xlabel(r"$k_x$")
       ax1.set_ylabel(r"$k_y$")
       ax1.set_xlim(-rmax_ext, rmax_ext)
       ax1.set_ylim(rmax_ext, 0)
       ax1.set_title(r"Fourier Phases")
       div1  = make_axes_locatable(ax1)
       cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
       cb1   = plt.colorbar(im1, cax = cbax1)
       cb1.set_label(r"$\phi_{\mathbf{k}}$")
       cb1.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
       cb1.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

       #--------------------------
       # Plot Vorticity  
       #--------------------------
       ax2 = fig.add_subplot(gs[0, 2])
       im2 = ax2.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu", vmin = -6, vmax = 6) #, vmin = w_min, vmax = w_max 
       ax2.set_xlabel(r"$y$")
       ax2.set_ylabel(r"$x$")
       ax2.set_xlim(0.0, y[-1])
       ax2.set_ylim(0.0, x[-1])
       ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
       ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
       ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
       ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
       div2  = make_axes_locatable(ax2)
       cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
       cb2   = plt.colorbar(im2, cax = cbax2)
       cb2.set_label(r"$\omega(x, y)$")
       ax2.set_title(r"Vorticity")

       #--------------------------
       # Plot Order Parameters
       #--------------------------
       ax3 = fig.add_subplot(gs[1, 0])
       div3   = make_axes_locatable(ax3)
       axtop3 = div3.append_axes("top", size = "100%", pad = 0.2)
       axtop3.plot(theta_k3, Phi, '.-', color = "orange")
       axtop3.set_xlim(theta_k3_min, theta_k3_max)
       axtop3.set_xticks(angticks)
       axtop3.set_xticklabels([])
       axtop3.set_ylim(-np.pi-0.2, np.pi+0.2)
       axtop3.set_yticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       axtop3.set_yticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       axtop3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       axtop3.set_title(r"Order Parameters (1D)")
       axtop3.set_ylabel(r"$\Phi^{1D}$")
       ax3.plot(theta_k3, R)
       ax3.set_xlim(theta_k3_min, theta_k3_max)
       ax3.set_xticks(angticks)
       ax3.set_xticklabels(angtickLabels)
       ax3.set_ylim(0 - 0.05, 1 + 0.05)
       ax3.set_xlabel(r"$\theta$")
       ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax3.set_ylabel(r"$\mathcal{R}^{1D}$")

       #--------------------------------
       # Plot Enstrophy Flux Per Sector  (1D)
       #--------------------------------
       ax4 = fig.add_subplot(gs[1, 1])
       ax4.plot(theta_k3, enst_flux, '.-')
       ax4.set_xlim(theta_k3_min, theta_k3_max)
       ax4.set_xticks(angticks)
       ax4.set_xticklabels(angtickLabels)
       ax4.set_xlabel(r"$\theta$")
       ax4.set_ylabel(r"$\Pi_{\mathcal{C}}^{1D}$")
       ax4.set_ylim(flux_lims[0], flux_lims[1])
       ax4.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax4.set_title(r"Enstrophy Flux Per Sector (1D)")

       #--------------------------------
       # Plot Enstrophy Spectrum  (2D)
       #--------------------------------
       ax5  = fig.add_subplot(gs[1, 2])
       enst_spec[Ny//3, Nx//3] = -10
       im5  = ax5.imshow(enst_spec, extent = (-Ny / 3, Ny / 3, -Nx / 3, Nx / 3), cmap = my_magma, norm = mpl.colors.LogNorm()) # cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm() 
       ax5.set_xlabel(r"$k_y$")
       ax5.set_ylabel(r"$k_x$")
       ax5.set_title("Enstrophy Spectrum")
       div5  = make_axes_locatable(ax5)
       cbax5 = div5.append_axes("right", size = "10%", pad = 0.05)
       cb5   = plt.colorbar(im5, cax = cbax5)
       cb5.set_label(r"$\mathcal{E}(\hat{\omega}_\mathbf{k})$")

       #--------------------------------
       # Plot Sync Across Sectors (2D)
       #--------------------------------
       ax6 = fig.add_subplot(gs[2, 0])
       im6 = ax6.imshow(np.flipud(R_a_sec), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), vmin = 0.0, vmax = 1.0)
       ax6.set_xticks(angticks)
       ax6.set_xticklabels(angtickLabels_alpha)
       ax6.set_yticks(angticks)
       ax6.set_yticklabels(angtickLabels)
       ax6.set_ylabel(r"$\theta$")
       ax6.set_xlabel(r"$\alpha$")
       ax6.set_title(r"Sync Across Sectors (2D)")
       div6  = make_axes_locatable(ax6)
       cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
       cb6   = plt.colorbar(im6, cax = cbax6)
       cb6.set_label(r"$\mathcal{R}^{2D}$")

       #--------------------------------
       # Plot Avg Phase Across Sectors (2D)
       #--------------------------------
       ax7 = fig.add_subplot(gs[2, 1])
       im7 = ax7.imshow(np.flipud(Phi_a_sec), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = -np.pi, vmax = np.pi)
       ax7.set_xticks(angticks)
       ax7.set_xticklabels(angtickLabels_alpha)
       ax7.set_yticks(angticks)
       ax7.set_yticklabels(angtickLabels)
       ax7.set_ylabel(r"$\theta$")
       ax7.set_xlabel(r"$\alpha$")
       ax7.set_title(r"Average Angle Across Sectors (2D)")
       div7  = make_axes_locatable(ax7)
       cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
       cb7   = plt.colorbar(im7, cax = cbax7)
       cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       cb7.set_label(r"$\Phi^{2D}$")

       #--------------------------------
       # Plot Enstrophy Flux Across Sectors  (2D)
       #--------------------------------
       ax8 = fig.add_subplot(gs[2, 2])
       im8 = ax8.imshow(np.flipud(enst_flux_a_sec), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = -0.01, vmax = 0.01) #vmin = flux_lims[2], vmax = flux_lims[3] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
       ax8.set_xticks(angticks)
       ax8.set_xticklabels(angtickLabels_alpha)
       ax8.set_yticks(angticks)
       ax8.set_yticklabels(angtickLabels)
       ax8.set_ylabel(r"$\theta$")
       ax8.set_xlabel(r"$\alpha$")
       ax8.set_title(r"Enstrophy Flux Across Sectors (2D)")
       div8  = make_axes_locatable(ax8)
       cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
       cb8   = plt.colorbar(im8, cax = cbax8)
       cb8.set_label(r"$\Pi_{\mathcal{C}}^{2D}$")

       ## Add title and save fig
       plt.suptitle(r"$t = {:.5f}$".format(t))
       plt.savefig(out_dir + "/Phase_Sync_SNAP_{:05d}.png".format(i), bbox_inches = 'tight')
       plt.close()



def plot_sector_phase_sync_snaps_all(i, out_dir, w, enst_spec, enst_flux_sec, enst_flux_1d, enst_flux_2d, phases, theta_k3, R_sec, R_1d, R_2d, Phi_sec, Phi_1d, Phi_2d, flux_lims, t, x, y, Nx, Ny):

       """
       Plots the phases and average phase and sync per sector of the phases

       i       : int
               - The current snap
       out_dir : str
               - Path to the output folder
       phases  : ndarray, float64
               - 2D array of the Fourier phases, only ky > section 
       theta_k3   : array, float64
               - 1D Array containing the angles of the sector boundaries
       R       : array, float64
               - Array of the phase sync parameter for each sector
       Phi     : array, float64
               - Array of the average phase for each sector
       t       : float64
               - The current time 
       Nx      : int
               - Size of the first dimension
       Ny      : int
               - Size of the second dimension
       """

       ## Print Update
       print("SNAP: {}".format(i))

       ## Set up figure
       fig = plt.figure(figsize = (20, 13))
       gs  = GridSpec(4, 4, hspace = 0.35, wspace = 0.35)
       
       ## Generate colour map
       my_jet = cm.jet
       my_jet.set_under(color = "white")
       my_jet.set_over(color = "white")
       my_hsv = cm.hsv
       my_hsv.set_under(color = "white")
       my_hsv.set_over(color = "white")
       my_hot = cm.hot
       my_hot.set_under(color = "white")
       my_hot.set_over(color = "white")
       # my_magma = cm.magma
       # my_magma.set_under(color = "white")
       my_magma = mpl.colors.ListedColormap(cm.magma.colors[::-1])
       my_magma.set_under(color = "white")
       my_magma.set_over(color = "white")

       ## Create appropriate ticks and ticklabels for the sector angles
       dtheta_k3 = theta_k3[1] - theta_k3[0]
       angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
       angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       # angtickLabels_alpha = [r"$\theta - \frac{\pi}{2}$", r"$\theta - \frac{\pi}{3}$", r"$\theta - \frac{\pi}{4}$", r"$\theta - \frac{\pi}{6}$", r"$\theta + \frac{\pi}{6}$", r"$\theta + \frac{\pi}{4}$", r"$\theta + \frac{\pi}{3}$", r"$\theta + \frac{\pi}{2}$"]
       angtickLabels_alpha = np.linspace(0, len(theta_k3) + 1, num = len(angtickLabels), endpoint = False, dtype = "int64").tolist()
       theta_k3_min     = -np.pi/2 - dtheta_k3 / 2
       theta_k3_max     = np.pi/2 + dtheta_k3 /2
       alpha_angticks      = angticks
       alpha_angticklabels = angtickLabels 
       alpha_min = theta_k3_min
       alpha_max = theta_k3_max

       #--------------------------
       # Plot Phases  
       #--------------------------
       ax1 = fig.add_subplot(gs[0:2, 0:2])
       im1 = ax1.imshow(np.rot90(phases, k=-1), extent = (int(-Nx / 3), int(Nx / 3), int(Ny / 3), 0), cmap = my_hsv, vmin = 0., vmax = 2. * np.pi) 
       ang = np.arange(0, np.pi + np.pi / 100, np.pi / 100)
       kspace_angticks      = [0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2, 5*np.pi/8, 6*np.pi/8.0, 7*np.pi/8, np.pi]
       kspace_angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       kspace_angtickLabels.reverse()
       rmax_ext = int(Nx / 3 + 20)
       for tick, label in zip(kspace_angticks, kspace_angtickLabels):
              if tick == 0:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) + 5.0, s = label) ## shift the +-pi/2 to the right 
              elif tick > np.pi/2:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick) - 5.0, y = (rmax_ext - 7.5) * np.sin(tick) + 4.0, s = label) ## shift the bottom quadrant ticks down
              else:
                     ax1.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick), s = label) 
              ax1.plot([(rmax_ext - 15) * np.cos(tick), (rmax_ext - 11) * np.cos(tick)], [(rmax_ext - 15) * np.sin(tick), (rmax_ext - 11) * np.sin(tick)], color = 'k', linestyle = '-', linewidth = 0.5)
       ax1.plot((rmax_ext - 15) * np.cos(ang), (rmax_ext - 15) * np.sin(ang), color = 'k', linestyle = '--', linewidth = 0.5)
       ax1.set_xlabel(r"$k_x$")
       ax1.set_ylabel(r"$k_y$")
       ax1.set_xlim(-rmax_ext, rmax_ext)
       ax1.set_ylim(rmax_ext, 0)
       ax1.set_title(r"Fourier Phases")
       div1  = make_axes_locatable(ax1)
       cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
       cb1   = plt.colorbar(im1, cax = cbax1)
       cb1.set_label(r"$\phi_{\mathbf{k}}$")
       cb1.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
       cb1.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

       #--------------------------------
       # Plot Enstrophy Spectrum  (2D)
       #--------------------------------
       ax5  = fig.add_subplot(gs[0:2, 2:])
       enst_spec[Ny//3, Nx//3] = -10
       # im5  = ax5.imshow(enst_spec, extent = (-Ny / 3, Ny / 3, -Nx / 3, Nx / 3), cmap = my_magma, norm = mpl.colors.LogNorm()) # cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm() 
       im5 = ax5.imshow(np.rot90(enst_spec, k=-1), extent = (int(-Nx / 3), int(Nx / 3), int(Ny / 3), 0), cmap = my_magma, norm = mpl.colors.LogNorm(vmin=1e-12, vmax=1)) 
       ang = np.arange(0, np.pi + np.pi / 100, np.pi / 100)
       kspace_angticks      = [0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2, 5*np.pi/8, 6*np.pi/8.0, 7*np.pi/8, np.pi]
       kspace_angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
       kspace_angtickLabels.reverse()
       rmax_ext = int(Nx / 3 + 20)
       for tick, label in zip(kspace_angticks, kspace_angtickLabels):
              if tick == 0:
                     ax5.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick) + 5.0, s = label) ## shift the +-pi/2 to the right 
              elif tick > np.pi/2:
                     ax5.text(x = (rmax_ext - 7.5) * np.cos(tick) - 5.0, y = (rmax_ext - 7.5) * np.sin(tick) + 4.0, s = label) ## shift the bottom quadrant ticks down
              else:
                     ax5.text(x = (rmax_ext - 7.5) * np.cos(tick), y = (rmax_ext - 7.5) * np.sin(tick), s = label) 
              ax5.plot([(rmax_ext - 15) * np.cos(tick), (rmax_ext - 11) * np.cos(tick)], [(rmax_ext - 15) * np.sin(tick), (rmax_ext - 11) * np.sin(tick)], color = 'k', linestyle = '-', linewidth = 0.5)
       ax5.plot((rmax_ext - 15) * np.cos(ang), (rmax_ext - 15) * np.sin(ang), color = 'k', linestyle = '--', linewidth = 0.5)
       ax5.set_xlim(-rmax_ext, rmax_ext)
       ax5.set_ylim(rmax_ext, 0)
       ax5.set_xlabel(r"$k_y$")
       ax5.set_ylabel(r"$k_x$")
       ax5.set_title("Enstrophy Spectrum")
       div5  = make_axes_locatable(ax5)
       cbax5 = div5.append_axes("right", size = "10%", pad = 0.05)
       cb5   = plt.colorbar(im5, cax = cbax5)
       cb5.set_label(r"$\mathcal{E}_k$")

       #--------------------------
       # Plot Vorticity  
       #--------------------------
       ax2 = fig.add_subplot(gs[2, 0])
       im2 = ax2.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu", vmin = -6, vmax = 6) #, vmin = w_min, vmax = w_max 
       ax2.set_xlabel(r"$y$")
       ax2.set_ylabel(r"$x$")
       ax2.set_xlim(0.0, y[-1])
       ax2.set_ylim(0.0, x[-1])
       ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
       ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
       ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
       ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
       div2  = make_axes_locatable(ax2)
       cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
       cb2   = plt.colorbar(im2, cax = cbax2)
       cb2.set_label(r"$\omega(x, y)$")
       ax2.set_title(r"Vorticity")

       #--------------------------
       # Plot Order Parameters (1D)
       #--------------------------
       ax3 = fig.add_subplot(gs[2, 1])
       div3   = make_axes_locatable(ax3)
       axtop3 = div3.append_axes("top", size = "100%", pad = 0.2)
       axtop3.plot(theta_k3, Phi_1d, '.-', color = "orange")
       axtop3.set_xlim(theta_k3_min, theta_k3_max)
       axtop3.set_xticks(angticks)
       axtop3.set_xticklabels([])
       axtop3.set_ylim(-np.pi-0.2, np.pi+0.2)
       axtop3.set_yticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       axtop3.set_yticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       axtop3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       axtop3.set_title(r"Order Parameters (1D)")
       axtop3.set_ylabel(r"$\Phi^{1D}$")
       ax3.plot(theta_k3, R_1d)
       ax3.set_xlim(theta_k3_min, theta_k3_max)
       ax3.set_xticks(angticks)
       ax3.set_xticklabels(angtickLabels)
       ax3.set_ylim(0 - 0.05, 1 + 0.05)
       ax3.set_xlabel(r"$\theta$")
       ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax3.set_ylabel(r"$\mathcal{R}^{1D}$")

       #--------------------------------
       # Plot Enstrophy Flux Per Sector  (1D)
       #--------------------------------
       ax4 = fig.add_subplot(gs[2, 2])
       ax4.plot(theta_k3, enst_flux_1d, '.-')
       ax4.set_xlim(theta_k3_min, theta_k3_max)
       ax4.set_xticks(angticks)
       ax4.set_xticklabels(angtickLabels)
       ax4.set_xlabel(r"$\theta$")
       ax4.set_ylabel(r"$\Pi_{\mathcal{C}}^{1D}$")
       ax4.set_ylim(flux_lims[2], flux_lims[3])
       ax4.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax4.set_title(r"Enstrophy Flux Per Sector (1D)")

       #--------------------------
       # Plot Order Parameters Per Sector (Total)
       #--------------------------
       ax3 = fig.add_subplot(gs[2, 3])
       div3   = make_axes_locatable(ax3)
       axtop3 = div3.append_axes("top", size = "100%", pad = 0.2)
       axtop3.plot(theta_k3, Phi_sec, '.-', color = "orange")
       axtop3.set_xlim(theta_k3_min, theta_k3_max)
       axtop3.set_xticks(angticks)
       axtop3.set_xticklabels([])
       axtop3.set_ylim(-np.pi-0.2, np.pi+0.2)
       axtop3.set_yticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       axtop3.set_yticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       axtop3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       axtop3.set_title(r"Order Parameters Per Sector (Total)")
       axtop3.set_ylabel(r"$\Phi$")
       ax3.plot(theta_k3, R_sec)
       ax3.set_xlim(theta_k3_min, theta_k3_max)
       ax3.set_xticks(angticks)
       ax3.set_xticklabels(angtickLabels)
       ax3.set_ylim(0 - 0.05, 1 + 0.05)
       ax3.set_xlabel(r"$\theta$")
       ax3.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax3.set_ylabel(r"$\mathcal{R}$")

       #--------------------------------
       # Plot Enstrophy Flux Per Sector  Per Sector (Total)
       #--------------------------------
       ax4 = fig.add_subplot(gs[3, 3])
       ax4.plot(theta_k3, enst_flux_sec, '.-')
       ax4.set_xlim(theta_k3_min, theta_k3_max)
       ax4.set_xticks(angticks)
       ax4.set_xticklabels(angtickLabels)
       ax4.set_xlabel(r"$\theta$")
       ax4.set_ylabel(r"$\Pi_{\mathcal{C}}$")
       ax4.set_ylim(flux_lims[0], flux_lims[1])
       ax4.grid(which = 'both', axis = 'both', linestyle = ':', linewidth = '0.6', alpha = 0.8)
       ax4.set_title(r"Enstrophy Flux Per Sector (Total)")

       #--------------------------------
       # Plot Sync Across Sectors (2D)
       #--------------------------------
       ax6 = fig.add_subplot(gs[3, 0])
       im6 = ax6.imshow(np.flipud(R_2d), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = my_magma, vmin = 0.0, vmax = 1.0)
       ax6.set_xticks(angticks)
       ax6.set_xticklabels(angtickLabels_alpha)
       ax6.set_yticks(angticks)
       ax6.set_yticklabels(angtickLabels)
       ax6.set_ylabel(r"$\theta$")
       ax6.set_xlabel(r"$\alpha$")
       ax6.set_title(r"Sync Across Sectors (2D)")
       div6  = make_axes_locatable(ax6)
       cbax6 = div6.append_axes("right", size = "10%", pad = 0.05)
       cb6   = plt.colorbar(im6, cax = cbax6)
       cb6.set_label(r"$\mathcal{R}^{2D}$")

       #--------------------------------
       # Plot Avg Phase Across Sectors (2D)
       #--------------------------------
       ax7 = fig.add_subplot(gs[3, 1])
       im7 = ax7.imshow(np.flipud(Phi_2d), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = -np.pi, vmax = np.pi)
       ax7.set_xticks(angticks)
       ax7.set_xticklabels(angtickLabels_alpha)
       ax7.set_yticks(angticks)
       ax7.set_yticklabels(angtickLabels)
       ax7.set_ylabel(r"$\theta$")
       ax7.set_xlabel(r"$\alpha$")
       ax7.set_title(r"Average Angle Across Sectors (2D)")
       div7  = make_axes_locatable(ax7)
       cbax7 = div7.append_axes("right", size = "10%", pad = 0.05)
       cb7   = plt.colorbar(im7, cax = cbax7)
       cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
       cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
       cb7.set_label(r"$\Phi^{2D}$")

       #--------------------------------
       # Plot Enstrophy Flux Across Sectors  (2D)
       #--------------------------------
       ax8 = fig.add_subplot(gs[3, 2])
       im8 = ax8.imshow(np.flipud(enst_flux_2d), extent = (theta_k3_min, theta_k3_max, alpha_min, alpha_max), cmap = "seismic", vmin = flux_lims[4], vmax = flux_lims[5]) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
       ax8.set_xticks(angticks)
       ax8.set_xticklabels(angtickLabels_alpha)
       ax8.set_yticks(angticks)
       ax8.set_yticklabels(angtickLabels)
       ax8.set_ylabel(r"$\theta$")
       ax8.set_xlabel(r"$\alpha$")
       ax8.set_title(r"Enstrophy Flux Across Sectors (2D)")
       div8  = make_axes_locatable(ax8)
       cbax8 = div8.append_axes("right", size = "10%", pad = 0.05)
       cb8   = plt.colorbar(im8, cax = cbax8)
       cb8.set_label(r"$\Pi_{\mathcal{C}}^{2D}$")

       ## Add title and save fig
       plt.suptitle(r"$t = {:.5f}$".format(t))
       plt.savefig(out_dir + "/Phase_Sync_SNAP_{:05d}.png".format(i), bbox_inches = 'tight')
       plt.close()


#############################
##       COLOURMAPS        ##
#############################
## Colourmap
colours = [[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0.25, 0], [1, 1, 1]]   #located @ 0, pi/2, pi, 3pi/2 and 2pi
# my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', colours, N = kmax)                            # set N to inertial range
my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', colours)                            # set N to inertial range

# ## Phases Colourmap
# myhsv   = cm.hsv(np.arange(255))
# norm    = mpl.colors.Normalize(vmin = 0.0, vmax = 2.0*np.pi)
# my_mhsv = mpl.colors.LinearSegmentedColormap.from_list('my_map', myhsv) # set N to inertial range
# phase_colors = cm.ScalarMappable(norm = norm, cmap = my_mhsv)               # map the values to rgba tuple