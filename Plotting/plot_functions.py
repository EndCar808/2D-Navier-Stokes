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
import matplotlib as mpl
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
def plot_summary_snaps(out_dir, i, w, w_hat, x, y, w_min, w_max, kx, ky, kx_max, tot_en, tot_ens, tot_pal, enrg_spec, enst_spec, enrg_diss, enst_diss, enrg_flux_sb, enrg_diss_sb, enst_flux_sb, enst_diss_sb, time, Nx, Ny):

    """
    Plots summary snaps for each iteration of the simulation. Plot: vorticity, energy and enstrophy spectra, dissipation, flux and totals.
    """
    
    ## Print Update
    print("SNAP: {}".format(i))

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(4, 2, hspace = 0.6, wspace = 0.3)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0:2, 0:1])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu", vmin = w_min, vmax = w_max) 
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
    ax2.set_ylim(1e-12, 10)
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
    ax3.set_ylim(1e-12, 10)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    # ax3.text(x = kx[61],  y = kpower[-1], s = r"$k^{-5/3}$")


    #--------------------------
    # Plot System Measures   
    #-------------------------- 
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time[:i], tot_en)
    ax4.plot(time[:i], tot_ens)
    ax4.plot(time[:i], tot_pal)
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

    #--------------------------
    # Plot Energy Flux 
    #-------------------------- 
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(time[:i], enrg_flux_sb)
    ax6.plot(time[:i], enrg_diss_sb)
    ax6.set_xlabel(r"$t$")
    ax6.set_xlim(time[0], time[-1])
    ax6.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax6.legend([r"Energy Flux", r"Energy Diss"])

    #--------------------------
    # Plot Energy Flux 
    #-------------------------- 
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(time[:i], enst_flux_sb)
    ax7.plot(time[:i], enst_diss_sb)
    ax7.set_xlabel(r"$t$")
    ax7.set_xlim(time[0], time[-1])
    ax7.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
    ax7.legend([r"Enstrophy Flux", r"Enstrophy Diss"])

    ## Save figure
    plt.savefig(out_dir + "SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()



def plot_phase_snaps(out_dir, i, w, w_h, w_min, w_max, x, y, time, Nx, Ny, k2Inv, kx, ky):

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

    ## Create Figure
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(2, 2, hspace = 0.35, wspace = -0.45)

    ##-------------------------
    ## Plot vorticity   
    ##-------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(w, extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu", vmin = w_min, vmax = w_max) 
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
    data = np.mod(np.angle(ZeroCentredField(w_h)), 2. * np.pi)
    # data = np.mod(np.angle(np.fft.ifftshift(FullField(w_h))), 2. * np.pi)
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
    ax3  = fig.add_subplot(gs[1, 0])
    spec = np.absolute(ZeroCentredField(w_h)) ** 2
    print(np.any(spec < 0))
    print()

    im3  = ax3.imshow(spec / np.sum(spec), extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1])) # , norm = mpl.colors.LogNorm(), vmax = 10, vmin = 10^-15
    # im3 = ax3.imshow(np.absolute(np.fft.ifftshift(FullField(w_h))) ** 2, extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm())
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
    ax4  = fig.add_subplot(gs[1, 1])
    spec = np.absolute(ZeroCentredField(w_h * k2Inv.astype('complex128')))** 2
    print(np.any(spec < 0))
    print()
    im4  = ax4.imshow(spec / np.sum(spec), extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1])) # , norm = mpl.colors.LogNorm(), vmax = 10, vmin = 10^-15
    # im4 = ax4.imshow(np.absolute(np.fft.ifftshift(FullField(w_h * k2Inv.astype('complex128')))) ** 2, extent = (Ny / 2, -Ny / 2 + 1, -Nx / 2 + 1, Nx / 2), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), norm = mpl.colors.LogNorm())
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
    plt.savefig(out_dir + "Phase_SNAP_{:05d}.png".format(i), bbox_inches='tight') 
    plt.close()



##################################
##       HELPER FUNCTIONS       ##
##################################
# @njit
def fft_ishift_freq(w_h, axes = None):

 """
 My version of numpy.fft.ifftshift - adjusted for FFTW wavenumber ordering

 w_h   : ndarray, complex128
        - Array containing the Fourier variables of a given field e.g. Fourier vorticity or velocity
 axes  : int or tuple
        - Specifies which axes to perform the shift over
 """

 ## If no axes provided
 if axes == None:
     ## Create axes tuple
     axes  = tuple(range(w_h.ndim))
     ## Create shift list -> adjusted for FFTW freq numbering
     shift = [-(dim // 2 + 1) for dim in w_h.shape]

 ## If axes is an integer
 elif isinstance(axes, int):
     ## Create the shift object on this axes
     shift = -(w_h.shape[axes] // 2 + 1)

 ## If axes is a tuple
 else:
     ## Create appropriate shift for each axis
     shift = [-(w_h.shape[ax] // 2 + 1) for ax in axes]

 return np.roll(w_h, shift, axes)

# @njit
def ZeroCentredField(w_h):

    """ 
    Returns the zero centred full field in Fourier space.

    Input Parameters:
    w_h : ndarray, complex128
         - Array containing the Fourier variables of a given field e.g. Fourier vorticity or velocity
         ordered according to FFTW library
    """

    return np.flipud(fft_ishift_freq(FullField(w_h)))

# @njit
def FullField(w_h):

    """
    Returns the full field of an containing the Fourier variables e.g. Fourier vorticity or velocity.

    w_h : ndarray, complex128
         - Array containing the Fourier variables of a given field e.g. Fourier vorticity or velocity
         ordered according to FFTW library
    """

    return np.concatenate((w_h, np.conjugate(w_h[:, -2:0:-1])), axis = 1)




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