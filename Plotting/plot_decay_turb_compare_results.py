#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to compare solver results with decaying turbulence papers
#        Solver data

#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os
from numba import njit
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import getopt
import pyfftw as fftw

from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, enstrophy_spectrum, energy_spectrum, total_energy, fftw_init_2D, fft, ifft
from functions import compute_pdf
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
        
        def __init__(self, in_dir = None, out_dir = None, post_file = None):
            self.in_dir     = in_dir
            self.out_dir    = out_dir
            self.post_file  = post_file

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:")
    except:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:
        
        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("Input Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)
        elif opt in ['-f']:
            ## Read input directory
            cargs.post_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.post_file) + tc.Rst)


    return cargs

def enst_spec(w_h, k2):

    """
    Alternative method for computing the enstrophy spectrum
    """

    ## Get dims
    Nx = w_h.shape[0]
    Nk = w_h.shape[1]

    ## Get |k|
    sqrt_k = np.sqrt(k2)

    ## Compute the enstrophy
    enst = np.absolute(w_h * np.conjugate(w_h))

    ## Set wavenumber and spectrum arrays
    k    = np.arange(1, Nk)
    spec = np.zeros_like(k)

    for i in range(len(k)):
        spec[i] += np.sum(enst[(sqrt_k <= k[i] + 0.5) & (sqrt_k > k[i] - 0.5)])

    return spec * (0.5 / Nx**2) * 4.0 * np.pi**2, k

def enrg_spec(w_h, k2, k2Inv):

    """
    Alternative method for computing the energy spectrum
    """

    ## Get dims
    Nx = w_h.shape[0]
    Nk = w_h.shape[1]

    ## Get |k|
    sqrt_k = np.sqrt(k2)

    ## Compute the energy
    enrg = np.absolute(w_h * np.conjugate(w_h)) * k2Inv

    ## Set wavenumber and spectrum arrays
    k    = np.arange(1, Nk)
    spec = np.zeros_like(k)

    for i in range(len(k)):
        spec[i] += np.sum(enrg[(sqrt_k <= k[i] + 0.5) & (sqrt_k > k[i] - 0.5)])

    return spec * (0.5 / Nx**2) * 4.0 * np.pi**2, k

def iso_enst_spec(w_h, dx, dy, k2):

    """
    Alternative method for computing the enstrophy spectrum - computes using radial transformation
    """

    ## Get dims
    Nk = w_h.shape[1]

    ## Get |k|
    sqrt_k = np.sqrt(k2)

    ## Compute the enstrophy
    enst = np.absolute(w_h * np.conjugate(w_h))

    ## Compute the radial wavenumber
    dk_radial = np.sqrt(dx**2 + dy**2)
    k_radial  = np.arange(dk_radial / 2., Nk + dk_radial, 1)
    spec      = np.zeros_like(k_radial)

    for i in range(len(k_radial)):
        indx    = (sqrt_k >= k_radial[i] - dk_radial / 2.) & (sqrt_k <= k_radial[i] + dk_radial / 2.)
        d_theta = np.pi / (indx.sum() - 1)
        spec[i] = np.sum(enst[indx]) * k_radial[i] * d_theta

    return spec, k_radial


def iso_enrg_spec(w_h, dx, dy, k2, k2Inv):

    """
    Alternative method for computing the energy spectrum - computes using radial transformation
    """

    ## Get dims
    Nk = w_h.shape[1]

    ## Get |k|
    sqrt_k = np.sqrt(k2)

    ## Compute the energy
    enrg = np.absolute(w_h * np.conjugate(w_h)) * k2Inv

    ## Compute the radial wavenumber
    dk_radial = np.sqrt(dx**2 + dy**2)
    k_radial  = np.arange(dk_radial / 2., Nk + dk_radial, 1)
    spec      = np.zeros_like(k_radial)

    for i in range(len(k_radial)):
        indx    = (sqrt_k >= k_radial[i] - dk_radial / 2.) & (sqrt_k <= k_radial[i] + dk_radial / 2.)
        d_theta = np.pi / (indx.sum() - 1)
        spec[i] = np.sum(enrg[indx]) * k_radial[i] * d_theta

    return spec, k_radial


def create_initial_conditions(psi_hat, fft2d, ifft2d, kx, ky, Nx, Ny):

    ## Transform abck to real space and zero the mean
    psi = ifft(ifft2d, psi_hat)
    # psi = np.fft.ifft2(psi_hat)
    # psi -= np.mean(psi)

    ## Transform to Fourier space
    psi_hat = fft(fft2d, psi)
    # psi_hat = np.fft.fft2(psi)

    ## Compute the total energy
    tot_enrg = 0.
    for i in range(psi_hat.shape[0]):
        for j in range(psi_hat.shape[1]):
            k_sqr = kx[i]**2 + ky[j]**2

            if j == 0 or j == psi_hat.shape[1] - 1:
                tot_enrg += np.absolute(psi_hat[i, j] * np.conjugate(psi_hat[i, j])) * k_sqr
            else:
                tot_enrg += 2. * np.absolute(psi_hat[i, j] * np.conjugate(psi_hat[i, j])) * k_sqr
    tot_enrg *= 4. * np.pi**2 * (0.5 / Nx * Ny)

    ## Normalize the initial energy
    psi_hat *= np.sqrt(0.5 / tot_enrg) 

    ## Compute the fourier vorticity
    w_h = np.empty_like(psi_hat)
    for i in range(psi_hat.shape[0]):
            for j in range(psi_hat.shape[1]): 
                k_sqr = kx[i]**2 + ky[j]**2

                w_h[i, j] = k_sqr * psi_hat[i, j]

    return w_h, psi_hat, psi

######################
##       MAIN       ##
######################
if __name__ == '__main__':
  
    # -------------------------------------
    ## --------- Parse Commnad Line
    # -------------------------------------
    cmdargs = parse_cml(sys.argv[1:]) 
    if cmdargs.post_file is None:
        print("Here", cmdargs.post_file)
        method = "default"
        post_file_path = cmdargs.in_dir
    else: 
        method = "file"
        post_file_path = cmdargs.in_dir + cmdargs.post_file

    # -----------------------------------------
    # # --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars)

    # if run_data.no_w:
    #     print("\nPreparing real space vorticity...", end = " ")
    #     for i in range(sys_vars.ndata):
    #         run_data.w[i, :, :] = np.fft.irfft2(run_data.w_hat[i, :, :])
    #     print("Finished!")
        
    ## Read in spectra data
    spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

    ## Read in post processing data
    post_data = import_post_processing_data(post_file_path, sys_vars, method)

    


    # with h5py.File(cmdargs.in_dir + "/Test_Data.h5", 'r') as f:
        
    #     psi_hat   = f["psi_hat"][:, :]
    #     psi       = f["psi"][:, :]  
    #     psi_hat_z = f["psi_hat_zero_mean"][:, :]
    #     w_h_b     = f["w_h_before_dealias"][:, :]
    #     w_h_benst = f["w_h_before_enstrophy"][:, :]
    #     w_h_benrg = f["w_h_before_energy"][:, :]
        

    # with h5py.File("/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/navierstokes_mpi/3D_Euler_Spectral/results/Test_Data.h5", "r") as f:
    #     psi_hat_bren = f["psi_hat"][:, :]

    # with h5py.File("/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/navierstokes_mpi/3D_Euler_Spectral/results/RESULTS_NAVIER_RK4_N[512][512]_T[0-0.01]_[22-14-46]_[Decay-Alt]/HDF_Global_FOURIER.h5", "r") as f:
    #     w_h_bren = np.ones((1001, sys_vars.Nx, sys_vars.Nk)) * np.complex(0.0, 0.0)
    #     ## Initialize counter
    #     nn = 0        
    #     # Read in the vorticity
    #     for group in f.keys():
    #         if "Timestep" in group:
    #             w_h_bren[nn, :, :] = f[group]["W_hat"][:, :] ##.astype('complex128')
    #         nn += 1

    # ## Initialize pyfftw transforms
    # fft2d, ifft2d = fftw_init_2D(sys_vars.Nx, sys_vars.Ny)

    # w_h_py, psi_hat_py, psi_py = create_initial_conditions(psi_hat, fft2d, ifft2d, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)

    # print("Spectra:")
    # print(np.allclose(w_h_benrg, w_h_benst, rtol = 1e-14))
    # print("IC:")
    # print(np.allclose(w_h_benrg, run_data.w_hat[0, :, :], rtol = 1e-14))
    # print("IC w_h:")
    # print(np.allclose(post_data.w_hat[0, :, :], run_data.w_hat[0, :, :], rtol = 1e-14))
    
    # print("Bren Psi")
    # print(np.allclose(psi_hat, psi_hat_bren))
    # print("Bren IC")
    # print(np.allclose(run_data.w_hat[0, :, :], w_h_bren[0, :, :]))
    # print()

    # print("Psi:")
    # print(np.allclose(psi, psi_py))
    # print()
    # print(np.allclose(psi_hat_z, psi_hat_py))
    # print()
    # print(np.allclose(w_h_py, w_h_b))

    # for i in range(2):
    #     for j in range(10):
    #         print("psih[{}, {}]: {} {} I | {} {} I\t".format(i, j, np.real(psi_hat_py[i, j]), np.imag(psi_hat_py[i, j]), np.real(psi_hat_z[i, j]), np.imag(psi_hat_z[i, j])))
    #     print()
    # print()
    # print()
    # for i in range(2):
    #     for j in range(10):
    #         print("wh[{}, {}]: {} {} I | {} {} I\t".format(i, j, np.real(w_h_py[i, j]), np.imag(w_h_py[i, j]), np.real(w_h_b[i, j]), np.imag(w_h_b[i, j])))
    #     print()

    # ## Testing Initial Spectra
    # fig = plt.figure(figsize = (16, 8))
    # gs  = GridSpec(1, 2)
    # ax1 = fig.add_subplot(gs[0, 0])
    # spec1,_ = energy_spectrum(w_h_b, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    # spec2,_ = energy_spectrum(w_h_benst, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    # spec3,_ = energy_spectrum(w_h_benrg, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    # ax1.plot(spec1)
    # ax1.plot(spec2)
    # ax1.plot(spec3)
    # ax1.plot(kk, spec_data.enrg_spectrum[0, 1:kmax])
    # ax1.set_xlabel(r"$|\mathbf{k}|$")
    # ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.legend([r"1", r"2", r"3", "Solver", "Isotropic"])
    
    # ax2 = fig.add_subplot(gs[0, 1])
    # spec1,_ = enstrophy_spectrum(w_h_b, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    # spec2,_ = enstrophy_spectrum(w_h_benst, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    # spec3,_ = enstrophy_spectrum(w_h_benrg, run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    # ax2.plot(spec1)
    # ax2.plot(spec2)
    # ax2.plot(spec3)
    # ax2.plot(kk, spec_data.enst_spectrum[0, 1:kmax])
    # ax2.set_xlabel(r"$|\mathbf{k}|$")
    # ax2.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.legend([r"1", r"2", r"3", "Solver", "Isotropic"])
    # plt.savefig(cmdargs.out_dir + "Initial_Spectra_Test.png", bbox_inches = 'tight') 
    # plt.close()


    # -----------------------------------------
    ## --------  Plot Data
    # -----------------------------------------
    ## System Measures plot
    fig = plt.figure(figsize = (16, 8))
    kmax = int(sys_vars.Nx/3 + 1)
    kk = np.arange(1, kmax)
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(run_data.time, run_data.tot_enrg[:] / run_data.tot_enrg[0])
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\mathcal{K}(t) / \mathcal{K}(0)$")
    ax1.set_xlim(run_data.time[0], run_data.time[-1])
    ax1.set_ylim(0.5, 1.0)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(run_data.time, run_data.tot_enst[:] / run_data.tot_enst[0])
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\mathcal{E}(t) / \mathcal{E}(0)$")
    ax2.set_xlim(run_data.time[0], run_data.time[-1])
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0.01, 11.0)
    ax2.set_ylim(1e-2, 1.01)
    plt.savefig(cmdargs.out_dir + "DecayTurb_System_Measures.png", bbox_inches = 'tight') 
    plt.close()


    ## Vorticity Field
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 3)
    if sys_vars.T == 1:
        indexes = [1, 3, 10]
    else:
        indexes = [11, 31, 100]
    for i, indx in enumerate(indexes):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(run_data.w[indx, :, :], extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
        ax.set_xlabel(r"$y$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0.0, run_data.y[-1])
        ax.set_ylim(0.0, run_data.x[-1])
        ax.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
        ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
        ax.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
        ax.set_title(r"$t = {:0.5f}$".format(run_data.time[indx]))
        div  = make_axes_locatable(ax)
        cbax = div.append_axes("right", size = "10%", pad = 0.05)
        cb   = plt.colorbar(im, cax = cbax)
        cb.set_label(r"$\omega(x, y)$")
    plt.savefig(cmdargs.out_dir + "DecayTurb_VorticityField.png", bbox_inches = 'tight') 
    plt.close()
    

    ## Testing Initial Spectra
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(kk, spec_data.enrg_spectrum[0, 1:kmax], '*-')
    ax1.plot(kk, post_data.enrg_spectrum_1d[0, 1:kmax], '.-')
    # ax1.plot(kk, post_data.enrg_spectrum_1d_alt[0, 1:kmax])
    if sys_vars.u0 == "DECAY_TURB_II" or sys_vars.u0 == "GAUSS_DECAY_TURB" or sys_vars.u0 == "DECAY_TURB_NB":
        spec_ic = (kk**6) / ((1 + kk/60)**18) / (10**7.6)
    elif sys_vars.u0 == "DECAY_TURB" :
        spec_ic = (kk) / ((1 + (kk**4)/6)) / (10**0.7)
    elif sys_vars.u0 == "DECAY_TURB_ALT":
        spec_ic = kk / (((1.0 + ((kk/6)**4)))) / (10**1.8)
    elif sys_vars.u0 == "DECAY_TURB_EXP":
        kp = 8.0
        spec_ic = (kk**7 / kp**8) * np.exp(-3.5 * (kk / kp)**2)
    ax1.plot(kk, spec_ic, '--')
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(["Solver", "Post", "Post Alt", "Paper IC"])
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(kk, spec_data.enst_spectrum[0, 1:kmax], '*-')
    ax2.plot(kk, post_data.enst_spectrum_1d[0, 1:kmax], '.-')
    # ax2.plot(kk, post_data.enst_spectrum_1d_alt[0, 1:kmax])
    if sys_vars.u0 == "DECAY_TURB_II" or sys_vars.u0 == "GAUSS_DECAY_TURB" or sys_vars.u0 == "DECAY_TURB_NB":
        spec_ic = (kk**8) / ((1 + kk/60)**18) / (10**7.6)
    elif sys_vars.u0 == "DECAY_TURB" :
        spec_ic = (kk**3) / ((1 + (kk**4)/6)) / (10**0.6)
    elif sys_vars.u0 == "DECAY_TURB_ALT":
        indx = kk != 0
        spec_ic[indx] = (kk[indx]**3) / ( (1 + (kk[indx]/6)**4)) / (10**1.8)
    elif sys_vars.u0 == "DECAY_TURB_EXP":
        kp = 8.0
        indx = kk_spec != 0
        spec_kk[indx] = (kk_spec[indx]**9 / kp**8) * np.exp(-3.5 * (kk_spec[indx] / kp)**2)
    ax2.plot(kk, spec_ic, '--')
    ax2.set_xlabel(r"$|\mathbf{k}|$")
    ax2.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(["Solver", "Post", "Post Alt", "Paper IC"])
    plt.savefig(cmdargs.out_dir + "Initial_SpectraMethods_Test.png", bbox_inches = 'tight') 
    plt.close()
        
    ## Energy Spectrum Test Initial Condition
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    kmax = int(sys_vars.Nx/3 + 1)
    kk = np.arange(1, kmax)
    kk_spec = np.ones((kmax, ))
    spec_ic = np.ones((kmax -1, ))
    spec_kk = np.ones((kmax, ))
    spec_indx_arr     = np.zeros((sys_vars.Nx, sys_vars.Nk))
    spec_indx_arr_int = np.zeros((sys_vars.Nx, sys_vars.Nk))
    for i in range(sys_vars.Nx):
        for j in range(sys_vars.Nk):
            spec_indx = np.sqrt(run_data.kx[i]**2 + run_data.ky[j]**2)
            spec_indx_arr[i, j] = spec_indx
            spec_indx_arr_int[i, j] = int(spec_indx)            
            if int(spec_indx) < kmax - 1:
                kk_spec[int(spec_indx)] = spec_indx
    if sys_vars.u0 == "DECAY_TURB_II" or sys_vars.u0 == "GAUSS_DECAY_TURB" or sys_vars.u0 == "DECAY_TURB_NB":
        spec_ic = (kk**6) / ((1 + kk/60)**18) / (10**7.6)
        spec_kk = (kk_spec**6) / ((1 + (kk_spec)/60)**18) / (10**7.6)
    elif sys_vars.u0 == "DECAY_TURB" :
        spec_ic = (kk) / ((1 + (kk**4)/6)) / (10**0.7)
        spec_kk = (kk_spec) / ((1 + (kk_spec**4)/6)) / (10**1.1)
    elif sys_vars.u0 == "DECAY_TURB_ALT":
        indx = kk_spec != 0
        spec_ic = kk / (((1.0 + ((kk/6)**4)))) / (10**1.8)
        spec_kk[indx] = kk_spec[indx] / ( ((1.0 + ((kk_spec[indx]/6)**4)))) / (10**1.8)
    elif sys_vars.u0 == "DECAY_TURB_EXP":
        kp = 8.0
        spec_ic = (kk**7 / kp**8) * np.exp(-3.5 * (kk / kp)**2)
        indx = kk_spec != 0
        spec_kk[indx] = (kk_spec[indx]**7 / kp**8) * np.exp(-3.5 * (kk_spec[indx] / kp)**2)
    ax1.plot(kk, spec_ic, '--')
    ax1.plot(kk, post_data.enrg_spectrum_1d[0, 1:kmax], '.-' )
    ax1.plot(kk, spec_data.enrg_spectrum[0, 1:kmax], '*')
    spec, _  = energy_spectrum(run_data.w_hat[0, :, :], run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    ax1.plot(kk, spec[1:kmax], 'o')    
    # enrg_spec_alt, k = enrg_spec(run_data.w_hat[1, :, :], run_data.k2, run_data.k2Inv)
    # ax1.plot(k, enrg_spec_alt / 10**5.5, '*-')
    # ax1.plot(kk, spec_kk[1:])
    # iso_enrg, k_rad = iso_enrg_spec(run_data.w_hat[0, :, :], 2.0 * np.pi / sys_vars.Nx, 2.0 * np.pi / sys_vars.Ny, run_data.k2, run_data.k2Inv)
    # ax1.plot(k_rad, iso_enrg / 10**9)
    # ax1.plot(kk, post_data.enrg_spectrum_1d_alt[0, 1:kmax], '*-')
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend([r"Paper", r"Post", r"IC", r"Py", r"Sqrt_k", r"Alt", r"Isotropic", r"Post Alt"])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnergySpectrum_TEST_IC.png", bbox_inches = 'tight') 
    plt.close()

    ## Enstrophy Spectrum Test Initial Condition
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    kmax = int(sys_vars.Nx/3 + 1)
    kk = np.arange(1, kmax)
    spec_ic = np.ones((kmax - 1, ))
    if sys_vars.u0 == "DECAY_TURB_II" or sys_vars.u0 == "GAUSS_DECAY_TURB" or sys_vars.u0 == "DECAY_TURB_NB":
        spec_ic = (kk**8) / ((1 + kk/60)**18) / (10**7.6)
    elif sys_vars.u0 == "DECAY_TURB" :
        spec_ic = (kk**3) / ((1 + (kk**4)/6)) / (10**0.6)
    elif sys_vars.u0 == "DECAY_TURB_ALT":
        indx = kk != 0
        spec_ic[indx] = (kk[indx]**3) / ( (1 + (kk[indx]/6)**4)) / (10**1.8)
    elif sys_vars.u0 == "DECAY_TURB_EXP":
        kp = 8.0
        indx = kk_spec != 0
        spec_kk[indx] = (kk_spec[indx]**9 / kp**8) * np.exp(-3.5 * (kk_spec[indx] / kp)**2)
    ax1.plot(kk, spec_ic, '--')
    ax1.plot(kk, spec_data.enst_spectrum[0, 1:kmax], '.-')
    spec, _  = enstrophy_spectrum(run_data.w_hat[0, :, :], run_data.kx, run_data.ky, sys_vars.Nx, sys_vars.Ny)
    ax1.plot(kk, spec[1:kmax], 'o')
    ax1.plot(kk, post_data.enst_spectrum_1d[0, 1:kmax], '*')
    # enst_spec_alt, k = enst_spec(run_data.w_hat[0, :, :], run_data.k2)
    # ax1.plot(k, enst_spec_alt / 10**5)
    # iso_enst, k_rad = iso_enst_spec(run_data.w_hat[0, :, :], 2.0 * np.pi / sys_vars.Nx, 2.0 * np.pi / sys_vars.Ny, run_data.k2)
    # ax1.plot(k_rad, iso_enrg / 10**9)
    # ax1.plot(kk, post_data.enst_spectrum_1d_alt[0, 1:kmax], '*-')
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend([r"Paper", r"IC", r"Py", r"Post", "Alt", "Isotropic", r"Post Alt"])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnstrophySpectrum_TEST_IC.png", bbox_inches = 'tight') 
    plt.close()


    ## Energy Spectrum
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]:
        ax1.plot(spec_data.enrg_spectrum[i, 1:kmax])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1, kmax)
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_EnergySpectrum.png", bbox_inches = 'tight') 
    plt.close()

    ## Vorticity & Velocity Distribution
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]:
        # bin_width   = post_data.w_pdf_ranges[i, 1] - post_data.w_pdf_ranges[i, 0]
        # bin_centres = (post_data.w_pdf_ranges[i, 1:] + post_data.w_pdf_ranges[i, :-1]) * 0.5
        # pdf = post_data.w_pdf_counts[i, :] / (np.sum((post_data.w_pdf_counts[i, :])) * bin_width)
        # var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        bin_counts, bin_ranges = np.histogram(run_data.w[i, :], bins=100)
        bin_centres, pdf = compute_pdf(bin_ranges,bin_counts)
        ax1.plot(bin_centres, pdf)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"PDF")
    ax1.set_yscale('log')
    ax1.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]])
    ax2 = fig.add_subplot(gs[0, 1])
    for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]:
        # bin_width   = post_data.u_pdf_ranges[i, 1] - post_data.u_pdf_ranges[i, 0]
        # bin_centres = (post_data.u_pdf_ranges[i, 1:] + post_data.u_pdf_ranges[i, :-1]) * 0.5
        # pdf = post_data.u_pdf_counts[i, :] / (np.sum((post_data.u_pdf_counts[i, :])) * bin_width)
        # var = np.sqrt(np.sum(pdf * bin_centres**2 * bin_width))
        bin_counts, bin_ranges = np.histogram(run_data.u[i, :], bins=100)
        bin_centres, pdf = compute_pdf(bin_ranges,bin_counts)
        ax2.plot(bin_centres, pdf)
    ax2.set_xlabel(r"$\mathbf{u}$")
    ax2.set_ylabel(r"PDF")
    ax2.set_yscale('log') 
    ax2.set_xlim(0, 1)   
    ax2.legend([r"$t = {:0.2}$".format(i * (run_data.time[1] - run_data.time[0])) for i in [0, int(sys_vars.ndata/2), sys_vars.ndata - 1]])
    plt.savefig(cmdargs.out_dir + "DecayTurb_VelVortPDFs_C.png", bbox_inches='tight') 
    plt.close()

    ## Flux Spectra
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.cumsum(spec_data.enrg_flux_spectrum[-1, :]))
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    ax1.set_title(r"Energy Flux Spectra")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.cumsum(spec_data.enst_flux_spectrum[-1, :]))
    ax2.set_title(r"Enstrophy Flux Spectra")
    ax2.set_xlabel(r"$|\mathbf{k}|$")
    ax2.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
    plt.savefig(cmdargs.out_dir + "DecayTurb_FluxSpectra.png", bbox_inches='tight') 
    plt.close()