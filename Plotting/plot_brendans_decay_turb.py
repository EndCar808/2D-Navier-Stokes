#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##
######################
import matplotlib as mpl
# mpl.use('PDF') # Use this backend for writing plots to file
mpl.use('Agg') # Use this backend for writing plots to file
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




######################
##       MAIN       ##
######################
if __name__ == '__main__':


    #--------------------------------
    ## --------- System Parameters
    #--------------------------------
    Nx   = 512
    Ny   = 512
    Nyf  = int(Ny / 2 + 1)
    t0   = 0.0
    T    = 0.01 
    date = "22-14-46"
    u0   = "DECAY_TURB_ALT"
    tag  = "Decay-Alt"
    endas_u0 = "DECAY_TURB_ALT"
    #-------------------------------
    ## --------- Directories
    #-------------------------------
    input_folder = "RESULTS_NAVIER_RK4_N[{}][{}]_T[{}-{:0.2f}]_[{}]_[{}]".format(Nx, Ny, int(t0), T, date, tag)
    input_dir    = "/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/navierstokes_mpi/3D_Euler_Spectral/results/" + input_folder
    output_dir   = input_dir

    #--------------------------------------
    ## --------- Open Files / Read In Data
    #--------------------------------------
    with h5py.File(input_dir + "/HDF_Global_FOURIER.h5", 'r') as file:
        
        ## Get the number of data saves
        num_saves = len([g for g in list(file.keys()) if 'Time' in g])

        ## Allocate arrays
        w_hat   = np.ones((num_saves, Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)
        time    = np.zeros((num_saves, ))
        Real    = 0
        Fourier = 0

        # Read in the vorticity
        for i, group in enumerate(file.keys()):
            if "Time" in group:
                if 'w_hat' in list(file[group].keys()):
                    w_hat[i, :, :] = file[group]["w_hat"][:, :]
                    Fourier = 1
                time[i] = file[group].attrs["TimeValue"]
        kx = file['Timestep_0000']['kx'][:]
        ky = file['Timestep_0000']['ky'][:]

    with h5py.File(input_dir + "/HDF_Energy_Spect.h5", 'r') as file:
        
        ## Get the number of data saves
        num_saves = len([g for g in list(file.keys()) if 'Time' in g])

        n_spect = int(np.sqrt((Nx/2)**2 + (Ny/2)**2) + 1)
        ## Allocate arrays
        enrg_spect = np.zeros((num_saves, n_spect))
        enst_spect = np.zeros((num_saves, n_spect))

        ## Read in data
        for i, group in enumerate(file.keys()):
            if "Time" in group:
                enrg_spect[i, :] = file[group]["EnergySpectrum"][:]
                enst_spect[i, :] = file[group]["EnstrophySpectrum"][:]


    #--------------------------------------
    ## --------- Plot Data
    #--------------------------------------
    ## Energy Spectrum Test Initial Condition
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    kmax = int(Nx/3 + 1)
    kk = np.arange(1, kmax)
    kk_spec = np.ones((kmax, ))
    spec_ic = np.ones((kmax -1, ))
    for i in range(Nx):
        for j in range(Nyf):
            spec_indx = np.sqrt(kx[i]**2 + ky[j]**2)
            if int(spec_indx) < kmax - 1:
                kk_spec[int(spec_indx)] = spec_indx
    if endas_u0 == "DECAY_TURB_II":
        spec_ic = (kk**6) / ((1 + kk/60)**18) / (10**9.5)
        spec_kk = (kk_spec**6) / ((1 + (kk_spec)/60)**18) / (10**9.5)

    elif endas_u0 == "DECAY_TURB" or endas_u0 == "GAUSS_DECAY_TURB" :
        spec_ic = (kk) / ((1 + (kk**4)/6)) / (10**1.1)
        spec_kk = (kk_spec) / ((1 + (kk_spec**4)/6)) / (10**1.1)
    elif endas_u0 == "DECAY_TURB_ALT":
        spec_ic = (kk) / ((1 + (kk/6)**4)) / (10**1.1)
        spec_kk = (kk_spec) / ((1 + (kk_spec/6)**4)) / (10**1.1)
    ax1.plot(kk, spec_ic, '--')
    ax1.plot(kk, enrg_spect[0, 1:kmax] / 10**8, '.-')
    ax1.plot(kk, spec_kk[1:], '-.')
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # ax1.set_ylim(1e-5, 10)
    ax1.legend([r"Paper", r"IC", r"Py"])
    plt.savefig(output_dir + "/DecayTurb_EnergySpectrum_TEST_IC.png", bbox_inches = 'tight') 
    plt.close()

    for i in range(len(enrg_spect)):
        print("{} - {}".format(enrg_spect[i], enst_spect[i]))
    
    ## Enstrophy Spectrum Test Initial Condition
    fig = plt.figure(figsize = (16, 8))
    gs  = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    kmax = int(Nx/3 + 1)
    kk = np.arange(1, kmax)
    spec_ic = np.ones((kmax - 1, ))
    if endas_u0 == "DECAY_TURB_II":
        spec_ic = (kk**6) / ((1 + kk/60)**18) / (10**9.5)
    elif endas_u0 == "DECAY_TURB":
        spec_ic = (kk) / ((1 + (kk**4)/6)) / (10**1)
    elif endas_u0 == "DECAY_TURB_ALT":
        spec_ic = (kk**3) / ((1 + (kk/6)**4)) / (10**1.1)
    ax1.plot(kk, spec_ic, '--')
    ax1.plot(kk, enst_spect[0, 1:kmax] / 10**8, '.-')
    # spec, _  = enstrophy_spectrum(run_data.w_hat[0, :, :], run_data.kx, run_data.ky, sys_params.Nx, sys_params.Ny)
    # ax1.plot(kk, spec[1:kmax])
    # ax1.plot(kk, enst_spect[0, 1:kmax])
    ax1.set_xlabel(r"$|\mathbf{k}|$")
    ax1.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # ax1.set_ylim(1e-5, 10)
    ax1.legend([r"Paper", r"IC", r"Py", r"Post"])
    plt.savefig(output_dir + "/DecayTurb_EnstrophySpectrum_TEST_IC.png", bbox_inches = 'tight') 
    plt.close()