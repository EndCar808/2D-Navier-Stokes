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


def empty_real_array(shape, fft):

    if fft == "pyfftw":
        out = pyfftw.empty_aligned(shape, dtype='float64')
        out.flat[:] = 0.
        return out
    else:
        return np.zeros(shape, dtype='float64')


def empty_cmplx_array(shape, fft):

    if fft == "pyfftw":
        out = pyfftw.empty_aligned(shape, dtype='complex128')
        out.flat[:] = 0. + 0.*1.0j
        return out
    else:
        return np.zeros(shape, dtype='complex128')

def inverse_transform(w_h):

    num_saves = w_h.shape[0]

    w_py = empty_real_array((num_saves, Nx, Ny), "pyfft")

    for i in range(num_saves):
        w_py[i, :, :] = ifft(w_h[i, :, :])

    return w_py

def TestTaylorGreen(x, y, t, kappa = 1.0, nu = 1.0):

    ## Vorticty in real space
    w = 2. * kappa * np.cos(kappa * x) * np.cos(kappa * y[:, np.newaxis]) * np.exp(- 2 * kappa**2 * nu * t)

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
    Nyf = int(Ny / 2 + 1)
    t0 = 0.0
    T  = 1.0 
    date = "10-38-45"
    u0 = "TAYLOR_GREEN"

    #-------------------------------
    ## --------- Directories
    #-------------------------------
    input_folder = "RESULTS_NAVIER_RK4_N[{}][{}]_T[{}-{}]_[{}]_[TG_Test]".format(Nx, Ny, int(t0), int(T), date, u0)
    input_dir  = "/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/navierstokes_mpi/3D_Euler_Spectral/results/" + input_folder
    output_dir = input_dir + "/SNAPS"

    if os.path.isdir(output_dir) != True:
        print("Making folder: SNAPS")
        os.mkdir(output_dir)

    #------------------------------------
    # -------- Open File & Read In data
    #------------------------------------
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

    #------------------------------------
    # -------- Open File & Read In data
    #------------------------------------
    with h5py.File(input_dir + "/HDF_Global_REAL.h5", 'r') as file:
        
        ## Get the number of data saves
        num_saves = len([g for g in list(file.keys()) if 'Time' in g])

        ## Allocate arrays
        w    = np.zeros((num_saves, Nx, Ny))
        Real = 0
        # Read in the vorticity
        for i, group in enumerate(file.keys()):
            if "Time" in group:
                if 'W' in list(file[group].keys()):
                    w[i, :, :] = file[group]["W"][:, :]
                    Fourier = 1
              
    # Define inverse transform
    w_hat_py = empty_cmplx_array((Nx, Nyf), "pyfft")
    w_py     = empty_real_array((Nx, Ny), "pyfft")
    ifft     = fftw.FFTW(w_hat_py,  w_py, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))

    w_py = inverse_transform(w_hat)


    t = 10
    x, dx = np.linspace(0., 2. * np.pi, Nx, endpoint = False, retstep = True)
    y, dy = np.linspace(0., 2. * np.pi, Ny, endpoint = False, retstep = True)

    plt.figure()
    plt.imshow(w[t, :, :])
    plt.colorbar()
    plt.savefig(output_dir + "/w.png")

    err = np.absolute(w_py[t, :, :] - TestTaylorGreen(x, y, time[t]))
    plt.figure()
    plt.imshow(err)
    plt.colorbar()
    plt.savefig(output_dir + "/err.png")