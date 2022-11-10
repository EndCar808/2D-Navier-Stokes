#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: General python functions for analysing
#        Solver data


#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os
import re
from numba import njit
import pyfftw
from collections.abc import Iterable
from itertools import zip_longest
from subprocess import Popen, PIPE

#################################
## Colour Printing to Terminal ##
#################################
class tc:
    H         = '\033[95m'
    B         = '\033[94m'
    C         = '\033[96m'
    G         = '\033[92m'
    Y         = '\033[93m'
    R         = '\033[91m'
    Rst       = '\033[0m'
    Bold      = '\033[1m'
    Underline = '\033[4m'




#################################
##          MISC               ##
#################################
def import_bren_data(input_dir):

    ## Define a data class for the solver data
    class BrenData:

        """
        Class for the run data.
        """

        def __init__(self):
            self.ndata = 0
            self.kx = None

    data = BrenData()


    Global_f_file = input_dir + "HDF_Global_FOURIER.h5"

    ## Open file and read in the data
    with h5py.File(Global_f_file, 'r') as f:

        ## Get the number of timesteps
        data.ndata = len([g for g in f.keys() if 'Timestep' in g])

        ## Get wavenumbers and dimension
        data.kx = f["Timestep_0000"]['kx'][:]
        data.ky = f["Timestep_0000"]['ky'][:]   
        data.Nx = len(data.kx)
        data.Ny = len(data.kx)
        data.Nf = len(data.ky)

        ## Allocate memory for w_hat
        data.w_hat = np.ones((data.ndata, data.Nx, data.Nf)) * np.complex(0.0, 0.0)

        nn = 0

        # Read in the vorticity
        for group in f.keys():
            if "Timestep" in group:
                if 'W_hat' in list(f[group].keys()):
                    data.w_hat[nn, :, :] = f[group]["W_hat"][:, :]
                nn += 1


    Global_r_file = input_dir + "HDF_Global_REAL.h5"
    ## Open file and read in the data
    with h5py.File(Global_r_file, 'r') as f:

        ## Get the number of timesteps
        data.ndata = len([g for g in f.keys() if 'Timestep' in g])

        ## Get colocation points
        data.x = f["Timestep_0000"]['x'][:]
        data.y = f["Timestep_0000"]['y'][:]

        ## Allocate memory for W and U
        data.w = np.zeros((data.ndata, data.Nx, data.Ny)) 
        data.u = np.zeros((data.ndata, data.Nx, data.Ny, 2)) 

        nn = 0

        # Read in the vorticity
        for group in f.keys():
            if "Timestep" in group:
                if 'W' in list(f[group].keys()):
                    data.w[nn, :, :] = f[group]["W"][:, :]
                if 'U' in list(f[group].keys()):
                    data.u[nn, :, :] = f[group]["U"][:, :]
                nn += 1


    Local_file = input_dir + "HDF_Local_N[{}][{}].h5".format(data.Nx, data.Ny)
    ## Open file and read in the data
    with h5py.File(Local_file, 'r') as f:

        if 'MaxVorticity' in list(f.keys()):
            data.max_vort = f["MaxVorticity"][:]
        if 'TimeValues' in list(f.keys()):
            data.time = f["TimeValues"][:]

        if 'TotEnergy' in list(f.keys()):
            data.tot_enrg = f["TotEnergy"][:]
        if 'TotEnergyDissipation' in list(f.keys()):
            data.tot_enrg_diss = f["TotEnergyDissipation"][:]
        if 'TotEnstrophy' in list(f.keys()):
            data.tot_enst = f["TotEnstrophy"][:]
        if 'TotEnstrophyDissipation' in list(f.keys()):
            data.tot_enst_diss = f["TotEnstrophyDissipation"][:]
        if 'TotHelicity' in list(f.keys()):
            data.tot_hel = f["TotHelicity"][:]
        if 'TotVelDivergence' in list(f.keys()):
            data.tot_div = f["TotVelDivergence"][:]
        if 'TotVelForcing' in list(f.keys()):
            data.tot_vel_forc = f["TotVelForcing"][:]
        if 'TotVorForcing' in list(f.keys()):
            data.tot_vort_forc = f["TotVorForcing"][:]


    Spect_file = input_dir + "HDF_Energy_Spect.h5"
    ## Open file and read in the data
    with h5py.File(Spect_file, 'r') as f:

        ## Get the number of timesteps
        data.ndata_spec = len([g for g in f.keys() if 'Timestep' in g])

        ## Get the number of timesteps
        data.k_bins       = f["Timestep_0000"]['k_bins'][:]
        data.k_bins_3d    = f["Timestep_0000"]['k_bins_3D_x'][:]
        data.spec_size    = len(data.k_bins)
        data.spec_size_3d = len(data.k_bins_3d)

        ## Allocate memory for spec data
        data.spec_time      = np.zeros((data.ndata_spec, ))
        data.enrg_spec      = np.zeros((data.ndata_spec, data.spec_size)) 
        data.enrg_spec_3d_x = np.zeros((data.ndata_spec, data.spec_size_3d)) 
        data.enrg_spec_3d_y = np.zeros((data.ndata_spec, data.spec_size_3d))
        data.enst_spec      = np.zeros((data.ndata_spec, data.spec_size)) 
        data.enst_spec_3d_x = np.zeros((data.ndata_spec, data.spec_size_3d)) 
        data.enst_spec_3d_y = np.zeros((data.ndata_spec, data.spec_size_3d))

        nn = 0

        for group in f.keys():
            if "Timestep" in group:
                if 'EnstrophySpectrum' in list(f[group].keys()):
                    data.enst_spec[nn, :] = f[group]["EnstrophySpectrum"][:]
                if 'EnstrophySpectrum_3D_x' in list(f[group].keys()):
                    data.enst_spec_3d_x[nn, :] = f[group]["EnstrophySpectrum_3D_x"][:]
                if 'EnstrophySpectrum_3D_y' in list(f[group].keys()):
                    data.enst_spec_3d_y[nn, :] = f[group]["EnstrophySpectrum_3D_y"][:]
                if 'EnergySpectrum' in list(f[group].keys()):
                    data.enrg_spec[nn, :] = f[group]["EnergySpectrum"][:]
                if 'EnergySpectrum_3D_x' in list(f[group].keys()):
                    data.enrg_spec_3d_x[nn, :] = f[group]["EnergySpectrum_3D_x"][:]
                if 'EnergySpectrum_3D_y' in list(f[group].keys()):
                    data.enrg_spec_3d_y[nn, :] = f[group]["EnergySpectrum_3D_y"][:]
                data.spec_time[nn] = f[group].attrs["TimeValue"]

                nn +=1


    Flux_Spec_file = input_dir + "HDF_Flux_Spect.h5"
    ## Open file and read in the data
    with h5py.File(Flux_Spec_file, 'r') as f:

        ## Get the number of timesteps
        data.ndata_spec = len([g for g in f.keys() if 'Timestep' in g])

        ## Get the number of timesteps
        data.k_bins       = f["Timestep_0000"]['k_bins'][:]
        data.spec_size    = len(data.k_bins)

        ## Allocate memory for spec data
        data.spec_time      = np.zeros((data.ndata_spec, ))
        data.flux_u_spec      = np.zeros((data.ndata_spec, data.spec_size)) 
        data.flux_w_spec      = np.zeros((data.ndata_spec, data.spec_size)) 

        nn = 0

        for group in f.keys():
            if "Timestep" in group:
                if 'Flux_U_Spectrum' in list(f[group].keys()):
                    data.flux_u_spec[nn, :] = f[group]["Flux_U_Spectrum"][:]
                if 'Flux_W_Spectrum' in list(f[group].keys()):
                    data.flux_w_spec[nn, :] = f[group]["Flux_W_Spectrum"][:]
                data.spec_time[nn] = f[group].attrs["TimeValue"]

                nn +=1
    return data


def fft_ishift_freq(w_h, axes = None):

    """
    My version of fft.ifftshift
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

def run_commands_parallel(cmd_lsit, proc_limit):

    """
    Runs commands in parallel.

    Input Parameters:
        cmd_list    : list
                     - List of commands to run
        proc_limit  : int
                     - the number of processes to create to execute the commands in parallel
    """

    ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
    groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit

    ## Loop through grouped iterable
    for processes in zip_longest(*groups):
        for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
            ## Print command to screen
            print("\nExecuting the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)

            # Communicate with process to retrive output and error
            [run_CodeOutput, run_CodeErr] = proc.communicate()

            ## Print both to screen
            print(run_CodeOutput)
            print(run_CodeErr)

            ## Wait until all finished
            proc.wait()

#####################################
##       DATA FILE FUNCTIONS       ##
#####################################
def import_data(input_dir, post_file):

    """
    Calls all import functions to import solver and post processing data

    Input Parameters:
        input_dir : string
                    - Path to the input folder
        post_file : string
                    - Post processing data file name
    """

    # ## Initialize data objects
    # sys_vars  = None
    # run_data  = None
    # spec_data = None
    # sync_data = None
    # post_data = None
    
    # ## Import simulation details
    # sys_vars = sim_data(input_dir)

    # ## Import runtime solver data
    # run_data =  import_data(input_dir, sys_vars)

    # ## Import spectra data
    # spec_data = import_spectra_data(input_dir, sys_vars)

    # ## Import post processing data
    # post_file_path = input_dir + post_file
    # post_data = import_post_processing_data(post_file_path, sys_vars, 'file')

    # ## Import sync data
    # sync_data = import_sync_data(input_dir, sys_vars)

    return sys_vars, run_data, spec_data, sync_data, post_data

def sim_data(input_dir, method = "default"):

    """
    Reads in the system parameters from the simulation provided in SimulationDetails.txt.

    Input Parameters:
        input_dir : string
                    - If method == "defualt" is True then this will be the path to
                    the input folder. if not then this will be the input folder
        method    : string
                    - Determines whether the data is to be read in from a file or
                    from an input folder
    """

    ## Define a data class 
    class SimulationData:

        """
        Class for the system parameters.
        """

        ## Initialize class
        def __init__(self, Nx = 0, Ny = 0, Nk = 0, nu = 0.0, t0 = 0.0, T = 0.0, ndata = 0, u0 = "TG_VORT", cfl = 0.0, spec_size = 0, dt = 0., dx = 0., dy = 0.):
            self.Nx         = int(Nx)
            self.Ny         = int(Ny)
            self.Nk         = int(Nk)
            self.nu         = float(nu)
            self.hyper      = False
            self.hyper_pow  = 2.0
            self.alpha      = float(0.0)
            self.drag       = float(0.0)
            self.drag_pow   = -2.0
            self.t0         = float(t0)
            self.T          = float(T)
            self.ndata      = int(ndata)
            self.u0         = str(u0)
            self.cfl        = float(cfl)
            self.dt         = float(dt)
            self.dx         = float(dx)
            self.dy         = float(dy)
            self.PO         = False
            self.forc_type  = "NONE"
            self.forc_k     = 0.0
            self.forc_scale = 1.0
            self.po_slope   = np.sqrt(3.0)
            self.spec_size  = int(spec_size)


    ## Create instance of class
    data = SimulationData()

    if method == "default":
        ## Read in simulation data from file
        with open(input_dir + "SimulationDetails.txt") as f:

            ## Loop through lines and parse data
            for line in f.readlines():

                ## Parse viscosity
                if line.startswith('Viscosity'):
                    data.nu = float(line.split()[-1])
                ## Parse Hyperviscosity Flag
                if line.startswith('Hyperviscosity'):
                    if str(line.split()[-1]) == "YES":
                        data.hyper = True
                    else:
                        data.hyper = False
                ## Parse Hyper viscosity power
                if line.startswith('Hyperviscosity Power'):
                    data.hyper_pow = float(line.split()[-1])

                ## Parse drag
                if line.startswith('Ekman Alpha'):
                    data.alpha = float(line.split()[-1])
                ## Parse Ekman Flag
                if line.startswith('Ekman Drag'):
                    if str(line.split()[-1]) == "YES":
                        data.hyper = True
                    else:
                        data.hyper = False
                ## Parse Ekman power
                if line.startswith('Ekman Power'):
                    data.drag_pow = float(line.split()[-1])

                ## Parse number of collocation points
                if line.startswith('Collocation Points'):
                    data.Nx = int(line.split()[-2].lstrip('[').rstrip(','))
                    data.Ny = int(line.split()[-1].split(']')[0])

                ## Parse the number of Fourier modes
                if line.startswith('Fourier Modes'):
                    data.Nk = int(line.split()[-1].split(']')[0])

                ## Parse the start and end time
                if line.startswith('Time Range'):
                    data.t0 = float(line.split()[-3].lstrip('['))
                    data.T  = float(line.split()[-1].rstrip(']'))

                ## Parse number of saving steps
                if line.startswith('Total Saving Steps'):
                    data.ndata = int(line.split()[-1]) + 1 # plus 1 to include initial condition

                ## Parse the initial condition
                if line.startswith('Initial Conditions'):
                    data.u0 = str(line.split()[-1])

                ## Parse the Forcing details
                if line.startswith('Forcing Type'):
                    data.forc_type = str(line.split()[-1])
                if line.startswith('Forcing Wavenumer'):
                    data.forc_k = int(line.split()[-1])
                if line.startswith('Forcing Scaling'):
                    data.forc_scale = float(line.split()[-1])

                ## Parse the Phase Only amplitude slope
                if line.startswith('Phase Only Amplitude Slope'):
                    data.po_slope = float(line.split()[-1])

                ## Parse the timestep
                if line.startswith('Finishing Timestep'):
                    data.dt = float(line.split()[-1])

                ## Parse the spatial increment
                if line.startswith('Spatial Increment'):
                    data.dy = float(line.split()[-1].rstrip(']'))
                    data.dx = float(line.split()[-2].rstrip(',').lstrip('['))

                ## Parse model type
                if line.startswith('Model Type'):
                    if str(line.split()[-1]) == "PHASEONLY":
                        data.PO = True
                    else:
                        data.PO = False

            ## Get spectrum size
            data.spec_size = int(np.round(np.sqrt((data.Nx / 2)**2 + (data.Ny / 2)**2)) + 1)
            # data.spec_size = int(np.round(np.sqrt(data.Nx)))
    else:

        for term in input_dir.split('_'):

            ## Parse Viscosity
            if term.startswith("NU"):
                data.nu = float(term.split('[')[-1].rstrip(']'))

            ## Parse Number of collocation points & Fourier modes
            if term.startswith("N["):
                data.Nx = int(term.split('[')[-1].split(',')[0])
                data.Ny = int(term.split('[')[-1].split(',')[-1].rstrip(']'))
                data.Nk = int(data.Nx / 2 + 1)

            ## Parse Time range
            if term.startswith('T['):
                data.t0 = float(term.split(',')[0].lstrip('T['))
                data.dt = float(term.split(',')[1])
                data.T  = float(term.split(',')[-1].rstrip(']'))

            ## Parse CFL number
            if term.startswith('CFL'):
                data.cfl = float(term.split('[')[-1].rstrip(']'))

            ## Parse initial condition
            if term.startswith('u0'):
                data.u0 = str(term.split('[')[-1])
            if not term.startswith('u0') and term.endswith('].h5'):
                data.u0 = data.u0 + '_' + str(term.split(']')[0])

        ## Get the number of data saves
        with h5py.File(input_dir, 'r') as f:
            data.ndata = len([g for g in f.keys() if 'Iter' in g])

        ## Get spectrum size
        data.spec_size = int(np.round(np.sqrt((data.Nx / 2)**2 + (data.Ny / 2)**2)) + 1)
        # data.spec_size = int(np.round(np.sqrt(data.Nx)))


    return data



def import_data(input_file, sim_data, method = "default"):

    """
    Reads in run data from main HDF5 file.

    input_dir : string
                - If method == "defualt" is True then this will be the path to
               the input folder. if not then this will be the input folder
    method    : string
                - Determines whether the data is to be read in from a file or
               from an input folder
    sim_data  : class
                - object containing the simulation parameters
    """


    ## Define a data class for the solver data
    class SolverData:

        """
        Class for the run data.
        """

        def __init__(self):
            ## Allocate global arrays
            self.w         = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.u         = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny, 2))
            self.tg_soln   = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.u_hat     = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk, 2)) * np.complex(0.0, 0.0)
            self.w_hat     = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)
            self.nonlin    = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)
            self.time      = np.zeros((sim_data.ndata, ))
            ## Data indicators
            self.no_w     = False
            self.no_w_hat = False
            self.no_u     = False
            self.no_u_hat = False

    ## Create instance of data class
    data = SolverData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_file = input_file + "Main_HDF_Data.h5"
    else:
        in_file = input_file

    ## Open file and read in the data
    with h5py.File(in_file, 'r') as f:

        ## Initialize counter
        nn = 0

        # Read in the vorticity
        for group in f.keys():
            if "Iter" in group:
                if 'w' in list(f[group].keys()):
                    data.w[nn, :, :] = f[group]["w"][:, :]
                if 'w' not in list(f[group].keys()):
                    data.no_w = True
                if 'w_hat' in list(f[group].keys()):
                    data.w_hat[nn, :, :] = f[group]["w_hat"][:, :]
                if 'NonlinearTerm' in list(f[group].keys()):
                    data.nonlin[nn, :, :] = f[group]["NonlinearTerm"][:, :]
                if 'w_hat' not in list(f[group].keys()):
                    data.no_w_hat = True
                if 'u' in list(f[group].keys()):
                    data.u[nn, :, :] = f[group]["u"][:, :, :]
                if 'u' not in list(f[group].keys()):
                    data.no_u = True
                if 'u_hat' in list(f[group].keys()):
                    data.u_hat[nn, :, :] = f[group]["u_hat"][:, :, :]
                if 'u_hat' not in list(f[group].keys()):
                    data.no_u_hat = True
                if 'TGSoln' in list(f[group].keys()):
                    data.tg_soln[nn, :, :] = f[group]["TGSoln"][:, :]
                data.time[nn] = f[group].attrs["TimeValue"]
                nn += 1
            else:
                continue

        if data.no_w:
            print("\nPreparing real space vorticity from Solver Data...", end = " ")
            for i in range(sim_data.ndata):
                data.w[i, :, :] = np.fft.irfft2(data.w_hat[i, :, :])
            print("Finished!")

        ## Read in the space arrays
        if 'kx' in list(f.keys()):
            data.kx = f["kx"][:]
        if 'ky' in list(f.keys()):
            data.ky = f["ky"][:]
        if 'x' in list(f.keys()):
            data.x  = f["x"][:]
        if 'y' in list(f.keys()):
            data.y  = f["y"][:]

        ## Read system measures
        if 'TotalEnergy' in list(f.keys()):
            data.tot_enrg = f['TotalEnergy'][:]
        if 'TotalEnstrophy' in list(f.keys()):
            data.tot_enst = f['TotalEnstrophy'][:]
        if 'TotalPalinstrophy' in list(f.keys()):
            data.tot_palin = f['TotalPalinstrophy'][:]
        if 'TotalDivergence' in list(f.keys()):
            data.tot_div = f['TotalDivergence'][:]
        if 'TotalForcing' in list(f.keys()):
            data.tot_forc = f['TotalForcing'][:]
        if 'EnergyDissipation' in list(f.keys()):
            data.enrg_diss = f['EnergyDissipation'][:]
        if 'EnstrophyDissipation' in list(f.keys()):
            data.enst_diss = f['EnstrophyDissipation'][:]
        if 'EnergyDissSubset' in list(f.keys()):
            data.enrg_diss_sbst = f['EnergyDissSubset'][:]
        if 'EnstrophyDissSubset' in list(f.keys()):
            data.enst_diss_sbst = f['EnstrophyDissSubset'][:]
        if 'EnergyFluxSubset' in list(f.keys()):
            data.enrg_flux_sbst = f['EnergyFluxSubset'][:]
        if 'EnstrophyFluxSubset' in list(f.keys()):
            data.enst_flux_sbst = f['EnstrophyFluxSubset'][:]
        if 'MeanFlow_x' in list(f.keys()):
            data.mean_flow_x = f['MeanFlow_x'][:]
        if 'MeanFlow_y' in list(f.keys()):
            data.mean_flow_y = f['MeanFlow_y'][:]

    return data


def import_sys_msr(input_file, sim_data, method = "default"):

    """
    Reads in run data from main HDF5 file.

    input_dir : string
                - If method == "defualt" is True then this will be the path to
               the input folder. if not then this will be the input folder
    method    : string
                - Determines whether the data is to be read in from a file or
               from an input folder
    sim_data  : class
                - object containing the simulation parameters
    """


    ## Define a data class for the solver data
    class SolverData:

        """
        Class for the run data.
        """

        def __init__(self):
            self.time = None

    ## Create instance of data class
    data = SolverData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_file = input_file + "SystemMeasures_HDF_Data.h5"
    else:
        in_file = input_file

    ## Open file and read in the data
    with h5py.File(in_file, 'r') as f:

        ## Read in the space arrays
        if 'kx' in list(f.keys()):
            data.kx = f["kx"][:]
        if 'ky' in list(f.keys()):
            data.ky = f["ky"][:]
        if 'x' in list(f.keys()):
            data.x  = f["x"][:]
        if 'y' in list(f.keys()):
            data.y  = f["y"][:]

        ## Read system measures
        if 'TotalEnergy' in list(f.keys()):
            data.tot_enrg = f['TotalEnergy'][:]
        if 'TotalEnstrophy' in list(f.keys()):
            data.tot_enst = f['TotalEnstrophy'][:]
        if 'TotalPalinstrophy' in list(f.keys()):
            data.tot_palin = f['TotalPalinstrophy'][:]
        if 'TotalDivergence' in list(f.keys()):
            data.tot_div = f['TotalDivergence'][:]
        if 'TotalForcing' in list(f.keys()):
            data.tot_forc = f['TotalForcing'][:]
        if 'EnergyDissipation' in list(f.keys()):
            data.enrg_diss = f['EnergyDissipation'][:]
        if 'EnstrophyDissipation' in list(f.keys()):
            data.enst_diss = f['EnstrophyDissipation'][:]
        if 'EnergyDissSubset' in list(f.keys()):
            data.enrg_diss_sbst = f['EnergyDissSubset'][:]
        if 'EnstrophyDissSubset' in list(f.keys()):
            data.enst_diss_sbst = f['EnstrophyDissSubset'][:]
        if 'EnergyFluxSubset' in list(f.keys()):
            data.enrg_flux_sbst = f['EnergyFluxSubset'][:]
        if 'EnstrophyFluxSubset' in list(f.keys()):
            data.enst_flux_sbst = f['EnstrophyFluxSubset'][:]
        if 'MeanFlow_x' in list(f.keys()):
            data.mean_flow_x = f['MeanFlow_x'][:]
        if 'MeanFlow_y' in list(f.keys()):
            data.mean_flow_y = f['MeanFlow_y'][:]
        if 'Time' in list(f.keys()):
            data.time = f['Time'][:]

        ## Get inv wavenumbers
        if 'kx' in list(f.keys()) and 'ky' in list(f.keys()):
            data.k2 = data.kx**2 + data.ky[:, np.newaxis]**2
            index   = data.k2 != 0.0
            data.k2Inv = np.zeros((sim_data.Nx, sim_data.Nk))
            data.k2Inv[index] = 1. / data.k2[index]
        
    return data

def import_spectra_data(input_file, sim_data, method = "default"):

    """
    Reads in run data from main HDF5 file.

    input_dir : string
                - If method == "defualt" is True then this will be the path to
               the input folder. if not then this will be the input folder
    method    : string
                - Determines whether the data is to be read in from a file or
                from an input folder
    sim_data  : class
                - object containing the simulation parameters
    """

    ## Define a data class for the solver data
    class SpectraData:

        """
        Class for the run data.
        """

        def __init__(self):
            ## Allocate spectra aarrays
            self.enrg_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enst_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            ## Allocate flux spectra arrays
            self.d_enrg_dt_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enrg_flux_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enrg_diss_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.d_enst_dt_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enst_flux_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enst_diss_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))


    ## Create instance of data class
    data = SpectraData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_file = input_file + "Spectra_HDF_Data.h5"
    else:
        in_file = input_file

    ## Open file and read in the data
    with h5py.File(in_file, 'r') as f:

        ## Initialze counter
        nn = 0

        # Read in the spectra
        for group in f.keys():
            if "Iter" in group:
                if 'EnergySpectrum' in list(f[group].keys()):
                    data.enrg_spectrum[nn, :] = f[group]["EnergySpectrum"][:]
                if 'EnstrophySpectrum' in list(f[group].keys()):
                    data.enst_spectrum[nn, :] = f[group]["EnstrophySpectrum"][:]
                if 'EnergyFluxSpectrum' in list(f[group].keys()):
                    data.enrg_flux_spectrum[nn, :] = f[group]["EnergyFluxSpectrum"][:]
                if 'EnergyDissipationSpectrum' in list(f[group].keys()):
                    data.enrg_diss_spectrum[nn, :] = f[group]["EnergyDissipationSpectrum"][:]
                if 'TimeDerivativeEnergySpectrum' in list(f[group].keys()):
                    data.d_enrg_dt_spectrum[nn, :] = f[group]["TimeDerivativeEnergySpectrum"][:]
                if 'EnstrophyFluxSpectrum' in list(f[group].keys()):
                    data.enst_flux_spectrum[nn, :] = f[group]["EnstrophyFluxSpectrum"][:]
                if 'EnstrophyDissipationSpectrum' in list(f[group].keys()):
                    data.enst_diss_spectrum[nn, :] = f[group]["EnstrophyDissipationSpectrum"][:]
                if 'TimeDerivativeEnstrophySpectrum' in list(f[group].keys()):
                    data.d_enst_dt_spectrum[nn, :] = f[group]["TimeDerivativeEnstrophySpectrum"][:]
                nn += 1
            else:
                continue

    return data

def import_sync_data(input_file, sim_data, method = "default"):

    """
    Reads in phase sync data from sync HDF5 file.

    input_dir : string
                - If method == "defualt" is True then this will be the path to
               the input folder. if not then this will be the input folder
    method    : string
                - Determines whether the data is to be read in from a file or
                from an input folder
    sim_data  : class
                - object containing the simulation parameters
    """

    ## Define a data class for the solver data
    class SyncData:

        """
        Class for the run data.
        """

        def __init__(self):
            ## Allocate sync arrays
            self.R_k            = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.theta_k        = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.normed_R_k     = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.normed_theta_k = np.zeros((sim_data.ndata, sim_data.spec_size))
                        

    ## Create instance of data class
    data = SyncData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_file = input_file + "PhaseSync_HDF_Data.h5"
    else:
        in_file = input_file

    ## Open file and read in the data
    with h5py.File(in_file, 'r') as f:

        ## Initialze counter
        nn = 0

        # Read in the spectra
        for group in f.keys():
            if "Iter" in group:
                if 'PhaseOrder_R_k' in list(f[group].keys()):
                    data.R_k[nn, :] = f[group]["PhaseOrder_R_k"][:]
                if 'PhaseOrder_Theta_k' in list(f[group].keys()):
                    data.theta_k[nn, :] = f[group]["PhaseOrder_Theta_k"][:]
                if 'NormedPhaseOrder_R_k' in list(f[group].keys()):
                    data.normed_R_k[nn, :] = f[group]["NormedPhaseOrder_R_k"][:]
                if 'NormedPhaseOrder_Theta_k' in list(f[group].keys()):
                    data.normed_theta_k[nn, :] = f[group]["NormedPhaseOrder_Theta_k"][:]
                nn += 1
            else:
                continue

    return data

def import_post_processing_data(input_file, sim_data, method = "default"):

    """
    Reads in post processing data from HDF5 file.

    input_dir : string
                - If method == "defualt" is True then this will be the path to
               the input folder. if not then this will be the input folder
    method    : string
                - Determines whether the data is to be read in from a file or
               from an input folder
    sim_data  : class
                - object containing the simulation parameters
    """

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_f = input_file + "PostProcessing_HDF_Data.h5"
    else:
        in_f = input_file

    ## Define a data class for the solver data
    class PostProcessData:

        """
        Class for the run data.
        """

        def __init__(self, in_file = in_f):
            ## Get non time dependent datasets
            with h5py.File(in_file, 'r') as f:
                ## Get the wavevectors
                if 'kx' in list(f.keys()):
                    self.kx = f["kx"][:]
                if 'ky' in list(f.keys()):
                    self.ky = f["ky"][:]
                ## Get the collocation points
                if 'x' in list(f.keys()):
                    self.x = f["x"][:]
                if 'y' in list(f.keys()):
                    self.y = f["y"][:]
                ## Get the number of sectors
                if 'SectorAngles_k3' in list(f.keys()):
                    self.theta_k3 = f["SectorAngles_k3"][:]
                    self.num_sect = self.theta_k3.shape[0]
                if 'SectorAngles_k3' not in list(f.keys()):
                    self.num_sect = int(in_f.split('_')[-3].split(',')[0].lstrip('SECTORS['))
                if 'SectorAngles_k1' not in list(f.keys()):
                    self.num_k1_sects = int(in_f.split('_')[-3].split(',')[-1].rstrip(']'))
                if 'SectorAngles_k1' in list(f.keys()):
                    self.theta_k1     = f["SectorAngles_k1"][:]
                    self.num_k1_sects = self.theta_k1.shape[0]
                ## Get the number of triads per sector
                if 'NumTriadsPerSector' in list(f.keys()):
                    self.num_triads = f["NumTriadsPerSector"][:, :]
                ## Get the number of triads per sector
                if 'NumTriadsTest' in list(f.keys()):
                    self.num_triads_test = f["NumTriadsTest"][:]
                 ## Get the wavevector data for the phase syc
                if 'WavevectorDataTest' in list(f.keys()):
                    self.wave_vec_data_test = f["WavevectorDataTest"][:]
                ## Get the number of triads per sector
                if 'NumTriadsPerSector_1D' in list(f.keys()):
                    self.num_triads_1d = f["NumTriadsPerSector_1D"][:, :]
                if 'NumTriadsPerSector_2D' in list(f.keys()):
                    self.num_triads_2d = f["NumTriadsPerSector_2D"][:, :, :]
                ## Get the enstrophy flux out of the set C
                if 'EnstrophyFluxC' in list(f.keys()):
                    self.enst_flux_C = f["EnstrophyFluxC"][:]
                if 'EnstrophyDissC' in list(f.keys()):
                    self.enst_diss_C = f["EnstrophyDissC"][:]
                ## Get the energy flux out of the set C
                if 'EnergyFluxC' in list(f.keys()):
                    self.enrg_flux_C = f["EnergyFluxC"][:]
                if 'EnergyDissC' in list(f.keys()):
                    self.enrg_diss_C = f["EnergyDissC"][:]

                ############# STATS DATA
                ## Get the Velocity increment histogram data
                if 'LongitudinalVelIncrements_BinRanges' in list(f.keys()):
                    self.vel_long_incr_ranges = f["LongitudinalVelIncrements_BinRanges"][:, :]
                if 'LongitudinalVelIncrements_BinCounts' in list(f.keys()):
                    self.vel_long_incr_counts = f["LongitudinalVelIncrements_BinCounts"][:, :]
                if 'TransverseVelIncrements_BinRanges' in list(f.keys()):
                    self.vel_trans_incr_ranges = f["TransverseVelIncrements_BinRanges"][:, :]
                if 'TransverseVelIncrements_BinCounts' in list(f.keys()):
                    self.vel_trans_incr_counts = f["TransverseVelIncrements_BinCounts"][:, :]
                if 'VelocityIncrementStats' in list(f.keys()):
                    self.vel_incr_stat = f["VelocityIncrementStats"][:, :, :]
                ## Get the Vorticit increment histogram data
                if 'LongitudinalVortIncrements_BinRanges' in list(f.keys()):
                    self.vort_long_incr_ranges = f["LongitudinalVortIncrements_BinRanges"][:, :]
                if 'LongitudinalVortIncrements_BinCounts' in list(f.keys()):
                    self.vort_long_incr_counts = f["LongitudinalVortIncrements_BinCounts"][:, :]
                if 'TransverseVortIncrements_BinRanges' in list(f.keys()):
                    self.vort_trans_incr_ranges = f["TransverseVortIncrements_BinRanges"][:, :]
                if 'TransverseVortIncrements_BinCounts' in list(f.keys()):
                    self.vort_trans_incr_counts = f["TransverseVortIncrements_BinCounts"][:, :]
                if 'VorticityIncrementStats' in list(f.keys()):
                    self.vort_incr_stat = f["VorticityIncrementStats"][:, :, :]
                ## Get the Velocity structure function data
                if 'VelocityLongitudinalStructureFunctions' in list(f.keys()):
                    self.vel_long_str_func = f["VelocityLongitudinalStructureFunctions"][:, :]
                if 'AbsoluteVelocityLongitudinalStructureFunctions' in list(f.keys()):
                    self.vel_long_str_func_abs = f["AbsoluteVelocityLongitudinalStructureFunctions"][:, :]
                if 'VelocityTransverseStructureFunctions' in list(f.keys()):
                    self.vel_trans_str_func = f["VelocityTransverseStructureFunctions"][:, :]
                if 'AbsoluteVelocityTransverseStructureFunctions' in list(f.keys()):
                    self.vel_trans_str_func_abs = f["AbsoluteVelocityTransverseStructureFunctions"][:, :]
                ## Get the Velocity structure function data
                if 'VorticityLongitudinalStructureFunctions' in list(f.keys()):
                    self.vort_long_str_func = f["VorticityLongitudinalStructureFunctions"][:, :]
                if 'AbsoluteVorticityLongitudinalStructureFunctions' in list(f.keys()):
                    self.vort_long_str_func_abs = f["AbsoluteVorticityLongitudinalStructureFunctions"][:, :]
                if 'VorticityTransverseStructureFunctions' in list(f.keys()):
                    self.vort_trans_str_func = f["VorticityTransverseStructureFunctions"][:, :]
                if 'AbsoluteVorticityTransverseStructureFunctions' in list(f.keys()):
                    self.vort_trans_str_func_abs = f["AbsoluteVorticityTransverseStructureFunctions"][:, :]
                ## Mixed structure funcitons
                if 'MixedVelocityStructureFunctions' in list(f.keys()):
                    self.mxd_vel_str_func = f["MixedVelocityStructureFunctions"][:]
                if 'MixedVorticityStructureFunctions' in list(f.keys()):
                    self.mxd_vort_str_func = f["MixedVorticityStructureFunctions"][:]
                ## Get the Velocity gradient histograms
                if 'VelocityGradient_x_BinRanges' in list(f.keys()):
                    self.grad_u_x_ranges = f["VelocityGradient_x_BinRanges"][:]
                if 'VelocityGradient_x_BinCounts' in list(f.keys()):
                    self.grad_u_x_counts = f["VelocityGradient_x_BinCounts"][:]
                if 'VelocityGradient_y_BinRanges' in list(f.keys()):
                    self.grad_u_y_ranges = f["VelocityGradient_y_BinRanges"][:]
                if 'VelocityGradient_y_BinCounts' in list(f.keys()):
                    self.grad_u_y_counts = f["VelocityGradient_y_BinCounts"][:]
                if 'VelocityGradientStats' in list(f.keys()):
                    self.grad_u_stats = f["VelocityGradientStats"][:, :]
                ## Get the Vorticity gradient histograms
                if 'VorticityGradient_x_BinRanges' in list(f.keys()):
                    self.grad_w_x_ranges = f["VorticityGradient_x_BinRanges"][:]
                if 'VorticityGradient_x_BinCounts' in list(f.keys()):
                    self.grad_w_x_counts = f["VorticityGradient_x_BinCounts"][:]
                if 'VorticityGradient_y_BinRanges' in list(f.keys()):
                    self.grad_w_y_ranges = f["VorticityGradient_y_BinRanges"][:]
                if 'VorticityGradient_y_BinCounts' in list(f.keys()):
                    self.grad_w_y_counts = f["VorticityGradient_y_BinCounts"][:]
                if 'VorticityGradientStats' in list(f.keys()):
                    self.grad_w_stats = f["VorticityGradientStats"][:, :]

            ## Get the max wavenumber
            self.kmax      = int((sim_data.Nx / 3))
            self.kmax_frac = float(in_file.split('_')[-2].split("[")[-1].split("]")[0])
            self.kmax_C    = int(self.kmax * self.kmax_frac)

            ## Read in Wavevector Data
            pre_data_path = re.search(r"[^Data]*", in_f).group()
            wave_data_file = pre_data_path + 'Data/PostProcess/PhaseSync/Wavevector_Data_N[{},{}]_SECTORS[{},{}]_KFRAC[{:0.2f}].h5'.format(sim_data.Nx, sim_data.Ny, int(self.num_sect), int(self.num_k1_sects), self.kmax_frac)
            if os.path.isfile(wave_data_file):
                with h5py.File(wave_data_file) as f:
                    self.num_wv = f["NumWavevectors"][:, :]
                    self.wv = np.zeros((self.num_sect, self.num_k1_sects, 16, np.amax(self.num_wv)))
                    for a in range(self.num_sect):
                        for l in range(self.num_k1_sects):
                            tmp_arr = f["WVData_Sector_{}_{}".format(a, l)][:, :]
                            for k in range(16):
                                for n in range(self.num_wv[a, l]):
                                    self.wv[a, l, k, n] = tmp_arr[k, n]

            ## Data indicators
            self.no_w     = False
            self.no_w_hat = False
            self.no_u     = False
            self.no_u_hat = False
            
            ## Set the number of Triad Types
            NUM_TRIAD_TYPES = 7

            # sim_data.spec_size = 91

            ## Allocate spectra aarrays
            self.phases        = np.zeros((sim_data.ndata, int(2 * self.kmax + 1), int(2 * self.kmax + 1)))
            self.enrg_spectrum = np.zeros((sim_data.ndata, int(2 * self.kmax + 1), int(2 * self.kmax + 1)))
            self.enst_spectrum = np.zeros((sim_data.ndata, int(2 * self.kmax + 1), int(2 * self.kmax + 1)))
            ## Allocate the increment arrays
            self.u_pdf_ranges = np.zeros((sim_data.ndata, 1001))
            self.u_pdf_counts = np.zeros((sim_data.ndata, 1000))
            self.w_pdf_ranges = np.zeros((sim_data.ndata, 1001))
            self.w_pdf_counts = np.zeros((sim_data.ndata, 1000))
            ## Allocate spectra arrays
            self.enst_spectrum_1d = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enrg_spectrum_1d = np.zeros((sim_data.ndata, sim_data.spec_size))
            # Enstrophy Dissipation Field
            self.enst_diss_field = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            ## Allocate solver data arrays
            self.w      = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.w_hat  = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)
            self.u_hat  = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk, 2)) * np.complex(0.0, 0.0)
            self.u      = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny, 2))
            self.nonlin = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)            
            ## Enstrophy Flux Spectrum
            self.enst_flux_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enst_diss_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.d_enst_dt_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            ## Energy Flux Spectrum
            self.enrg_flux_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enrg_diss_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.d_enrg_dt_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            ## Enstorphy Flux and Diss from C_theta
            self.enst_flux_C_theta = np.zeros((sim_data.ndata, self.num_sect))
            self.enst_diss_C_theta = np.zeros((sim_data.ndata, self.num_sect))
            ## Collective Phase order parameter for C_theta
            self.phase_order_C_theta                    = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads             = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads_1d          = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads_2d          = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads_unidirec    = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads_unidirec_1d = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads_unidirec_2d = np.ones((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects)) * np.complex(0.0, 0.0)
            ## Enstrophy Flux Per Sector
            self.enst_flux_per_sec    = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.enst_flux_per_sec_1d = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.enst_flux_per_sec_2d = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects))
            ## Phase Sync arrays
            self.phase_R      = np.zeros((sim_data.ndata, self.num_sect))
            self.phase_Phi    = np.zeros((sim_data.ndata, self.num_sect))
            self.triad_R      = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_Phi    = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_R_1d   = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_Phi_1d = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_R_2d   = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects))
            self.triad_Phi_2d = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects))
            ## Phase Sync Stats
            self.phase_sector_counts = np.zeros((sim_data.ndata, self.num_sect, 200))
            self.phase_sector_ranges = np.zeros((sim_data.ndata, self.num_sect, 201))
            self.triad_sector_counts = np.zeros((sim_data.ndata, self.num_sect, 200, NUM_TRIAD_TYPES))
            self.triad_sector_ranges = np.zeros((sim_data.ndata, self.num_sect, 201, NUM_TRIAD_TYPES))
            self.triad_sector_wghtd_counts = np.zeros((sim_data.ndata, self.num_sect, 200, NUM_TRIAD_TYPES))
            self.triad_sector_wghtd_ranges = np.zeros((sim_data.ndata, self.num_sect, 201, NUM_TRIAD_TYPES))

            ## Test phase sync data
            self.enst_flux_test = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES))
            self.triad_R_test   = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES))
            self.triad_Phi_test = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES))

    ## Create instance of data class
    data = PostProcessData()

    ## Open file and read in the data
    with h5py.File(in_f, 'r') as f:

        ## Initialize counter
        nn = 0

        # Read in the spectra
        for group in f.keys():
            if "Snap" in group:
                if 'FullFieldPhases' in list(f[group].keys()):
                    data.phases[nn, :] = f[group]["FullFieldPhases"][:, :]
                if 'FullFieldEnstrophySpectrum' in list(f[group].keys()):
                    data.enst_spectrum[nn, :] = f[group]["FullFieldEnstrophySpectrum"][:, :]
                if 'FullFieldEnergySpectrum' in list(f[group].keys()):
                    data.enrg_spectrum[nn, :] = f[group]["FullFieldEnergySpectrum"][:, :]
                if 'VelocityPDFCounts' in list(f[group].keys()):
                    data.u_pdf_counts[nn, :] = f[group]["VelocityPDFCounts"][:]
                if 'VelocityPDFRanges' in list(f[group].keys()):
                    data.u_pdf_ranges[nn, :] = f[group]["VelocityPDFRanges"][:]
                if 'VorticityPDFCounts' in list(f[group].keys()):
                    data.w_pdf_counts[nn, :] = f[group]["VorticityPDFCounts"][:]
                if 'VorticityPDFRanges' in list(f[group].keys()):
                    data.w_pdf_ranges[nn, :] = f[group]["VorticityPDFRanges"][:]
                if 'EnstrophySpectrum' in list(f[group].keys()):
                    data.enst_spectrum_1d[nn, :] = f[group]["EnstrophySpectrum"][:]
                if 'EnergySpectrum' in list(f[group].keys()):
                    data.enrg_spectrum_1d[nn, :] = f[group]["EnergySpectrum"][:]
                if 'PhaseSync' in list(f[group].keys()):
                    data.phase_R[nn, :] = f[group]["PhaseSync"][:]
                if 'AverageAngle' in list(f[group].keys()):
                    data.phase_Phi[nn, :] = f[group]["AverageAngle"][:]
                if 'TriadPhaseSync' in list(f[group].keys()):
                    data.triad_R[nn, :, :] = f[group]["TriadPhaseSync"][:, :]
                if 'TriadAverageAngle' in list(f[group].keys()):
                    data.triad_Phi[nn, :, :] = f[group]["TriadAverageAngle"][:, :]
                if 'TriadPhaseSync_1D' in list(f[group].keys()):
                    data.triad_R_1d[nn, :, :] = f[group]["TriadPhaseSync_1D"][:, :]
                if 'TriadAverageAngle_1D' in list(f[group].keys()):
                    data.triad_Phi_1d[nn, :, :] = f[group]["TriadAverageAngle_1D"][:, :]
                if 'TriadPhaseSync_2D' in list(f[group].keys()):
                    data.triad_R_2d[nn, :, :] = f[group]["TriadPhaseSync_2D"][:, :]
                if 'TriadAverageAngle_2D' in list(f[group].keys()):
                    data.triad_Phi_2d[nn, :, :] = f[group]["TriadAverageAngle_2D"][:, :]
                if 'EnstrophyFlux_C_theta' in list(f[group].keys()):
                    data.enst_flux_C_theta[nn, :] = f[group]["EnstrophyFlux_C_theta"][:]
                if 'EnstrophyDiss_C_theta' in list(f[group].keys()):
                    data.enst_diss_C_theta[nn, :] = f[group]["EnstrophyDiss_C_theta"][:]
                if 'CollectivePhaseOrder_C_theta' in list(f[group].keys()):
                    data.phase_order_C_theta[nn, :] = f[group]["CollectivePhaseOrder_C_theta"][:]
                if 'CollectivePhaseOrder_C_theta_Triads' in list(f[group].keys()):
                    data.phase_order_C_theta_triads[nn, :, :] = f[group]["CollectivePhaseOrder_C_theta_Triads"][:, :]
                if 'CollectivePhaseOrder_C_theta_Triads_1D' in list(f[group].keys()):
                    data.phase_order_C_theta_triads_1d[nn, :, :] = f[group]["CollectivePhaseOrder_C_theta_Triads_1D"][:, :]
                if 'CollectivePhaseOrder_C_theta_Triads_2D' in list(f[group].keys()):
                    data.phase_order_C_theta_triads_2d[nn, :, :, :] = f[group]["CollectivePhaseOrder_C_theta_Triads_2D"][:, :, :]
                if 'CollectivePhaseOrder_C_theta' in list(f[group].keys()):
                    data.phase_order_C_theta[nn, :] = f[group]["CollectivePhaseOrder_C_theta"][:]
                if 'CollectivePhaseOrder_C_theta_Triads_Unidirectional' in list(f[group].keys()):
                    data.phase_order_C_theta_triads_unidirec[nn, :, :] = f[group]["CollectivePhaseOrder_C_theta_Triads_Unidirectional"][:, :]
                if 'CollectivePhaseOrder_C_theta_Triads_1D_Unidirectional' in list(f[group].keys()):
                    data.phase_order_C_theta_triads_unidirec_1d[nn, :, :] = f[group]["CollectivePhaseOrder_C_theta_Triads_1D_Unidirectional"][:, :]
                if 'CollectivePhaseOrder_C_theta_Triads_2D_Unidirectional' in list(f[group].keys()):
                    data.phase_order_C_theta_triads_unidirec_2d[nn, :, :, :] = f[group]["CollectivePhaseOrder_C_theta_Triads_2D_Unidirectional"][:, :, :]
                if 'w_hat' in list(f[group].keys()):
                    data.w_hat[nn, :, :] = f[group]["w_hat"][:, :]
                if 'NonlinearTerm' in list(f[group].keys()):
                    data.nonlin[nn, :, :] = f[group]["NonlinearTerm"][:, :]
                if 'u_hat' in list(f[group].keys()):
                    data.u_hat[nn, :, :, :] = f[group]["u_hat"][:, :, :]
                if 'w' in list(f[group].keys()):
                    data.w[nn, :, :] = f[group]["w"][:, :]
                if 'u' in list(f[group].keys()):
                    data.u[nn, :, :, :] = f[group]["u"][:, :, :]
                if 'w' not in list(f[group].keys()):
                    data.no_w = True
                if 'EnstrophyDissipationField'  in list(f[group].keys()):
                    data.enst_diss_field[nn, :, :] = np.fft.irfft2(f[group]["EnstrophyDissipationField"][:, :])
                if 'EnstrophyTimeDerivativeSpectrum' in list(f[group].keys()):
                    data.d_enst_dt_spec[nn, :] = f[group]["EnstrophyTimeDerivativeSpectrum"][:]
                if 'EnstrophyFluxSpectrum' in list(f[group].keys()):
                    data.enst_flux_spec[nn, :] = f[group]["EnstrophyFluxSpectrum"][:]
                if 'EnstrophyDissSpectrum' in list(f[group].keys()):
                    data.enst_diss_spec[nn, :] = f[group]["EnstrophyDissSpectrum"][:]
                if 'EnergyFluxSpectrum' in list(f[group].keys()):
                    data.enrg_flux_spec[nn, :] = f[group]["EnergyFluxSpectrum"][:]
                if 'EnergyTimeDerivativeSpectrum' in list(f[group].keys()):
                    data.d_enrg_dt_spec[nn, :] = f[group]["EnergyTimeDerivativeSpectrum"][:]
                if 'EnergyDissSpectrum' in list(f[group].keys()):
                    data.enrg_diss_spec[nn, :] = f[group]["EnergyDissSpectrum"][:]
                if 'EnstrophyFluxPerSector' in list(f[group].keys()):
                    data.enst_flux_per_sec[nn, :, :] = f[group]["EnstrophyFluxPerSector"][:, :]
                if 'EnstrophyFluxPerSector_1D' in list(f[group].keys()):
                    data.enst_flux_per_sec_1d[nn, :, :] = f[group]["EnstrophyFluxPerSector_1D"][:, :]
                if 'EnstrophyFluxPerSector_2D' in list(f[group].keys()):
                    data.enst_flux_per_sec_2d[nn, :, :, :] = f[group]["EnstrophyFluxPerSector_2D"][:, :, :]
                if 'SectorPhasePDF_InTime_Counts' in list(f[group].keys()):
                    data.phase_sector_counts[nn, :, :] = f[group]["SectorPhasePDF_InTime_Counts"][:, :]
                if 'SectorPhasePDF_InTime_Ranges' in list(f[group].keys()):
                    data.phase_sector_ranges[nn, :, :] = f[group]["SectorPhasePDF_InTime_Ranges"][:, :]
                if 'SectorTriadPhasePDF_InTime_Counts' in list(f[group].keys()):
                    data.triad_sector_counts[nn, :, :, :] = f[group]["SectorTriadPhasePDF_InTime_Counts"][:, :, :]
                if 'SectorTriadPhasePDF_InTime_Ranges' in list(f[group].keys()):
                    data.triad_sector_ranges[nn, :, :, :] = f[group]["SectorTriadPhasePDF_InTime_Ranges"][:, :, :]
                if 'SectorTriadPhaseWeightedPDF_InTime_Counts' in list(f[group].keys()):
                    data.triad_sector_wghtd_counts[nn, :, :, :] = f[group]["SectorTriadPhaseWeightedPDF_InTime_Counts"][:, :, :]
                if 'SectorTriadPhaseWeightedPDF_InTime_Ranges' in list(f[group].keys()):
                    data.triad_sector_wghtd_ranges[nn, :, :, :] = f[group]["SectorTriadPhaseWeightedPDF_InTime_Ranges"][:, :, :]
                if 'TriadPhaseSyncTest' in list(f[group].keys()):
                    data.triad_R_test[nn, :] = f[group]["TriadPhaseSyncTest"][:]
                if 'TriadAverageAngleTest' in list(f[group].keys()):
                    data.triad_Phi_test[nn, :] = f[group]["TriadAverageAngleTest"][:]
                if 'EnstrophyFluxTest' in list(f[group].keys()):
                    data.enst_flux_test[nn, :] = f[group]["EnstrophyFluxTest"][:]
                nn += 1
            else:
                continue

        if data.no_w:
            print("\nPreparing real space vorticity from post data...", end = " ")
            for i in range(sim_data.ndata):
                data.w[i, :, :] = np.fft.irfft2(data.w_hat[i, :, :])
            print("Finished!")

    return data


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


###################################
##      FOURIER TRANSFORMS       ##
###################################
def fftw_init_2D(Nx, Ny):

    ## Store transforms in cache for quick look up
    pyfftw.interfaces.cache.enable()

    ## Create class for forward transform
    class fftw_2D:

        """
        Class for forward transform.
        """

        def __init__(self, nx = Nx, ny = Ny):
            # dummy variables forward transofmr
            self._in       = empty_real_array(nx, ny)
            self.dummy_in  = self._in
            self._out      = empty_complex_array(nx, int(ny / 2 + 1))
            self.dummy_out = self._out

            # dummy ffts 
            self.fft = pyfftw.FFTW(self._in, self._out, threads = 1, direction = 'FFTW_FORWARD', axes = (-2,-1))

    ## Create class for backward transform
    class ifftw_2D:

        """
        Class for backward transform.
        """

        def __init__(self, nx = Nx, ny = Ny):
            # dummy variables backward transform
            self._in       = empty_complex_array(nx, int(ny / 2 + 1))
            self.dummy_in  = self._in
            self._out      = empty_real_array(nx, ny)
            self.dummy_out = self._out

            ## dummy ifft
            self.ifft = pyfftw.FFTW(self._in, self._out, threads = 1, direction = 'FFTW_BACKWARD', axes = (-2,-1))

    ## Create instance of these classes
    fft  = fftw_2D()
    ifft = ifftw_2D()

    return fft, ifft

def empty_real_array(nx, ny):

    """
    Allocate a space-grid-sized variable for use with fftw transformations.
    """

    ## Get the shape of the array
    shape = (nx, ny)

    ## Allocate array and initialize
    out        = pyfftw.empty_aligned(shape, dtype = "float64")
    out.flat[:] = 0.

    return out

def empty_complex_array(nx, nk):

    """
    Allocate a Fourier-grid-sized variable for use with fftw transformations.
    """
    ## Get the shape of the array
    shape = (nx, nk)

    ## Allocate and initialize array
    out         = pyfftw.empty_aligned(shape, dtype = "complex128")
    out.flat[:] = 0. + 0. * 1.j

    return out

def fft(fftw, v):

    """"
    Generic FFT function for real grid-sized variables.
    """

    ## Get copy of input array
    v_view = v

    # copy input into memory view
    fftw.dummy_in[:] = v_view
    fftw.fft()

    # return a copy of the output
    return np.asarray(fftw.dummy_out).copy()

def ifft(ifftw, v):

    """"
    Generic IFFT function for complex grid-sized variables.
    """
    ## Get copy of input array
    v_view = v

    # copy input into memory view
    ifftw.dummy_in[:] = v_view
    ifftw.ifft()

    return np.asarray(ifftw.dummy_out).copy()

###################################
##       SPECTRA FUNCTIONS       ##
###################################
@njit
def energy_spectrum(w_h, kx, ky, Nx, Ny):

    """
    Computes the energy spectrum from the Fourier vorticity

    w_h : 2d complex array
          - Contains the Fourier vorticity
    kx  : int array
          - The wavenumbers in the first dimension
    ky  : int array
          - The wavenumbers in the second dimension
    Nx  : int
          - Number of collocations in the first dimension
    Ny  : int
          - Number of collocations in the second dimension
    """

    ## Spectrum size
    spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)) + 1)

    ## Velocity arrays
    energy_spec = np.zeros(spec_size)

    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            ## Compute the mode
            spec_indx = int(np.round(np.sqrt(kx[j] * kx[j] + ky[i] * ky[i])))

            k_sqr = (kx[j] ** 2 + ky[i] ** 2)

            if kx[j] == 0.0 and ky[i] == 0.0 or k_sqr == 0.0:
                continue
            else:
                ## Compute prefactor
                k_sqr_inv = 1.0 / k_sqr

                if j == 0 or j == w_h.shape[0] - 1:
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) * k_sqr_inv
                else: 
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) * k_sqr_inv

    return 4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2)

@njit
def energy_spectrum_vel(w_h, kx, ky, Nx, Ny):

    """
    Computes the energy spectrum from the Fourier vorticity in terms of the Fourier velocities

    w_h : 2d complex array
          - Contains the Fourier vorticity
    kx  : int array
          - The wavenumbers in the first dimension
    ky  : int array
          - The wavenumbers in the second dimension
    Nx  : int
          - Number of collocations in the first dimension
    Ny  : int
          - Number of collocations in the second dimension
    """

    ## Spectrum size
    spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)) + 1)

    ## Velocity arrays
    energy_spec = np.zeros(spec_size)

    ## Find u_hat
    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            k_sqr = (kx[j] ** 2 + ky[i] ** 2)

            if kx[j] == 0.0 and ky[i] == 0.0 or k_sqr == 0.0:
                u_hat = np.complex(0.0 + 0.0)
                v_hat = np.complex(0.0 + 0.0)
            else:
                ## Compute prefactor
                k_sqr_inv = 1j / k_sqr

                ## Compute Fourier velocities
                u_hat = ky[i] * k_sqr * w_h[i, j]
                v_hat = -kx[j] * k_sqr * w_h[i, j]

            ## Compute the mode
            spec_indx = int(np.round(np.sqrt(kx[j] * kx[j] + ky[i] * ky[i])))

            if j == 0 or j == w_h.shape[0] - 1:
                ## Update spectrum sum for current mode
                energy_spec[spec_indx] += np.absolute(u_hat * np.conjugate(u_hat)) + np.absolute(v_hat * np.conjugate(v_hat))
            else: 
                ## Update spectrum sum for current mode
                energy_spec[spec_indx] += 2. * np.absolute(u_hat * np.conjugate(u_hat)) + np.absolute(v_hat * np.conjugate(v_hat))

    return 4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2)

@njit
def enstrophy_spectrum(w_h, kx, ky, Nx, Ny):

    ## Spectrum size
    spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)) + 1)

    ## Velocity arrays
    energy_spec = np.zeros(spec_size)

    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            ## Compute the mode
            spec_indx = int(np.round(np.sqrt(kx[j] * kx[j] + ky[i] * ky[i])))

            if kx[j] == 0.0 and ky[i] == 0.0:
                continue
            else:
                if j == 0 or j == w_h.shape[0] - 1:
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))
                else:
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))

    return 4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2)



#####################################
##       SYSTEM MEASUREABLES       ##
#####################################
@njit
def total_energy(w_h, kx, ky, Nx, Ny):

    """
    Computes the total energy from the Fourier vorticity

    w_h : 2d complex array
          - Contains the Fourier vorticity
    kx  : int array
          - The wavenumbers in the first dimension
    ky  : int array
          - The wavenumbers in the second dimension
    Nx  : int
          - Number of collocations in the first dimension
    Ny  : int
          - Number of collocations in the second dimension
    """

    ## The running sum var for the total energy
    tot_enrg = 0.

    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            ## Get |k|^2
            k_sqr = kx[i] ** 2 + ky[j] ** 2

            ## Update running sum
            if k_sqr != 0:
                if j == 0 or j == w_h.shape[1] - 1:
                    tot_enrg += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) / k_sqr
                else:
                    tot_enrg += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) / k_sqr

    return 4. * np.pi * np.pi * tot_enrg * (0.5 / (Nx * Ny)**2)

@njit
def total_enstrophy(w_h, Nx, Ny):

    """
    Computes the total enstrophy from the Fourier vorticity

    w_h : 2d complex array
          - Contains the Fourier vorticity
    Nx  : int
          - Number of collocations in the first dimension
    Ny  : int
          - Number of collocations in the second dimension
    """

    ## The running sum var for the total energy
    tot_enst = 0.

    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            ## Update running sum
            if k_sqr != 0:
                if j == 0 or j == w_h.shape[1] - 1:
                    tot_enst += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))
                else:
                    tot_enst += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))

    return 4. * np.pi * np.pi * tot_enst * (0.5 / (Nx * Ny)**2)


@njit
def total_palinstrophy(w_h, kx, ky, Nx, Ny):

    """
    Computes the total palinstrophy from the Fourier vorticity

    w_h : 2d complex array
          - Contains the Fourier vorticity
    kx  : int array
          - The wavenumbers in the first dimension
    ky  : int array
          - The wavenumbers in the second dimension
    Nx  : int
          - Number of collocations in the first dimension
    Ny  : int
          - Number of collocations in the second dimension
    """

    ## The running sum var for the total energy
    tot_palin = 0.

    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            ## Get |k|^2
            k_sqr = kx[i] ** 2 + ky[j] ** 2

            ## Update running sum
            if k_sqr != 0:
                if j == 0 or j == w_h.shape[1] - 1:
                    tot_palin += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) * k_sqr
                else:
                    tot_palin += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) * k_sqr

    return 4. * np.pi * np.pi * tot_palin * (0.5 / (Nx * Ny)**2)



