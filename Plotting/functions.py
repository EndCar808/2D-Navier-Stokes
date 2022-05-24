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
            self.Nx        = int(Nx)
            self.Ny        = int(Ny)
            self.Nk        = int(Nk)
            self.nu        = float(nu)
            self.t0        = float(t0)
            self.T         = float(T)
            self.ndata     = int(ndata)
            self.u0        = str(u0)
            self.cfl       = float(cfl)
            self.dt        = float(dt)
            self.dx        = float(dx)
            self.dy        = float(dy)
            self.spec_size = int(spec_size)


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

                ## Parse the timestep
                if line.startswith('Finishing Timestep'):
                    data.dt = float(line.split()[-1])

                ## Parse the spatial increment
                if line.startswith('Spatial Increment'):
                    data.dy = float(line.split()[-1].rstrip(']'))
                    data.dx = float(line.split()[-2].rstrip(',').lstrip('['))

            ## Get spectrum size
            data.spec_size = int(np.sqrt((data.Nx / 2)**2 + (data.Ny / 2)**2) + 1)
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
                data.t0 = float(term.split('-')[0].lstrip('T['))
                data.T  = float(term.split('-')[-1].rstrip(']'))

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
        data.spec_size = int(np.sqrt((data.Nx / 2)**2 + (data.Ny / 2)**2) + 1)

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
            ## Allocate system measure arrays
            self.tot_enrg  = np.zeros((int(sim_data.ndata * 2), ))
            self.tot_enst  = np.zeros((int(sim_data.ndata * 2), ))
            self.tot_palin = np.zeros((int(sim_data.ndata * 2), ))
            self.enrg_diss = np.zeros((int(sim_data.ndata * 2), ))
            self.enst_diss = np.zeros((int(sim_data.ndata * 2), ))
            self.enrg_diss_sbst = np.zeros((int(sim_data.ndata * 2), ))
            self.enst_diss_sbst = np.zeros((int(sim_data.ndata * 2), ))
            self.enrg_flux_sbst = np.zeros((int(sim_data.ndata * 2), ))
            self.enst_flux_sbst = np.zeros((int(sim_data.ndata * 2), ))
            ## Allocate spatial arrays
            self.kx    = np.zeros((sim_data.Nx, ))
            self.ky    = np.zeros((sim_data.Nk, ))
            self.x     = np.zeros((sim_data.Nx, ))
            self.y     = np.zeros((sim_data.Ny, ))
            self.k2    = np.zeros((sim_data.Nx, sim_data.Nk))
            self.k2Inv = np.zeros((sim_data.Nx, sim_data.Nk))
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
            print("\nPreparing real space vorticity...", end = " ")
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

    ## Get inv wavenumbers
    data.k2 = data.ky**2 + data.kx[:, np.newaxis]**2
    index   = data.k2 != 0.0
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
                ## Get the number of sectors
                if 'SectorAngles' in list(f.keys()):
                    self.theta        = f["SectorAngles"][:]
                    self.num_sect     = self.theta.shape[0]
                    self.num_k1_sects = self.num_sect
                if 'SectorAngles' not in list(f.keys()):
                    self.num_sect     = int(in_f.split('_')[-3].split("[")[-1].split("]")[0])
                    self.num_k1_sects = self.num_sect
                ## Get the number of triads per sector
                if 'NumTriadsPerSector' in list(f.keys()):
                    self.num_triads = f["NumTriadsPerSector"][:, :]
                ## Get the number of triads per sector
                if 'NumTriadsPerSector_1D' in list(f.keys()):
                    self.num_triads_1d = f["NumTriadsPerSector_1D"][:, :]
                if 'NumTriadsPerSectorAcrossSector' in list(f.keys()):
                    self.num_triads_across_sec = f["NumTriadsPerSectorAcrossSector"][:, :]
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
                ## Get the increment histogram data
                if 'LongitudinalVelIncrements_BinRanges' in list(f.keys()):
                    self.long_vel_incr_ranges = f["LongitudinalVelIncrements_BinRanges"][:, :]
                if 'LongitudinalVelIncrements_BinCounts' in list(f.keys()):
                    self.long_vel_incr_counts = f["LongitudinalVelIncrements_BinCounts"][:, :]
                if 'TransverseVelIncrements_BinRanges' in list(f.keys()):
                    self.trans_vel_incr_ranges = f["TransverseVelIncrements_BinRanges"][:, :]
                if 'TransverseVelIncrements_BinCounts' in list(f.keys()):
                    self.trans_vel_incr_counts = f["TransverseVelIncrements_BinCounts"][:, :]
                if 'LongitudinalVortIncrements_BinRanges' in list(f.keys()):
                    self.long_vort_incr_ranges = f["LongitudinalVortIncrements_BinRanges"][:, :]
                if 'LongitudinalVortIncrements_BinCounts' in list(f.keys()):
                    self.long_vort_incr_counts = f["LongitudinalVortIncrements_BinCounts"][:, :]
                if 'TransverseVortIncrements_BinRanges' in list(f.keys()):
                    self.trans_vort_incr_ranges = f["TransverseVortIncrements_BinRanges"][:, :]
                if 'TransverseVortIncrements_BinCounts' in list(f.keys()):
                    self.trans_vort_incr_counts = f["TransverseVortIncrements_BinCounts"][:, :]
                ## Get the structure function data
                if 'LongitudinalStructureFunctions' in list(f.keys()):
                    self.long_str_func = f["LongitudinalStructureFunctions"][:, :]
                if 'TransverseStructureFunctions' in list(f.keys()):
                    self.trans_str_func = f["TransverseStructureFunctions"][:, :]
                if 'AbsoluteLongitudinalStructureFunctions' in list(f.keys()):
                    self.long_str_func_abs = f["AbsoluteLongitudinalStructureFunctions"][:, :]
                if 'AbsoluteTransverseStructureFunctions' in list(f.keys()):
                    self.trans_str_func_abs = f["AbsoluteTransverseStructureFunctions"][:, :]
                ## Get the gradient histograms
                if 'VelocityGradient_x_BinRanges' in list(f.keys()):
                    self.grad_u_x_ranges = f["VelocityGradient_x_BinRanges"][:]
                if 'VelocityGradient_x_BinCounts' in list(f.keys()):
                    self.grad_u_x_counts = f["VelocityGradient_x_BinCounts"][:]
                if 'VelocityGradient_y_BinRanges' in list(f.keys()):
                    self.grad_u_y_ranges = f["VelocityGradient_y_BinRanges"][:]
                if 'VelocityGradient_y_BinCounts' in list(f.keys()):
                    self.grad_u_y_counts = f["VelocityGradient_y_BinCounts"][:]
                if 'VorticityGradient_x_BinRanges' in list(f.keys()):
                    self.grad_w_x_ranges = f["VorticityGradient_x_BinRanges"][:]
                if 'VorticityGradient_x_BinCounts' in list(f.keys()):
                    self.grad_w_x_counts = f["VorticityGradient_x_BinCounts"][:]
                if 'VorticityGradient_y_BinRanges' in list(f.keys()):
                    self.grad_w_y_ranges = f["VorticityGradient_y_BinRanges"][:]
                if 'VorticityGradient_y_BinCounts' in list(f.keys()):
                    self.grad_w_y_counts = f["VorticityGradient_y_BinCounts"][:]

                ## Get the number of k1 sectors 
                if 'TriadPhaseSyncAcrossSector' in list(f['Snap_00000'].keys()):
                    self.num_k1_sects = f['Snap_00000']['TriadPhaseSyncAcrossSector'][:, :, :].shape[-1]

            ## Data indicators
            self.no_w     = False
            self.no_w_hat = False
            self.no_u     = False
            self.no_u_hat = False
            
            ## Set the number of Triad Types
            NUM_TRIAD_TYPES = 7

            ## Get the max wavenumber
            self.kmax = int(sim_data.Nx / 3)
            ## Allocate spectra aarrays
            self.phases        = np.zeros((sim_data.ndata, int(2 * self.kmax - 1), int(2 * self.kmax - 1)))
            self.enrg_spectrum = np.zeros((sim_data.ndata, int(2 * self.kmax - 1), int(2 * self.kmax - 1)))
            self.enst_spectrum = np.zeros((sim_data.ndata, int(2 * self.kmax - 1), int(2 * self.kmax - 1)))
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
            self.kmax_frac      = float(in_file.split('_')[-2].split("[")[-1].split("]")[0])
            self.kmax_C         = int(self.kmax * self.kmax_frac)
            ## Energy Flux Spectrum
            self.enrg_flux_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enrg_diss_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.d_enrg_dt_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            ## Enstorphy Flux and Diss from C_theta
            self.enst_flux_C_theta = np.zeros((sim_data.ndata, self.num_sect))
            self.enst_diss_C_theta = np.zeros((sim_data.ndata, self.num_sect))
            ## Collective Phase order parameter for C_theta
            self.phase_order_C_theta           = np.ones((sim_data.ndata, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads    = np.ones((sim_data.ndata, self.num_sect)) * np.complex(0.0, 0.0)
            self.phase_order_C_theta_triads_1d = np.ones((sim_data.ndata, self.num_sect)) * np.complex(0.0, 0.0)
            ## Enstrophy Flux Per Sector
            self.enst_flux_per_sec    = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.enst_flux_per_sec_1d = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.enst_flux_per_sec_across_sec = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects))
            ## Phase Sync arrays
            self.phase_R              = np.zeros((sim_data.ndata, self.num_sect))
            self.phase_Phi            = np.zeros((sim_data.ndata, self.num_sect))
            self.triad_R              = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_Phi            = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_R_1d           = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_Phi_1d         = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect))
            self.triad_R_across_sec   = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects))
            self.triad_Phi_across_sec = np.zeros((sim_data.ndata, NUM_TRIAD_TYPES, self.num_sect, self.num_k1_sects))
            ## Phase Sync Stats
            self.phase_sector_counts = np.zeros((sim_data.ndata, self.num_sect, 200))
            self.phase_sector_ranges = np.zeros((sim_data.ndata, self.num_sect, 201))
            self.triad_sector_counts = np.zeros((sim_data.ndata, self.num_sect, 200, NUM_TRIAD_TYPES))
            self.triad_sector_ranges = np.zeros((sim_data.ndata, self.num_sect, 201, NUM_TRIAD_TYPES))
            self.triad_sector_wghtd_counts = np.zeros((sim_data.ndata, self.num_sect, 200, NUM_TRIAD_TYPES))
            self.triad_sector_wghtd_ranges = np.zeros((sim_data.ndata, self.num_sect, 201, NUM_TRIAD_TYPES))

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
                if 'TriadPhaseSyncAcrossSector' in list(f[group].keys()):
                    data.triad_R_across_sec[nn, :, :] = f[group]["TriadPhaseSyncAcrossSector"][:, :]
                if 'TriadAverageAngleAcrossSector' in list(f[group].keys()):
                    data.triad_Phi_across_sec[nn, :, :] = f[group]["TriadAverageAngleAcrossSector"][:, :]
                if 'EnstrophyFlux_C_theta' in list(f[group].keys()):
                    data.enst_flux_C_theta[nn, :] = f[group]["EnstrophyFlux_C_theta"][:]
                if 'EnstrophyDiss_C_theta' in list(f[group].keys()):
                    data.enst_diss_C_theta[nn, :] = f[group]["EnstrophyDiss_C_theta"][:]
                if 'PhaseOrder_C_theta' in list(f[group].keys()):
                    data.phase_order_C_theta[nn, :] = f[group]["PhaseOrder_C_theta"][:]
                if 'PhaseOrder_C_theta_triads' in list(f[group].keys()):
                    data.phase_order_C_theta_triads[nn, :] = f[group]["PhaseOrder_C_theta_triads"][:]
                if 'PhaseOrder_C_theta_triads_1D' in list(f[group].keys()):
                    data.phase_order_C_theta_triads_1d[nn, :] = f[group]["PhaseOrder_C_theta_triads_1D"][:]
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
                if 'EnstrophyFluxPerSectorAcrossSector' in list(f[group].keys()):
                    data.enst_flux_per_sec_across_sec[nn, :, :, :] = f[group]["EnstrophyFluxPerSectorAcrossSector"][:, :, :]
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
                nn += 1
            else:
                continue

        if data.no_w:
            print("\nPreparing real space vorticity...", end = " ")
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
            spec_indx = int(np.round(np.sqrt(kx[i] * kx[i] + ky[j] * ky[j])))

            if kx[i] == 0.0 and ky[i] == 0.0:
                continue
            else:
                ## Compute prefactor
                k_sqr = 1.0 / (kx[i] ** 2 + ky[j] ** 2)

                if j == 0 or j == w_h.shape[0] - 1:
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) * k_sqr
                else: 
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j])) * k_sqr

    return 4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2), np.sum(4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2))

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
            spec_indx = int(np.round(np.sqrt(kx[i] * kx[i] + ky[j] * ky[j])))

            if j == 0 or j == w_h.shape[0] - 1:
                ## Update spectrum sum for current mode
                energy_spec[spec_indx] += np.absolute(u_hat * np.conjugate(u_hat)) + np.absolute(v_hat * np.conjugate(v_hat))
            else: 
                ## Update spectrum sum for current mode
                energy_spec[spec_indx] += 2. * np.absolute(u_hat * np.conjugate(u_hat)) + np.absolute(v_hat * np.conjugate(v_hat))

    return 4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2), np.sum(4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2))

@njit
def enstrophy_spectrum(w_h, kx, ky, Nx, Ny):

    # """
    # Computes the enstrophy spectrum from the Fourier vorticity

    # w_h : 2d complex array
    #       - Contains the Fourier vorticity
    # kx  : int array 
    #       - The wavenumbers in the first dimension
    # ky  : int array 
    #       - The wavenumbers in the second dimension
    # Nx  : int
    #       - Number of collocations in the first dimension 
    # Ny  : int
    #       - Number of collocations in the second dimension
    # """

    # ## Spectrum size
    # spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)) + 1)

     ## Velocity arrays
    # enstrophy_spec = np.zeros(spec_size)

    # for i in range(w_h.shape[0]):
    #     for j in range(w_h.shape[1]):

             ## Compute the indx
     #        spec_indx = int(np.round(np.sqrt(kx[i] * kx[i] + ky[j] * ky[j])))

      #       if j == 0 or j == w_h.shape[0] - 1:
                 ## Update the spectrum sum for the current mode
     #            enstrophy_spec[spec_indx] += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))
     #        else:
     #            ## Update the spectrum sum for the current mode
     #             enstrophy_spec[spec_indx] += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))

    # return 4. * np.pi * np.pi * enstrophy_spec * 0.5 / (Nx * Ny)**2, np.sum(4. * np.pi * np.pi * enstrophy_spec * 0.5 / (Nx * Ny)**2)

    ## Spectrum size
    spec_size = int(np.sqrt((Nx / 2) * (Nx / 2) + (Ny / 2) * (Ny / 2)) + 1)

    ## Velocity arrays
    energy_spec = np.zeros(spec_size)

    for i in range(w_h.shape[0]):
        for j in range(w_h.shape[1]):

            ## Compute the mode
            spec_indx = int(np.round(np.sqrt(kx[i] * kx[i] + ky[j] * ky[j])))

            if kx[i] == 0.0 and ky[i] == 0.0:
                continue
            else:
                ## Compute prefactor
                k_sqr = 1.0 / (kx[i] ** 2 + ky[j] ** 2)

                if j == 0 or j == w_h.shape[0] - 1:
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))
                else:
                    ## Update spectrum sum for current mode
                    energy_spec[spec_indx] += 2. * np.absolute(w_h[i, j] * np.conjugate(w_h[i, j]))

    return 4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2), np.sum(4. * np.pi * np.pi * energy_spec * 0.5 / ((Nx * Ny) ** 2))



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



