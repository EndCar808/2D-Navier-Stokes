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
    H    = '\033[95m'
    B    = '\033[94m'
    C    = '\033[96m'
    G    = '\033[92m'
    Y    = '\033[93m'
    R    = '\033[91m'
    Rst  = '\033[0m'
    Bold = '\033[1m'
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
            self.Nx     = int(Nx)
            self.Ny     = int(Ny)
            self.Nk     = int(Nk)
            self.nu     = float(nu)
            self.t0     = float(t0)
            self.T      = float(T)
            self.ndata  = int(ndata)
            self.u0     = str(u0)
            self.cfl    = float(cfl)
            self.dt     = float(dt)
            self.dx     = float(dx)
            self.dy     = float(dy)
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
        with h5py.File(input_dir, 'r') as file:
            data.ndata = len([g for g in file.keys() if 'Iter' in g])

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

    ## Create instance of data class
    data = SolverData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_file = input_file + "Main_HDF_Data.h5"
    else:
        in_file = input_file

    ## Open file and read in the data
    with h5py.File(in_file, 'r') as file:

        ## Initialize counter
        nn = 0

        # Read in the vorticity
        for group in file.keys():
            if "Iter" in group:
                if 'w' in list(file[group].keys()):
                    data.w[nn, :, :] = file[group]["w"][:, :]
                if 'w_hat' in list(file[group].keys()):
                    data.w_hat[nn, :, :] = file[group]["w_hat"][:, :]
                if 'u' in list(file[group].keys()):
                    data.u[nn, :, :] = file[group]["u"][:, :, :]
                if 'u_hat' in list(file[group].keys()):
                    data.u_hat[nn, :, :] = file[group]["u_hat"][:, :, :]
                if 'TGSoln' in list(file[group].keys()):
                    data.tg_soln[nn, :, :] = file[group]["TGSoln"][:, :]
                data.time[nn] = file[group].attrs["TimeValue"]
                nn += 1
            else:
                continue

        ## Read in the space arrays
        if 'kx' in list(file.keys()):
            data.kx = file["kx"][:]
        if 'ky' in list(file.keys()):
            data.ky = file["ky"][:]
        if 'x' in list(file.keys()):
            data.x  = file["x"][:]
        if 'y' in list(file.keys()):
            data.y  = file["y"][:]

        ## Read system measures
        if 'TotalEnergy' in list(file.keys()):
            data.tot_enrg = file['TotalEnergy'][:]
        if 'TotalEnstrophy' in list(file.keys()):
            data.tot_enst = file['TotalEnstrophy'][:]
        if 'TotalPalinstrophy' in list(file.keys()):
            data.tot_palin = file['TotalPalinstrophy'][:]
        if 'EnergyDissipation' in list(file.keys()):
            data.enrg_diss = file['EnergyDissipation'][:]
        if 'EnstrophyDissipation' in list(file.keys()):
            data.enst_diss = file['EnstrophyDissipation'][:]
        if 'EnergyDissSubset' in list(file.keys()):
            data.enrg_diss_sbst = file['EnergyDissSubset'][:]
        if 'EnstrophyDissSubset' in list(file.keys()):
            data.enst_diss_sbst = file['EnstrophyDissSubset'][:]
        if 'EnergyFluxSubset' in list(file.keys()):
            data.enrg_flux_sbst = file['EnergyFluxSubset'][:]
        if 'EnstrophyFluxSubset' in list(file.keys()):
            data.enst_flux_sbst = file['EnstrophyFluxSubset'][:]

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
            self.enrg_flux_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enst_flux_spectrum = np.zeros((sim_data.ndata, sim_data.spec_size))


    ## Create instance of data class
    data = SpectraData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        file = input_file + "Spectra_HDF_Data.h5"
    else:
        file = input_file

    ## Open file and read in the data
    with h5py.File(file, 'r') as file:

        ## Initialze counter
        nn = 0

        # Read in the spectra
        for group in file.keys():
            if "Iter" in group:
                if 'EnergySpectrum' in list(file[group].keys()):
                    data.enrg_spectrum[nn, :] = file[group]["EnergySpectrum"][:]
                if 'EnstrophySpectrum' in list(file[group].keys()):
                    data.enst_spectrum[nn, :] = file[group]["EnstrophySpectrum"][:]
                if 'EnergyFluxSpectrum' in list(file[group].keys()):
                    data.enrg_flux_spectrum[nn, :] = file[group]["EnergyFluxSpectrum"][:]
                if 'EnstrophyFluxSpectrum' in list(file[group].keys()):
                    data.enst_flux_spectrum[nn, :] = file[group]["EnstrophyFluxSpectrum"][:]
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
        file = input_file + "PostProcessing_HDF_Data.h5"
    else:
        file = input_file

    ## Define a data class for the solver data
    class PostProcessData:

        """
        Class for the run data.
        """

        def __init__(self, in_file = file):
            ## Get the max wavenumber
            self.kmax = int(sim_data.Nx / 3)
            ## Allocate spectra aarrays
            self.phases        = np.zeros((sim_data.ndata, int(2 * self.kmax - 1), int(2 * self.kmax - 1)))
            self.enrg_spectrum = np.zeros((sim_data.ndata, int(2 * self.kmax - 1), int(2 * self.kmax - 1)))
            self.enst_spectrum = np.zeros((sim_data.ndata, int(2 * self.kmax - 1), int(2 * self.kmax - 1)))
            ## Allocate histogram arrays
            self.w_pdf_counts = np.zeros((sim_data.ndata, 1000))
            self.u_pdf_counts = np.zeros((sim_data.ndata, 1000))
            self.w_pdf_ranges = np.zeros((sim_data.ndata, 1001))
            self.u_pdf_ranges = np.zeros((sim_data.ndata, 1001))
            ## Allocate spectra arrays
            self.enst_spectrum_1d = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.enrg_spectrum_1d = np.zeros((sim_data.ndata, sim_data.spec_size))
            ## Allocate solver data arrays
            self.w_hat = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)
            ## Get the number of sectors and the number of triads
            with h5py.File(in_file, 'r') as file:
                if 'SectorAngles' in list(file.keys()):
                    self.theta = file["SectorAngles"][:]
                if 'NumTriadsPerSector' in list(file.keys()):
                    self.num_triads = file["NumTriadsPerSector"][:, :]
                if 'EnstrophyFluxC' in list(file.keys()):
                    self.enst_flux_C = file["EnstrophyFluxC"][:]
            self.num_sect  = self.theta.shape[0] - 1
            ## Enstrophy Flux Spectrum
            self.enst_flux_spec = np.zeros((sim_data.ndata, sim_data.spec_size))
            self.kmax_frac      = float(in_file.split('_')[-1].split("[")[-1].split("]")[0])
            self.kmax_C         = int(self.kmax * self.kmax_frac)
            ## Enstrophy Flux Per Sector
            self.enst_flux_per_sec = np.zeros((sim_data.ndata, self.num_sect))
            ## Phase Sync arrays
            self.phase_R       = np.zeros((sim_data.ndata, self.num_sect))
            self.phase_Phi     = np.zeros((sim_data.ndata, self.num_sect))
            self.triad_R       = np.zeros((sim_data.ndata, 5, self.num_sect))
            self.triad_Phi     = np.zeros((sim_data.ndata, 5, self.num_sect))
            ## Phase Sync Stats
            self.phase_sector_counts = np.zeros((sim_data.ndata, self.num_sect, 200))
            self.phase_sector_ranges = np.zeros((sim_data.ndata, self.num_sect, 201))
            self.triad_sector_counts = np.zeros((sim_data.ndata, self.num_sect, 200, 5))
            self.triad_sector_ranges = np.zeros((sim_data.ndata, self.num_sect, 201, 5))
            self.triad_sector_wghtd_counts = np.zeros((sim_data.ndata, self.num_sect, 200, 5))
            self.triad_sector_wghtd_ranges = np.zeros((sim_data.ndata, self.num_sect, 201, 5))

    ## Create instance of data class
    data = PostProcessData()

    ## Open file and read in the data
    with h5py.File(file, 'r') as file:

        ## Initialize counter
        nn = 0

        # Read in the spectra
        for group in file.keys():
            if "Snap" in group:
                if 'FullFieldPhases' in list(file[group].keys()):
                    data.phases[nn, :] = file[group]["FullFieldPhases"][:]
                if 'FullFieldEnstrophySpectrum' in list(file[group].keys()):
                    data.enst_spectrum[nn, :] = file[group]["FullFieldEnstrophySpectrum"][:]
                if 'FullFieldEnergySpectrum' in list(file[group].keys()):
                    data.enrg_spectrum[nn, :] = file[group]["FullFieldEnergySpectrum"][:]
                if 'VelocityPDFCounts' in list(file[group].keys()):
                    data.u_pdf_counts[nn, :] = file[group]["VelocityPDFCounts"][:]
                if 'VelocityPDFRanges' in list(file[group].keys()):
                    data.u_pdf_ranges[nn, :] = file[group]["VelocityPDFRanges"][:]
                if 'VorticityPDFCounts' in list(file[group].keys()):
                    data.w_pdf_counts[nn, :] = file[group]["VorticityPDFCounts"][:]
                if 'VorticityPDFRanges' in list(file[group].keys()):
                    data.w_pdf_ranges[nn, :] = file[group]["VorticityPDFRanges"][:]
                if 'EnstrophySpectrum' in list(file[group].keys()):
                    data.enst_spectrum_1d[nn, :] = file[group]["EnstrophySpectrum"][:]
                if 'EnergySpectrum' in list(file[group].keys()):
                    data.enrg_spectrum_1d[nn, :] = file[group]["EnergySpectrum"][:]
                if 'PhaseSync' in list(file[group].keys()):
                    data.phase_R[nn, :] = file[group]["PhaseSync"][:]
                if 'AverageAngle' in list(file[group].keys()):
                    data.phase_Phi[nn, :] = file[group]["AverageAngle"][:]
                if 'TriadPhaseSync' in list(file[group].keys()):
                    data.triad_R[nn, :, :] = file[group]["TriadPhaseSync"][:, :]
                if 'TriadAverageAngle' in list(file[group].keys()):
                    data.triad_Phi[nn, :, :] = file[group]["TriadAverageAngle"][:, :]
                if 'w_hat' in list(file[group].keys()):
                    data.w_hat[nn, :, :] = file[group]["w_hat"][:, :]
                if 'EnstrophyFluxSpectrum' in list(file[group].keys()):
                    data.enst_flux_spec[nn, :] = file[group]["EnstrophyFluxSpectrum"][:]
                if 'EnstrophyFluxPerSector' in list(file[group].keys()):
                    data.enst_flux_per_sec[nn, :] = file[group]["EnstrophyFluxPerSector"][:]
                if 'SectorPhasePDF_InTime_Counts' in list(file[group].keys()):
                    data.phase_sector_counts[nn, :, :] = file[group]["SectorPhasePDF_InTime_Counts"][:, :]
                if 'SectorPhasePDF_InTime_Ranges' in list(file[group].keys()):
                    data.phase_sector_ranges[nn, :, :] = file[group]["SectorPhasePDF_InTime_Ranges"][:, :]
                if 'SectorTriadPhasePDF_InTime_Counts' in list(file[group].keys()):
                    data.triad_sector_counts[nn, :, :, :] = file[group]["SectorTriadPhasePDF_InTime_Counts"][:, :, :]
                if 'SectorTriadPhasePDF_InTime_Ranges' in list(file[group].keys()):
                    data.triad_sector_ranges[nn, :, :, :] = file[group]["SectorTriadPhasePDF_InTime_Ranges"][:, :, :]
                if 'SectorTriadPhaseWeightedPDF_InTime_Counts' in list(file[group].keys()):
                    data.triad_sector_wghtd_counts[nn, :, :, :] = file[group]["SectorTriadPhaseWeightedPDF_InTime_Counts"][:, :, :]
                if 'SectorTriadPhaseWeightedPDF_InTime_Ranges' in list(file[group].keys()):
                    data.triad_sector_wghtd_ranges[nn, :, :, :] = file[group]["SectorTriadPhaseWeightedPDF_InTime_Ranges"][:, :, :]
                nn += 1
            else:
                continue

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



