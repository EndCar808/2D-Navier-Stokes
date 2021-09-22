#!/usr/bin/env python    

#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os



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
def SimData(input_dir, method = "default"):

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
        def __init__(self, Nx = 0, Ny = 0, Nk = 0, nu = 0.0, t0 = 0.0, T = 0.0, ndata = 0, u0 = "TG_VORT", cfl = 0.0, spec_size = 0):
            self.Nx     = int(Nx)
            self.Ny     = int(Ny)
            self.Nk     = int(Nk)
            self.nu     = float(nu)
            self.t0     = float(t0)
            self.T      = float(T)
            self.ndata  = int(ndata)
            self.u0     = str(u0)
            self.cfl    = float(cfl)
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



def ImportData(input_file, sim_data, method = "default"):

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
            self.w       = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.tg_soln = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.w_hat   = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)
            self.time    = np.zeros((sim_data.ndata, ))
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

        nn = 0
        # Read in the vorticity
        for group in file.keys():
            if "Iter" in group:
                if 'w' in list(file[group].keys()):
                    data.w[nn, :, :] = file[group]["w"][:, :]
                if 'w_hat' in list(file[group].keys()):
                    data.w_hat[nn, :, :] = file[group]["w_hat"][:, :]
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
        if 'EnergyDissSubst' in list(file.keys()):
            data.enrg_diss_sbst = file['EnergyDissSubst'][:]
        if 'EnstrophyDissSubst' in list(file.keys()):
            data.enst_diss_sbst = file['EnstrophyDissSubst'][:]
        if 'EnergyFluxSubst' in list(file.keys()):
            data.enrg_flux_sbst = file['EnergyFluxSubst'][:]
        if 'EnstrophyFluxSubst' in list(file.keys()):
            data.enst_flux_sbst = file['EnstrophyFluxSubst'][:]

    ## Get inv wavenumbers
    data.k2 = data.ky**2 + data.kx[:, np.newaxis]**2
    index   = data.k2 != 0.0
    data.k2Inv[index] = 1. / data.k2[index]

    return data


def ImportSpectraData(input_file, sim_data, method = "default"):

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