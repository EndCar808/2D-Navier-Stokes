#!/usr/bin/env python    

######################
##  Library Imports ##
######################
import numpy as np
import h5py
import sys
import os




###############################
##       FUNCTION DEFS       ##
###############################
def read_input_file(input_file, Nx, Ny):

    with h5py.File(input_file, 'r') as file:

        ## Get the number of data saves
        num_saves = len([g for g in list(file.keys()) if 'Iter' in g])

        ## Allocate arrays
        w       = np.zeros((num_saves, Nx, Ny))
        tg_soln = np.zeros((num_saves, Nx, Ny))
        w_hat   = np.ones((num_saves, Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)
        time    = np.zeros((num_saves, ))
        Real    = 0
        Fourier = 0

        # Read in the vorticity
        for i, group in enumerate(file.keys()):
            if "Iter" in group:
                if 'w' in list(file[group].keys()):
                    w[i, :, :] = file[group]["w"][:, :]
                    Real = 1
                if 'w_hat' in list(file[group].keys()):
                    w_hat[i, :, :] = file[group]["w_hat"][:, :]
                    Fourier = 1
                if 'TGSoln' in list(file[group].keys()):
                    tg_soln[i, :, :] = file[group]["TGSoln"][:, :]
                    Fourier = 1
                time[i] = file[group].attrs["TimeValue"]
            else:
                continue

            # Define min and max for plotiting
            w_min = np.amin(w)
            w_max = np.amax(w)
       
            ## Read in the space arrays
            if 'kx' in list(file.keys()):
                kx = file["kx"][:]
            if 'ky' in list(file.keys()):
                ky = file["ky"][:]
            if 'x' in list(file.keys()):
                x  = file["x"][:]
            if 'y' in list(file.keys()):
                y  = file["y"][:]
            ## Read system measures
            if 'TotalEnergy' in list(file.keys()):
                tot_energy = file['TotalEnergy'][:]
            if 'TotalEnstrophy' in list(file.keys()):
                tot_enstr = file['TotalEnstrophy'][:]
            if 'TotalPalinstrophy' in list(file.keys()):
                tot_palin = file['TotalPalinstrophy'][:]


    return w, w_hat, tg_soln, time, x, y, kx, ky, tot_energy, tot_enstr, tot_palin