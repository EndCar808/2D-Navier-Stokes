#!/usr/bin/env python    
######################
##  Library Imports ##
######################
from configparser import ConfigParser
import numpy as np
import os
import sys

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
#########################
##  READ COMMAND LINE  ##
#########################
## Read in .ini file
if len(sys.argv) == 2:
    config_file_name = sys.argv[1]
    print("Configuration file: " + tc.C + config_file_name + '.ini' + tc.Rst)
else:
    print("[" + tc.R + "ERROR" + tc.Rst + "] --- No file name provided...Exiting!")
    sys.exit()

###########################
##       VARIABLES       ##
###########################
## NOTE: If providing multiple values they must be in python list type e.g. [1, 2, 3]
## Space variables
Nx = [64]
Ny = [64]
if len(Ny) >= 1:
    Nk = []
    for n in Ny:
        Nk.append(int(n / 2 + 1))
else:
    Nk = [int(Ny / 2 + 1)]

## System parameters
nu             = 3.125e-06
ekmn_alpha     = 0.0
hypervisc      = 2.0 
ekmn_hypo_diff = 0

## Time parameters
t0        = 0.0
T         = 100.0
dt        = 25e-4
cfl       = 0.8
step_type = False

## Solver parameters
ic          = ["DECAY_TURB_BB", "DECAY_TURB_NB", "DECAY_TURB_EXP"]
forcing     = "NONE"
force_k     = 1.0
force_scale = 1.0
save_every  = 100

## Directory/File parameters
input_dir       = "NONE"
output_dir      = "./Data/Tmp/" #os.getenv('NS_DATA_DIR')   ##"./Data/Tmp/"
file_only_mode  = False
solver_tag      = "Decay-Turb-Post-Test"
sys_tag         = "_NAVIER_RK4_FULL_"   ## _EULER_RK4_FULL_
post_input_dir  = output_dir + "SIM_DATA" + sys_tag
post_output_dir = output_dir + "SIM_DATA" + sys_tag

## Job parameters
executable                  = "Solver/bin/solver"
plot_script                 = "Plotting/plot_jet_sync.py"
plot_options                = "--triads=0 --par --plot --vid -f PostProcessing_HDF_Data_SECTORS[24]_KFRAC[0.50]_TAG[" + solver_tag + "].h5 --full=sec -t Test-Sync"
solver                      = True
postprocessing              = True
plotting                    = True
solver_procs                = 4
collect_data                = True
num_solver_job_threads      = 1
num_postprocess_job_threads = 1
num_plottin_job_threads     = 1

##############################
##       CONFIG SETUP       ##
##############################
## Create parser instance
config = ConfigParser()

##---------------------------
## -------- Create Sections
##---------------------------
## System variables
config['SYSTEM'] = {
    'Nx'                : Nx,
    'Ny'                : Ny,
    'Nk'                : Nk,
    'viscosity'         : nu,
    'drag_coefficient'  : ekmn_alpha,
    'hyperviscosity'    : hypervisc,
    'hypo_diffusion'    : ekmn_hypo_diff
}

## Solver variables
config['SOLVER'] = {
    'initial_condition'  : ic,
    'forcing'            : forcing,
    'forcing_wavenumber' : force_k,
    'forcing_scale'      : force_scale,
    'save_data_every'    : save_every
}

## Time variables
config['TIME'] = {
    'start_time'         : t0,
    'end_time'           : T,
    'timestep'           : dt,
    'cfl'                : cfl,
    'adaptive_step_type' : step_type
}

## Directories / Files
config['DIRECTORIES'] = {
    'solver_output_dir'     : output_dir,
    'solver_input_dir'      : input_dir,
    'solver_file_only_mode' : file_only_mode,
    'solver_tag'            : solver_tag,
    'post_output_dir'       : post_output_dir,
    'post_input_dir'        : post_input_dir,
    'system_tag'            : sys_tag
}

## Job variables
config['JOB'] = {
    'executable'                  : executable,
    'plot_script'                 : plot_script,
    'plot_options'                : plot_options,
    'call_solver'                 : solver,
    'call_postprocessing'         : postprocessing,
    'plotting'                    : plotting,
    'collect_data'                : collect_data,
    'solver_procs'                : solver_procs,
    'num_solver_job_threads'      : num_solver_job_threads,
    'num_postprocess_job_threads' : num_postprocess_job_threads,
    'num_plotting_job_threads'    : num_plottin_job_threads
}

###################################
##       WRITE CONFIG FILE       ##
###################################
with open(config_file_name + '.ini', 'w') as f:
    config.write(f)