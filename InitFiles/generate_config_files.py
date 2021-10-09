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
## Space variables
Nx = 128
Ny = 128
Nk = int(Ny / 2 + 1)

## System parameters
nu             = 0.001
ekmn_alpha     = 1.
hypervisc      = 0 
ekmn_hypo_diff = 0

## Time parameters
t0        = 0.0
T         = 1.0
dt        = 1e-3
step_type = True

## Solver parameters
ic         = "DECAY_TURB"
forcing    = "NONE"
force_k    = 0
save_every = 2

## Directory/File parameters
input_dir       = "NONE"
output_dir      = "./Data/Tmp"
file_only_mode  = False
solver_tag      = "Decay-Test"
post_input_dir  = output_dir
post_output_dir = output_dir

## Job parameters
solver                      = True
postprocessing              = True
solver_procs                = 4
num_solver_job_threads      = 1
num_postprocess_job_threads = 1

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
    'save_data_every'    : save_every
}

## Time variables
config['TIME'] = {
    'start_time'         : t0,
    'end_time'           : T,
    'timestep'           : dt,
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
}

## Job variables
config['JOB'] = {
    'call_solver'                 : solver,
    'call_postprocessing'         : postprocessing,
    'solver_procs'                : solver_procs,
    'num_solver_job_threads'      : num_solver_job_threads,
    'num_postprocess_job_threads' : num_postprocess_job_threads
}

###################################
##       WRITE CONFIG FILE       ##
###################################
with open(config_file_name + '.ini', 'w') as f:
    config.write(f)