#!/usr/bin/env python    
#######################
##  LIBRARY IMPORTS  ##
#######################
from configparser import ConfigParser
import numpy as np
import os
import sys

from subprocess import Popen
from multiprocessing import Pool
from Plotting.functions import tc
#########################
##  READ COMMAND LINE  ##
#########################
## Read in .ini file
if len(sys.argv) == 2:
    config_file = sys.argv[1]
    print("Input configuration file: " + tc.C + config_file + tc.Rst)
else:
    print("[" + tc.R + "ERROR" + tc.Rst + "] --- No config file provided...Exiting!")
    sys.exit()

############################
##  FUNCTION DEFINITIONS  ##
############################




##########################
##  DEFAULT PARAMETERS  ##
##########################
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

#########################
##  PARSE CONFIG FILE  ##
#########################
## Create parser instance
parser = ConfigParser()

## Read in config file
parser.read(config_file)

## Parse input parameters
for section in parser.sections():
    if section in ['SYSTEM']:
        Nx         = int(parser[section]['nx'])
        Ny         = int(parser[section]['ny'])
        Nk         = int(parser[section]['nk'])
        nu         = float(parser[section]['viscosity'])
        ekmn_alpha = float(parser[section]['drag_coefficient'])
    if section in ['SOLVER']:
        ic         = str(parser[section]['initial_condition'])
        forcing    = str(parser[section]['forcing'])
        force_k    = int(parser[section]['forcing_wavenumber'])
        save_every = int(parser[section]['save_data_every'])
    if section in ['TIME']:
        t0        = float(parser[section]['start_time'])
        T         = float(parser[section]['end_time'])
        dt        = float(parser[section]['start_time'])
        step_type = bool(parser[section]['adaptive_step_type'])
    if section in ['DIRECTORIES']:
        input_dir       = str(parser[section]['solver_input_dir'])
        output_dir      = str(parser[section]['solver_output_dir'])
        solver_tag      = str(parser[section]['solver_tag'])
        post_input_dir  = str(parser[section]['post_input_dir'])
        post_output_dir = str(parser[section]['post_output_dir'])
        file_only_mode  = bool(parser[section]['solver_file_only_mode'])
    if section in ['JOB']:
        solver                      = bool(parser[section]['call_solver'])
        postprocessing              = bool(parser[section]['call_postprocessing'])
        solver_procs                = int(parser[section]['solver_procs'])
        num_solver_job_threads      = int(parser[section]['num_solver_job_threads'])
        num_postprocess_job_threads = int(parser[section]['num_postprocess_job_threads'])
    
        


#########################
##      RUN SOLVER     ##
#########################
if solver:
    print(solver)



##################################
##      RUN POST PROCESSING     ##
##################################
if postprocessing:
    print(postprocessing)