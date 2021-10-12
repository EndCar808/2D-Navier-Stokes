#!/usr/bin/env python    
#######################
##  LIBRARY IMPORTS  ##
#######################
from configparser import ConfigParser
import numpy as np
import os
import sys

from datetime import datetime
from collections.abc import Iterable
from itertools import zip_longest
from subprocess import Popen, PIPE
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
cfl       = np.sqrt(3)
## Solver parameters
ic         = "DECAY_TURB"
forcing    = "NONE"
force_k    = 0
save_every = 2
## Directory/File parameters
input_dir       = "NONE"
output_dir      = "./Data/Tmp/"
file_only_mode  = False
solver_tag      = "Decay-Test"
post_input_dir  = output_dir
post_output_dir = output_dir
## Job parameters
executable                  = "Solver/bin/main"
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

## Create list objects
Nx         = []
Ny         = []
Nk         = []
nu         = []
ic         = []
T          = []
dt         = []
cfl        = []
solver_tag = []

## Parse input parameters
for section in parser.sections():
    if section in ['SYSTEM']:
        Nx.append(int(parser[section]['nx']))
        Ny.append(int(parser[section]['ny']))
        Nk.append(int(parser[section]['nk']))
        nu.append(float(parser[section]['viscosity']))
        ekmn_alpha = float(parser[section]['drag_coefficient'])
    if section in ['SOLVER']:
        ic.append(str(parser[section]['initial_condition']))
        forcing    = str(parser[section]['forcing'])
        force_k    = int(parser[section]['forcing_wavenumber'])
        save_every = int(parser[section]['save_data_every'])
    if section in ['TIME']:
        T.append(float(parser[section]['end_time']))
        dt.append(float(parser[section]['timestep']))
        cfl.append(float(parser[section]['cfl']))
        t0        = float(parser[section]['start_time'])
        step_type = bool(parser[section]['adaptive_step_type'])
    if section in ['DIRECTORIES']:
        input_dir       = str(parser[section]['solver_input_dir'])
        output_dir      = str(parser[section]['solver_output_dir'])
        solver_tag.append(str(parser[section]['solver_tag']))
        post_input_dir  = str(parser[section]['post_input_dir'])
        post_output_dir = str(parser[section]['post_output_dir'])
        file_only_mode  = bool(parser[section]['solver_file_only_mode'])
        system_tag      = str(parser[section]['system_tag'])
    if section in ['JOB']:
        executable                  = str(parser[section]['executable'])
        solver                      = bool(parser[section]['call_solver'])
        postprocessing              = bool(parser[section]['call_postprocessing'])
        solver_procs                = int(parser[section]['solver_procs'])
        collect_data                = bool(parser[section]['collect_data'])
        num_solver_job_threads      = int(parser[section]['num_solver_job_threads'])
        num_postprocess_job_threads = int(parser[section]['num_postprocess_job_threads'])
    

#########################
##      RUN SOLVER     ##
#########################
if solver:

    ## Get the number of processes to launch
    proc_limit = num_solver_job_threads
    print("Number of Solver Processes Created = [" + tc.C + "{}".format(proc_limit) + tc.Rst + "]")

    # Create output objects to store process error and output
    if collect_data:
        solver_output = []
        solver_error  = []

    ## Generate command list 
    cmd_list = [["mpirun -n {} {} -o {} -n {} -n {} -s {:3.1f} -e {:3.1f} -c {:1.6f} -h {:1.6f} -v {:1.6f} -d {:1.6f} -i {} -t {} -f {} -f {} -p {}".format(solver_procs, executable, output_dir, nx, ny, t0, t, c, h, v, ekmn_alpha, u0, s_tag, forcing, force_k, save_every)] for nx, ny in zip(Nx, Ny) for t in T for h in dt for u0 in ic for v in nu for c in cfl for s_tag in solver_tag]
   

    ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
    groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit 

    ## Loop through grouped iterable
    for processes in zip_longest(*groups): 
        for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
            ## Print command to screen
            print("Executing the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)
            
            # Communicate with process to retrive output and error
            [run_CodeOutput, run_CodeErr] = proc.communicate()

            # Append to output and error objects
            if collect_data:
                solver_output.append(run_CodeOutput)
                solver_error.append(run_CodeErr)
            
            ## Print both to screen
            print(run_CodeOutput)
            print(run_CodeErr)

            ## Wait until all finished
            proc.wait()

    if collect_data:
        # Get data and time
        now = datetime.now()
        d_t = now.strftime("%d%b%Y_%H:%M:%S")

        # Write output to file
        with open("Data/ParallelRunsDump/par_run_solver_output_{}_{}.txt".format(config_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
            for item in solver_output:
                file.write("%s\n" % item)

        # Write error to file
        with open("Data/ParallelRunsDump/par_run_solver_error_{}_{}.txt".format(config_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
            for i, item in enumerate(solver_error):
                file.write("%s\n" % cmd_list[i])
                file.write("%s\n" % item)

##################################
##      RUN POST PROCESSING     ##
##################################
if postprocessing:
    
    ## Get the number of processes to launch
    proc_limit = num_postprocess_job_threads
    print("Number of Post Processing Processes Created = [" + tc.C + "{}".format(proc_limit) + tc.Rst + "]")

    # Create output objects to store process error and output
    if collect_data:
        post_output = []
        post_error  = []

    ## Generate command list 
    cmd_list = [["PostProcessing/bin/main -i {} -o {}".format(post_input_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag), post_output_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag))] for nx, ny in zip(Nx, Ny) for t in T for v in nu for c in cfl for u0 in ic for s_tag in solver_tag]
    

    ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
    groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit 

    ## Loop through grouped iterable
    for processes in zip_longest(*groups): 
        for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
            ## Print command to screen
            print("Executing the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)
            
            # Communicate with process to retrive output and error
            [run_CodeOutput, run_CodeErr] = proc.communicate()

            # Append to output and error objects
            if collect_data:
                post_output.append(run_CodeOutput)
                post_error.append(run_CodeErr)
            
            ## Print both to screen
            print(run_CodeOutput)
            print(run_CodeErr)

            ## Wait until all finished
            proc.wait()

    if collect_data:
        # Get data and time
        now = datetime.now()
        d_t = now.strftime("%d%b%Y_%H:%M:%S")

        # Write output to file
        with open("Data/ParallelRunsDump/par_run_post_output_{}_{}.txt".format(config_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
            for item in post_output:
                file.write("%s\n" % item)

        # Write error to file
        with open("Data/ParallelRunsDump/par_run_post_error_{}_{}.txt".format(config_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
            for i, item in enumerate(post_error):
                file.write("%s\n" % cmd_list[i])
                file.write("%s\n" % item)