#!/usr/bin/env python    
#######################
##  LIBRARY IMPORTS  ##
#######################
from configparser import ConfigParser
import numpy as np
import os
import sys
import getopt
import distutils.util as utils
from datetime import datetime
from collections.abc import Iterable
from itertools import zip_longest
from subprocess import Popen, PIPE
from Plotting.functions import tc
#########################
##  READ COMMAND LINE  ##
#########################
def parse_cml(argv):

    """
    Parses command line arguments
    """

    ## Create arguments class
    class cmd_args:
        """
        Class for command line arguments
        """

        def __init__(self, init_file = None, cmd_only = False):
            self.init_file = init_file
            self.cmd_only  = cmd_only
            
    ## Initialize class
    cargs = cmd_args()

    # print(getopt.getopt(argv, "i:c:", ["cmdonly"]))
    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:c:", ["cmdonly"])
    except Exception as e:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        print(e)
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:

        if opt in ['-i']:
            ## Read in config file
            cargs.init_file = str(arg)
            print("Input configuration file: " + tc.C + cargs.init_file + tc.Rst)

            if not os.path.isfile(cargs.init_file):
                print("[" + tc.R + "ERROR" + tc.Rst + "] ---> File Does not exist, double check input file path.")
                sys.exit()

        if opt in ['--cmdonly']:
            ## Read in indicator to print out commands to terminal only
            cargs.cmd_only = True

    return cargs

######################
##       MAIN       ##
######################
if __name__ == '__main__':
    ##########################
    ##  PARSE COMMAND LINE  ##
    ##########################
    cmdargs = parse_cml(sys.argv[1:])

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
    hypervisc      = False
    hypervisc_pow  = 2.0 
    ekmn_hypo_diff = False
    ekmn_hypo_pow  = -2.0
    ## Time parameters
    t0          = 0.0
    T           = 1.0
    dt          = 1e-3
    step_type   = True
    cfl_cond    = True
    trans_iters = True
    cfl         = 0.9
    ## Solver parameters
    ic          = "DECAY_TURB"
    forcing     = "NONE"
    force_k     = 0
    force_scale = 1.0
    save_every  = 2
    ## Directory/File parameters
    input_dir       = "NONE"
    output_dir      = "./Data/Tmp/"
    file_only_mode  = False
    solver_tag      = "Decay-Test"
    post_input_dir  = output_dir
    post_output_dir = output_dir
    ## Job parameters
    executable                  = "Solver/bin/main"
    plot_options                = "--full_snap --base_snap --plot --vid"
    plotting                    = True
    solver                      = True
    postprocessing              = True
    collect_data                = False
    solver_procs                = 4
    num_solver_job_threads      = 1
    num_postprocess_job_threads = 1
    num_plotting_job_threads    = 1

    #########################
    ##  PARSE CONFIG FILE  ##
    #########################
    ## Create parser instance
    parser = ConfigParser()

    ## Read in config file
    parser.read(cmdargs.init_file)

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
            if 'nx' in parser[section]:
                for n in parser[section]['nx'].lstrip('[').rstrip(']').split(', '):
                    Nx.append(int(n))
            if 'ny' in parser[section]:    
                for n in parser[section]['ny'].lstrip('[').rstrip(']').split(', '):
                    Ny.append(int(n))
            if 'nk' in parser[section]:
                for n in parser[section]['nk'].lstrip('[').rstrip(']').split(', '):
                    Nk.append(int(n))
            if 'viscosity' in parser[section]:
                for n in parser[section]['viscosity'].lstrip('[').rstrip(']').split(', '):
                    nu.append(float(n))
            if 'drag_coefficient' in parser[section]:
                ekmn_alpha = float(parser[section]['drag_coefficient'])
            if 'hyperviscosity' in parser[section]:
                hypervisc = int(parser[section]['hyperviscosity'] == 'True')
            if 'hypo_diffusion' in parser[section]:
                ekmn_hypo_diff = int(parser[section]['hypo_diffusion'] == 'True')
            if 'hyperviscosity_pow' in parser[section]:
                hypervisc_pow = float(parser[section]['hyperviscosity_pow'])
            if 'hypo_diffusion_pow' in parser[section]:
                ekmn_hypo_pow = float(parser[section]['hypo_diffusion_pow'])
        if section in ['SOLVER']:
            if 'initial_condition' in parser[section]:
                for n in parser[section]['initial_condition'].lstrip('[').rstrip(']').split(', '):
                    ic.append(str(n.lstrip('"').rstrip('"')))
            if 'forcing' in parser[section]:
                forcing = str(parser[section]['forcing'])
            if 'forcing_wavenumber' in parser[section]:
                force_k = int(float(parser[section]['forcing_wavenumber']))
            if 'forcing_scale' in parser[section]:
                force_scale = int(float(parser[section]['forcing_scale']))
            if 'save_data_every' in parser[section]:
                save_every = int(parser[section]['save_data_every'])
        if section in ['TIME']:
            if 'end_time' in parser[section]:
                for n in parser[section]['end_time'].lstrip('[').rstrip(']').split(', '):
                    T.append(float(parser[section]['end_time']))
            if 'timestep' in parser[section]:
                for n in parser[section]['timestep'].lstrip('[').rstrip(']').split(', '):
                    dt.append(float(parser[section]['timestep']))
            if 'cfl' in parser[section]:
                for n in parser[section]['cfl'].lstrip('[').rstrip(']').split(', '):
                    cfl.append(float(parser[section]['cfl']))
            if 'start_time' in parser[section]:
                t0 = float(parser[section]['start_time'])
            if 'cfl_cond' in parser[section]:
                cfl_cond = int(parser[section]['cfl_cond'] == 'True')
            if 'trans_iters' in parser[section]:
                trans_iters = int(parser[section]['trans_iters'] == 'True')
            if 'adaptive_step_type' in parser[section]:
                step_type = int(parser[section]['adaptive_step_type'] == 'True')
        if section in ['DIRECTORIES']:
            if 'solver_input_dir' in parser[section]:
                input_dir = str(parser[section]['solver_input_dir'])
            if 'solver_output_dir' in parser[section]:
                output_dir = str(parser[section]['solver_output_dir'])
            if 'solver_tag' in parser[section]:
                for n in parser[section]['solver_tag'].lstrip('[').rstrip(']').split(', '):
                    solver_tag.append(str(parser[section]['solver_tag']))
            if 'post_input_dir' in parser[section]:
                post_input_dir = str(parser[section]['post_input_dir'])
            if 'post_output_dir' in parser[section]:
                post_output_dir = str(parser[section]['post_output_dir'])
            if 'solver_file_only_mode' in parser[section]:
                file_only_mode = bool(utils.strtobool(parser[section]['solver_file_only_mode']))
            if 'system_tag' in parser[section]:
                system_tag = str(parser[section]['system_tag'])
        if section in ['JOB']:
            if 'executable' in parser[section]:
                executable = str(parser[section]['executable'])
            if 'plotting' in parser[section]:
                plotting = str(parser[section]['plotting'])
            if 'plot_script' in parser[section]:
                plot_script = str(parser[section]['plot_script'])
            if 'plot_options' in parser[section]:
                plot_options = str(parser[section]['plot_options'])
            if 'post_options' in parser[section]:
                post_options = str(parser[section]['post_options'])
            if 'call_solver' in parser[section]:
                solver = bool(utils.strtobool(parser[section]['call_solver']))
            if 'call_postprocessing' in parser[section]:
                postprocessing = bool(utils.strtobool(parser[section]['call_postprocessing']))
            if 'solver_procs' in parser[section]:
                solver_procs = int(parser[section]['solver_procs'])
            if 'collect_data' in parser[section]:
                collect_data = bool(utils.strtobool(parser[section]['collect_data']))
            if 'num_solver_job_threads' in parser[section]:
                num_solver_job_threads = int(parser[section]['num_solver_job_threads'])
            if 'num_postprocess_job_threads' in parser[section]:
                num_postprocess_job_threads = int(parser[section]['num_postprocess_job_threads'])
            if 'num_plotting_job_threads' in parser[section]:
                num_plotting_job_threads = int(parser[section]['num_plotting_job_threads'])

    ## Get the path to the runs output directory
    par_runs_output_dir = os.path.split(output_dir)[0]
    par_runs_output_dir += '/ParallelRunsDump/'
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
        cmd_list = [["mpirun -n {} {} -o {} -n {} -n {} -s {:3.1f} -e {:3.1f} -T {} -c {} -c {:1.6f} -h {:1.6f} -h {} -v {:1.10f} -v {} -v {:1.1f} -d {:1.6f} -d {} -d {:1.1f} -i {} -t {} -f {} -f {} -f {} -p {}".format(
                                                                                                                                                                                    solver_procs, 
                                                                                                                                                                                    executable, 
                                                                                                                                                                                    output_dir, 
                                                                                                                                                                                    nx, ny, 
                                                                                                                                                                                    t0, t, trans_iters, 
                                                                                                                                                                                    cfl_cond, c, 
                                                                                                                                                                                    h, step_type, 
                                                                                                                                                                                    v, hypervisc, hypervisc_pow, 
                                                                                                                                                                                    ekmn_alpha, ekmn_hypo_diff, ekmn_hypo_pow,
                                                                                                                                                                                    u0, 
                                                                                                                                                                                    s_tag, 
                                                                                                                                                                                    forcing, force_k, force_scale, 
                                                                                                                                                                                    save_every)] for nx, ny in zip(Nx, Ny) for t in T for h in dt for u0 in ic for v in nu for c in cfl for s_tag in solver_tag]

        if cmdargs.cmd_only:
            print(tc.C + "\nSolver Commands:\n" + tc.Rst)
            for c in cmd_list:
                print(c)
                print()
        else:
            ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
            groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit 

            ## Loop through grouped iterable
            for processes in zip_longest(*groups): 
                for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
                    ## Print command to screen
                    print("Executing the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)
                    
                    ## Print output to terminal as it comes
                    for line in proc.stdout:
                        sys.stdout.write(line)

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
                with open(par_runs_output_dir + "par_run_solver_output_{}_{}.txt".format(cmdargs.init_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
                    for item in solver_output:
                        file.write("%s\n" % item)

                # Write error to file
                with open(par_runs_output_dir + "par_run_solver_error_{}_{}.txt".format(cmdargs.init_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
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
        cmd_list = [["PostProcessing/bin/main -i {} -o {} -v {:1.10f} -v {} -v {:1.1f} -d {:1.6f} -d {} -d {:1.1f} -f {} -f {} -f {} {}".format(
                                                        post_input_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag), 
                                                        post_output_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag),
                                                        v, hypervisc, hypervisc_pow, 
                                                        ekmn_alpha, ekmn_hypo_diff, ekmn_hypo_pow,
                                                        forcing, force_k, force_scale,
                                                        post_options)] for nx, ny in zip(Nx, Ny) for t in T for v in nu for c in cfl for u0 in ic for s_tag in solver_tag]

        if cmdargs.cmd_only:
            print(tc.C + "\nPost Processing Commands:\n" + tc.Rst)
            for c in cmd_list:
                print(c)
                print()
        else:
            ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
            groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit 

            ## Loop through grouped iterable
            for processes in zip_longest(*groups): 
                for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
                    ## Print command to screen
                    print("Executing the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)

                    ## Print output to terminal as it comes
                    for line in proc.stdout:
                        sys.stdout.write(line)
                    
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
                with open(par_runs_output_dir + "par_run_post_output_{}_{}.txt".format(cmdargs.init_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
                    for item in post_output:
                        file.write("%s\n" % item)

                # Write error to file
                with open(par_runs_output_dir + "par_run_post_error_{}_{}.txt".format(cmdargs.init_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
                    for i, item in enumerate(post_error):
                        file.write("%s\n" % cmd_list[i])
                        file.write("%s\n" % item)
                    

    ###########################
    ##      RUN PLOTTING     ##
    ###########################
    if plotting:
        
        ## Get the number of processes to launch
        proc_limit = num_plotting_job_threads
        print("Number of Post Processing Processes Created = [" + tc.C + "{}".format(proc_limit) + tc.Rst + "]")

        # Create output objects to store process error and output
        if collect_data:
            plot_output = []
            plot_error  = []
        print(nu)
        ## Generate command list 
        cmd_list = [["python3 {} -i {} {}".format(
                                            plot_script, 
                                            post_input_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag), 
                                            plot_options)] for nx, ny in zip(Nx, Ny) for t in T for v in nu for c in cfl for u0 in ic for s_tag in solver_tag]

        if cmdargs.cmd_only:
            print(tc.C + "\nPlotting Commands:\n" + tc.Rst)
            for c in cmd_list:
                print(c)
                print()
        else:
            ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
            groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit 

            ## Loop through grouped iterable
            for processes in zip_longest(*groups): 
                for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
                    ## Print command to screen
                    print("Executing the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)

                    ## Print output to terminal as it comes
                    for line in proc.stdout:
                        sys.stdout.write(line)
                    
                    # Communicate with process to retrive output and error
                    [run_CodeOutput, run_CodeErr] = proc.communicate()

                    # Append to output and error objects
                    if collect_data:
                        plot_output.append(run_CodeOutput)
                        plot_error.append(run_CodeErr)
                    
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
                with open(par_runs_output_dir + "par_run_plot_output_{}_{}.txt".format(cmdargs.init_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
                    for item in plot_output:
                        file.write("%s\n" % item)

                # Write error to file
                with open(par_runs_output_dir + "par_run_plot_error_{}_{}.txt".format(cmdargs.init_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
                    for i, item in enumerate(plot_error):
                        file.write("%s\n" % cmd_list[i])
                        file.write("%s\n" % item)
                    