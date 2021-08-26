#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##
######################
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
import re
import glob
from matplotlib.gridspec import GridSpec
from subprocess import Popen, PIPE

from Plotting.functions import read_input_file

## For colour printing to terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

######################
##       MAIN       ##
######################
if __name__ == '__main__':

    # executable
    exe = "./Solver/bin/main_TG_test"

    # output dirs
    out_dir_n  = "./Data/ErrorTesting/Error_N/"
    out_dir_dt = "./Data/ErrorTesting/Error_dt/"

    ###------------------------
    ### Check Executable
    ###------------------------
    ## If executable doesn't exist than make it:
    print("\nChecking if executable exists...")
    if not os.path.isfile(exe):

        print(bcolors.WARNING + "Executable does not exist!")
        print(bcolors.WARNING + "Now making executable:" + bcolors.OKCYAN + exe)

        ## Command to execute
        cmd = "cd ./Solver; make test; cd .."

        ## Open subprocess to execute make command
        process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

        ## Communicate with subprocess - print to screen errors  and output
        [runCodeOutput, runCodeErr] = process.communicate()
        print(runCodeOutput)
        print(runCodeErr)

        ## Wait unitl it is finished
        process.wait()
    else:
        print(bcolors.OKGREEN + "[SUCCESS]: " + bcolors.ENDC + "Executable Exists!")
    print()

    ###------------------------
    ### Check for Output Dirs
    ###------------------------
    ## If output folder does not exist create it
    print("Checking if output error directory exists...")
    if not os.path.isdir(out_dir_n):
        print(bcolors.WARNING + "Output folder does not exist...")
        print(bcolors.WARNING + "Now creating output folder:" + bcolors.OKCYAN + out_dir_n)

        os.mkdir(out_dir_n)
        print(bcolors.OKGREEN + "[SUCCESS]: " +  bcolors.ENDC + "Output directory for errror vs N now created!")
    else:
        print(bcolors.OKGREEN + "[SUCCESS]: " +  bcolors.ENDC + "Output directory for errror vs N already exists!")
    if not os.path.isdir(out_dir_dt):
        print(bcolors.WARNING + "Output folder does not exist...")
        print(bcolors.WARNING + "Now creating output folder:" + bcolors.OKCYAN + out_dir_dt)

        os.mkdir(out_dir_dt)
        print(bcolors.OKGREEN + "[SUCCESS]: " +  bcolors.ENDC + "Output directory for errror vs N now created!")
    else:
        print(bcolors.OKGREEN + "[SUCCESS]: " +  bcolors.ENDC + "Output directory for errror vs N already exists!")
    print()



    #####################################################################################
    ######                                                                          #####
    ######                          ERROR AS A FUNCTION OF dt                       #####
    ######                                                                          #####
    #####################################################################################
    N   = 128
    cfl     = [2.0, 1.75, 1.5, 1.25, 1.0, 0.75]  
    cfl_run = cfl.copy() 

    ## Loop files in output directory to see which datafiles exist
    for file in os.listdir(out_dir_dt):
        
        if file.endswith('.h5'):
            for dt in cfl:
                if "CFL[" + str(cfl) + "]" in file:
                    print(bcolors.OKGREEN + "[SUCCESS]:" + bcolors.ENDC + " Data file for CFL = " + bcolors.OKCYAN + "{}".format(dt) + bcolors.ENDC + " exists!")

                    ## Remove from running list if data file exists already
                    cfl_run.remove(dt)
                    break
    
    ## Loop through and generate files that dont exist            
    for dt in cfl_run:

        ## Print update
        print(bcolors.WARNING + "\nExecuting CFL = " + bcolors.OKCYAN + "{}".format(dt) + bcolors.ENDC)

        ## Create command
        cmd = 'mpirun -n 4 ' + exe + ' -o ' + out_dir_dt + ' -n ' + str(N) + ' -n ' + str(N) + ' -s 0.0 -e 1.0 -h 0.0001 -i \"TG_VORT\" -v 1.0 -c ' + str(dt)
        print(bcolors.WARNING + "Command: " + bcolors.ENDC + cmd)

        # Open subprocess to execute command
        process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

        ## Communicate with subprocess - print to screen errors and output
        [runCodeOutput, runCodeErr] = process.communicate()
        print("Output: \n{}".format(runCodeOutput))
        print("Errors: {}".format(runCodeErr))

        ## Wait unitl it is finished
        process.wait()
    print()



    #####################################################################################
    ######                                                                          #####
    ######                          ERROR AS A FUNCTION OF N                        #####
    ######                                                                          #####
    #####################################################################################
    ###------------------------
    ### Check for Data
    ###------------------------
    ## Create dataspace
    # N        = [2 ** i for i in range(6, 9)]
    # N_run    = N.copy()
    # printing = {2**6: 100, 2**7: 250, 2**8: 500, 2**9: 2500, 2**10: 5000}

    # ## Loop files in output directory to see which datafiles exist
    # for file in os.listdir(out_dir_n):
        
    #     if file.endswith('.h5'):
    #         for n in N:
    #             if "N[" + str(n) + "," + str(n) + "]" in file:
    #                 print(bcolors.OKGREEN + "[SUCCESS]:" + bcolors.ENDC + " Data file for N = " + bcolors.OKCYAN + "{}".format(n) + bcolors.ENDC + " exists!")

    #                 ## Remove from running list if data file exists already
    #                 N_run.remove(n)
    #                 break
    
    # ## Loop through and generate files that dont exist            
    # for n in N_run:

    #     ## Print update
    #     print(bcolors.WARNING + "\nExecuting N = " + bcolors.OKCYAN + "{}".format(n) + bcolors.ENDC)

    #     ## Create command
    #     cmd = 'mpirun -n 4 ' + exe + ' -o ' + out_dir_n + ' -n ' + str(n) + ' -n ' + str(n) + ' -s 0.0 -e 1.0 -h 0.0001 -i \"TG_VORT\" -v 1.0 -p ' + str(printing[n]) 
    #     print(bcolors.WARNING + "Command: " + bcolors.ENDC + cmd)

    #     # Open subprocess to execute command
    #     process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

    #     ## Communicate with subprocess - print to screen errors and output
    #     [runCodeOutput, runCodeErr] = process.communicate()
    #     print("Output: \n{}".format(runCodeOutput))
    #     print("Errors: {}".format(runCodeErr))

    #     ## Wait unitl it is finished
    #     process.wait()
    # print()

    # ###------------------------
    # ### Compute Errors
    # ###------------------------
    # data_linf = {}
    # data_l2   = {}
    # for file in os.listdir(out_dir_n):
        
    #     if file.endswith('.h5'): 
    #         ## Retirve the n from the file name
    #         n = int(re.split('\[', file)[1].split(',')[0])
    #         data_linf[n] = []
    #         data_l2[n]   = []

    #         ## Read in data
    #         print(bcolors.WARNING + "Reading in file: " + bcolors.OKCYAN+ "{}".format(file) + bcolors.ENDC)
    #         w       = read_input_file(out_dir_n + file, n, n)[0]
    #         tg_soln = read_input_file(out_dir_n + file, n, n)[2]

    #         ## Compute error norms
    #         l2   = np.zeros(w.shape[0])
    #         linf = np.zeros(w.shape[0])
    #         for i in range(w.shape[0]):
    #             linf[i]   = np.amax(np.absolute(w[i, :, :] - tg_soln[i, :, :]))
    #             # linf[i]   = np.linalg.norm(np.absolute(w[i, :, :] - tg_soln[i, :, :]), ord = np.inf)
    #             l2[i]     = np.linalg.norm(np.absolute(w[i, :, :] - tg_soln[i, :, :]), ord = 2)

    #         ## Collect error norms for this n
    #         data_linf[n].append(np.amax(linf))
    #         data_l2[n].append(np.linalg.norm(l2, ord = 2))

    # ## Order the data
    # l2norm   = []
    # linfnorm = []
    # for n in N:
    #     l2norm.append(data_l2[n])
    #     linfnorm.append(data_linf[n])
    # print(bcolors.OKGREEN + "[SUCCESS]" + bcolors.ENDC + "All errors computed\n")

    # ###------------------------
    # ### Plot Results
    # ###------------------------
    # print(bcolors.WARNING + "Now plotting data.." + bcolors.ENDC)
    # fig = plt.figure(figsize = (16, 8))
    # gs  = GridSpec(1, 2, hspace = 0.4, wspace = 0.4)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(N, l2norm, '.-')
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')
    # ax1.set_title("L2 Norm")
    # ax1.grid(True, which = True)

    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.plot(N, linfnorm, '.-')
    # ax2.set_yscale('log')
    # ax2.set_xscale('log')
    # ax2.set_title("Max Absolute Error")
    # ax2.grid(True, which = True)

    # plt.savefig(out_dir_n + './TG_Error_as_function_of_N.png')
    # plt.close()

