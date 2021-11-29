#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##
######################
import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
import h5py
import sys
import os
import re
import glob
from matplotlib.gridspec import GridSpec
from subprocess import Popen, PIPE
from itertools import zip_longest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Plotting.functions import tc, sim_data, import_data, import_spectra_data


######################
##       MAIN       ##
######################
if __name__ == '__main__':

	## System parameters
	Nx  = 512
	Ny  = Nx
	t0  = 0.0
	T   = 0.01
	dt  = 0.0001
	tag = "DECAY-TURB-ALT-TEST"
	u0  = "DECAY_TURB_ALT"
	nu  = 3.5e-9
	num_procs    = 4
	test_res_dir = "/home/ecarroll/PhD/2D_Navier_Stokes/Data/Test"
	test_res_folder = test_res_dir + "/" + tag + "/"

	## Make sub directory to store results and run commands
	if not os.path.isdir(test_res_folder):
		os.mkdir(test_res_folder)
		print("Making results sub folder: " + tc.C + test_res_folder + tc.Rst)

		## Commands
		my_exec   = "/home/ecarroll/PhD/2D_Navier_Stokes/Solver/bin/solver_DT2_test"
		bren_exec = "/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/navierstokes_mpi/3D_Euler_Spectral/bin/PS_navier_RK4_H2D_PDE.x"
		py_exec   = ""

		## Generate solver commands
		my_cmd     = ["mpirun -n {} {} -n {} -n {} -s {} -e {} -h {} -c 0.900000 -t {} -i {} -v {} -o {} -f 0 -p 1".format(num_procs, my_exec, Nx, Ny, t0, T, dt, tag, u0, nu, test_res_folder)]
		bren_cmd   = ["mpirun -n {} {} -n {} -n {} -s {} -e {} -h {} -t {} -i {} -v {} -o {}".format(num_procs, bren_exec, Nx, Ny, t0, T, dt, tag, u0, nu, test_res_folder)]
		cmd_list   = [my_cmd, bren_cmd] # , py_exec

		print("Running solvers...")
		#############################
		##       RUN SOLVERS       ##
		#############################
		## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
		groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * 1

		## Loop through grouped iterable
		for processes in zip_longest(*groups):
                    for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
                        ## Print command to screen
                        print("\nExecuting the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)

                        ## Communicate with process to retrive output and error
                        [run_CodeOutput, run_CodeErr] = proc.communicate()

                        ## Print both to screen
                        print(run_CodeOutput)
                        print(run_CodeErr)

                        ## Wait until all finished
                        proc.wait()

	if os.path.isdir(test_res_folder):

		#################################
		##       GATHERING DATA        ##
		#################################
		print("Gathering data...")

		my_code_folder   = ""
		bren_code_folder = ""
		py_code_folder   = ""
		for f in os.listdir(test_res_folder):
			if "SIM_DATA" in f:
				my_code_folder = test_res_folder + f + '/'
			if "RESULTS" in f:
				bren_code_folder = test_res_folder + f  + '/'
			if "Python_Data" in f:
				py_code_folder = test_res_folder + f + '/'

		if not my_code_folder:
			print("Missing results from 2D NS in: " + tc.C + test_res_folder + tc.Rst)
		else:
			## Read in simulation data from my code
			sys_vars = sim_data(my_code_folder)

			## Read in solver data from my code
			run_data  = import_data(my_code_folder, sys_vars)
			spec_data = import_spectra_data(my_code_folder, sys_vars)
		if not bren_code_folder:
			print("Missing results from Brendan's code in: " + tc.C + test_res_folder + tc.Rst)
		else:
			## Read in Brendan's code
			with h5py.File(bren_code_folder + "HDF_Global_FOURIER.h5", 'r') as globalf:

				## Get the number of groups
				ndata = len([g for g in globalf.keys() if 'Timestep' in g])

				## Allocate data
				b_w_hat = np.ones((ndata, Nx, int(Ny /2 + 1))) * np.complex(0.0, 0.0)
				b_solv_time = np.ones((ndata, ))

				## Initialze counter
				nn = 0
				# Read in the spectra
				for group in globalf.keys():
                                    if "Timestep" in group:
                                        if "W_hat" in list(globalf[group].keys()):
                                            b_w_hat[nn, :, :] = globalf[group]["W_hat"][:, :]
                                        b_solv_time[nn] = globalf[group].attrs["TimeValue"]
                                        nn += 1
				if "kx" in list(globalf["Timestep_0000"].keys()):
					b_kx = globalf["Timestep_0000"]["kx"][:]
				if "ky" in list(globalf["Timestep_0000"].keys()):
					b_ky = globalf["Timestep_0000"]["ky"][:]


			with h5py.File(bren_code_folder + "HDF_Local_N[{}][{}].h5".format(Nx, Ny), 'r') as localf:

				## Allocate data
				b_max_vort      = localf["MaxVorticity"][:]
				b_time     		= localf["TimeValues"][:]
				b_tot_enrg 		= localf["TotEnergy"][:]
				b_tot_enst 		= localf["TotEnstrophy"][:]
				b_tot_enst_diss = localf["TotEnstrophyDissapation"][:]
				b_tot_enrg_diss = localf["TotEnergyDissapation"][:]

			with h5py.File(bren_code_folder + "HDF_Energy_Spect.h5", 'r') as specf:

				## Get the number of groups
				ndata_spec = len([g for g in specf.keys() if 'Timestep' in g])

				## Get the spectrum size
				nspec = int(np.sqrt((Nx/2)**2 + (Ny/2)**2) + 1)

				## Allocate data
				b_enrg_spec = np.zeros((ndata_spec, nspec))
				b_enst_spec = np.zeros((ndata_spec, nspec))
				b_k_bins    = np.zeros((ndata_spec, nspec))
				b_spec_time = np.zeros((ndata_spec))

				## Initialze counter
				nn = 0

				# Read in the spectra
				for group in specf.keys():
					if "Timestep" in group:
						if "EnergySpectrum" in list(specf[group].keys()):
							b_enrg_spec[nn, :] = specf[group]["EnergySpectrum"][:]
						if "EnstrophySpectrum" in list(specf[group].keys()):
							b_enst_spec[nn, :] = specf[group]["EnstrophySpectrum"][:]
						if "k_bins" in list(specf[group].keys()):
							b_k_bins[nn, :] = specf[group]["k_bins"][:]
						b_spec_time[nn] = specf[group].attrs["TimeValue"]
						nn += 1

			with h5py.File(bren_code_folder + "HDF_Flux_Spect.h5", 'r') as specf:

				## Get the number of groups
				ndata_spec = len([g for g in specf.keys() if 'Timestep' in g])

				## Get the spectrum size
				nspec = int(np.sqrt((Nx/2)**2 + (Ny/2)**2) + 1)

				## Allocate data
				b_enrg_flux = np.zeros((ndata_spec, nspec))
				b_enst_flux = np.zeros((ndata_spec, nspec))

				## Initialze counter
				nn = 0

				# Read in the spectra
				for group in specf.keys():
					if "Timestep" in group:
						if "Flux_U_Spectrum" in list(specf[group].keys()):
							b_enrg_flux[nn, :] = specf[group]["Flux_U_Spectrum"][:]
						if "Flux_W_Spectrum" in list(specf[group].keys()):
							b_enst_flux[nn, :] = specf[group]["Flux_W_Spectrum"][:]
						nn += 1
		if not py_code_folder:
			print("Missing results from Python code in: " + tc.C + test_res_folder + tc.Rst)



		################################
		##       PLOTTING DATA        ##
		################################
		print("Plotting data...")

		## Plot error in vorticity
		fig = plt.figure(figsize = (16, 8))
		gs  = GridSpec(1, 3, wspace = 0.2)
		indexes = [0, int(ndata/2), run_data.w_hat.shape[0] - 1]
		for i, indx in enumerate(indexes):
                    ax = fig.add_subplot(gs[0, i])
                    im = ax.imshow(np.absolute(run_data.w[indx, :, :] - np.fft.irfft2(b_w_hat[i, :, :])), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "jet")
                    ax.set_xlabel(r"$y$")
                    ax.set_ylabel(r"$x$")
                    ax.set_xlim(0.0, run_data.y[-1])
                    ax.set_ylim(0.0, run_data.x[-1])
                    ax.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
                    ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
                    ax.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
                    ax.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
                    ax.set_title(r"$t = {:0.5f}$".format(run_data.time[indx]))
                    div  = make_axes_locatable(ax)
                    cbax = div.append_axes("right", size = "10%", pad = 0.05)
                    cb   = plt.colorbar(im, cax = cbax)
                    cb.set_label(r"$|\omega - \omega_b|$")
		plt.savefig(test_res_folder + "VorticityFieldError.png", bbox_inches = 'tight')
		plt.close()


		## System Measures
		fig = plt.figure(figsize = (16, 8))
		gs  = GridSpec(2, 2)
		ax1 = fig.add_subplot(gs[0, 0])
		ax1.plot(run_data.time, run_data.tot_enrg[:])
		ax1.plot(b_time, b_tot_enrg[:], '--')
		ax1.set_xlabel(r"$t$")
		ax1.set_ylabel(r"$\mathcal{K}(t)$")
		ax1.set_xlim(run_data.time[0], run_data.time[-1])
		ax1.legend([r"My Code", r"Brendan's Code", "Python"])
		ax1.set_title(r"Total Energy")
		ax2 = fig.add_subplot(gs[0, 1])
		ax2.plot(run_data.time, run_data.tot_enst[:])
		ax2.plot(b_time, b_tot_enst[:], ":")
		ax2.set_xlabel(r"$t$")
		ax2.set_ylabel(r"$\mathcal{E}(t)$")
		ax2.set_xlim(run_data.time[0], run_data.time[-1])
		ax2.legend([r"My Code", r"Brendan's Code", "Python"])
		ax2.set_title(r"Total Enstorphy")
		ax3 = fig.add_subplot(gs[1, 0])
		ax3.plot(run_data.time, run_data.enrg_diss[:])
		ax3.plot(b_time, b_tot_enrg_diss[:], '--')
		ax3.set_xlabel(r"$t$")
		ax3.set_ylabel(r"$\epsilon(t)$")
		ax3.set_xlim(run_data.time[0], run_data.time[-1])
		ax3.legend([r"My Code", r"Brendan's Code", "Python"])
		ax3.set_title(r"Energy Dissipation")
		ax4 = fig.add_subplot(gs[1, 1])
		ax4.plot(run_data.time, run_data.enst_diss[:])
		ax4.plot(b_time, b_tot_enst_diss[:], ":")
		ax4.set_xlabel(r"$t$")
		ax4.set_ylabel(r"$\eta(t)$")
		ax4.set_xlim(run_data.time[0], run_data.time[-1])
		ax4.legend([r"My Code", r"Brendan's Code", "Python"])
		ax4.set_title(r"Enstrophy Dissipation")
		plt.savefig(test_res_folder + "System_Measures.png", bbox_inches = 'tight')
		plt.close()

		## Initial Spectra
		kmax = int(sys_vars.Nx/3 + 1)
		kk   = np.arange(1, kmax)
		fig  = plt.figure(figsize = (16, 8))
		gs   = GridSpec(1, 2, hspace = 0.2)
		ax1  = fig.add_subplot(gs[0, 0])
		ax1.plot(kk, spec_data.enrg_spectrum[0, 1:kmax], '--')
		ax1.plot(b_k_bins[0, 1:kmax], b_enrg_spec[0, 1:kmax] * 4. * np.pi**2 * (0.5 / (Nx * Ny)**2))
		ax1.set_xlabel(r"$|\mathbf{k}|$")
		ax1.set_ylabel(r"$\mathcal{K}(|\mathbf{k}|)$")
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.legend([r"My Code", r"Brendan's Code", "Python"])
		ax1.set_title(r"Energy Spectrum")
		ax2 = fig.add_subplot(gs[0, 1])
		ax2.plot(kk, spec_data.enst_spectrum[0, 1:kmax], '--')
		ax2.plot(b_k_bins[0, 1:kmax], b_enst_spec[0, 1:kmax] * 4. * np.pi**2 * (0.5 / (Nx * Ny)**2))
		ax2.set_xlabel(r"$|\mathbf{k}|$")
		ax2.set_ylabel(r"$\mathcal{E}(|\mathbf{k}|)$")
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.legend([r"My Code", r"Brendan's Code", "Python"])
		ax2.set_title(r"Enstrophy Spectrum")
		plt.savefig(test_res_folder + "InitialSpectra.png", bbox_inches='tight')
		plt.close()

		print(b_spec_time)

		## Flux Spectra
		fig = plt.figure(figsize = (16, 8))
		gs  = GridSpec(1, 2)
		ax1 = fig.add_subplot(gs[0, 0])
		ax1.plot(np.cumsum(spec_data.enrg_flux_spectrum[0, :kmax]))
		ax1.plot(b_enrg_flux[0, :kmax], '--')
		ax1.set_xlabel(r"$|\mathbf{k}|$")
		ax1.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
		ax1.set_title(r"Energy Flux Spectra")
		ax2 = fig.add_subplot(gs[0, 1])
		ax2.plot(np.cumsum(spec_data.enst_flux_spectrum[0, :kmax]))
		ax2.plot(b_enst_flux[0, :kmax], '--')
		ax2.set_title(r"Enstrophy Flux Spectra")
		ax2.set_xlabel(r"$|\mathbf{k}|$")
		ax2.set_ylabel(r"$\Pi(|\mathbf{k}|)$")
		plt.savefig(test_res_folder + "FluxSpectra.png", bbox_inches='tight')
		plt.close()

		#################################
		##       COMPARING DATA        ##
		#################################
		print("Comparing data...")


		if len(my_code_folder) > 0 and len(bren_code_folder) > 0:

			## Check space arrays
			bren_kx_passed = 0
			if np.allclose(run_data.kx, b_kx) and np.allclose(run_data.ky, b_ky):
				print(tc.G + "[TEST PASSED]: " + tc.Rst + "Solver " + tc.C + "[Wavenumbers]" + tc.Rst + " data matches Brendan's code")
			else:
				print(tc.R + "[TEST FAILURE]: " + tc.Rst + "Solver Wavenumbers mismatch")

			## Loop through time arrays to compare
			bren_passed = 0
			for i in range(len(run_data.time)):
				if np.isclose(b_time[i], run_data.time[i], rtol = 1e-10):
					bren_passed += 1
					continue
				else:
					print(tc.R + "[TEST FAILURE]: " + tc.Rst + "Time data at iteration " + tc.C + "[{}]".format(i) + tc.Rst + " ---- My code value: " + tc.C + "[{}]".format(run_data.time[i]) + tc.Rst + " Brendan's code value: "  + tc.C + "[{}]".format(b_time[i]) + tc.Rst)
					break

			## Check if test passed
			if bren_passed == len(run_data.time):
				print(tc.G + "[TEST PASSED]: " + tc.Rst + "Solver " + tc.C + "[Time]" + tc.Rst + " data matches Brendan's code")

				## Loop through and compare Fourier vorticity
				bren_passed = 0
				for i in range(run_data.w_hat.shape[0]):
					if np.isclose(b_solv_time[i], run_data.time[i], rtol = 1e-10):
						if np.allclose(run_data.w_hat[i, :, :], b_w_hat[i, :, :]):
							bren_passed += 1
							continue
						else:
							print(tc.R + "[TEST FAILURE]: " + tc.Rst + "Fourier Vorticity data at iteration " + tc.C + "[{}]".format(i) + tc.Rst + " ---- L2 Error: " + tc.C + "[{}]".format(np.linalg.norm(run_data.w_hat[i, :, :] - b_w_hat[i, :, :])) + tc.Rst + " Linf Error: "  + tc.C + "[{}]".format(np.max(np.sum(np.absolute(run_data.w_hat[i, :, :] - b_w_hat[i, :, :]), axis = 1))) + tc.Rst)
							break

				if bren_passed == len(run_data.time):
					print(tc.G + "[TEST PASSED]: " + tc.Rst + "Solver" + tc.C + "[Fourier Vorticity]" + tc.Rst + " data matches Brendan's code")
	else:
		print(tc.R + "[ERROR]: " + tc.Rst + "Double check parameters in python script / solvers are made correctly and working")
		sys.exit()

