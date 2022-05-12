#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to compare solver results with decaying turbulence papers
#        Solver data

#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os
from numba import njit
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import getopt
from itertools import zip_longest
import multiprocessing as mprocs
import time as TIME
from subprocess import Popen, PIPE, run
from matplotlib.pyplot import cm
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, import_sync_data
from plot_functions import plot_sector_phase_sync_snaps, plot_sector_phase_sync_snaps_full, plot_sector_phase_sync_snaps_full_sec
###############################
##       FUNCTION DEFS       ##
###############################
def parse_cml(argv):

	"""
	Parses command line arguments
	"""

	## Create arguments class
	class cmd_args:
		"""
		Class for command line arguments
		"""

		def __init__(self, in_dir = None, out_dir = None, in_file = None, num_procs = 20, parallel = False, plotting = False, video = False, triads = False, phases = False, triad_type = None, full = False):
			self.in_dir         = in_dir
			self.out_dir_phases = out_dir
			self.out_dir_triads = out_dir
			self.in_file        = out_dir
			self.parallel       = parallel
			self.plotting       = plotting
			self.video          = video 
			self.phases         = phases
			self.triads         = triads
			self.triad_type     = triad_type
			self.full           = full
			self.triad_plot_type = None
			self.num_procs = num_procs
			self.tag = "None"


	## Initialize class
	cargs = cmd_args()

	try:
		## Gather command line arguments
		opts, args = getopt.getopt(argv, "i:o:f:p:t:", ["plot"])
	except Exception as e:
		print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
		print(e)
		sys.exit()

	## Parse command line args
	for opt, arg in opts:

		if opt in ['-i']:
			## Read input directory
			cargs.in_dir = str(arg)
			print("\nInput Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

			cargs.out_dir = str(arg)
			print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

	return cargs
######################
##       MAIN       ##
######################
if __name__ == '__main__':
	# -------------------------------------
	# # --------- Parse Commnad Line
	# -------------------------------------
	cmdargs = parse_cml(sys.argv[1:])
	if cmdargs.in_file is None:
		method = "default"
		post_file_path = cmdargs.in_dir
	else: 
		method = "file"
		post_file_path = cmdargs.in_dir + cmdargs.in_file
	# -----------------------------------------
	# # --------  Read In data
	# -----------------------------------------
	## Read in simulation parameters
	sys_vars = sim_data(cmdargs.in_dir)

	## Read in solver data
	run_data = import_data(cmdargs.in_dir, sys_vars)

	## Read in spectra data
	spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

	## Read in sync data
	sync_data = import_sync_data(cmdargs.in_dir, sys_vars)


	trim_size = np.trim_zeros(sync_data.theta_k[0, 1:]).shape[0]
	tmp_time_order_k        = np.ones((trim_size, )) * np.complex(0.0, 0.0)
	time_order_k            = np.ones((sys_vars.ndata, trim_size)) * np.complex(0.0, 0.0)
	normed_trim_size = np.trim_zeros(sync_data.theta_k[0, 1:]).shape[0]
	tmp_normed_time_order_k = np.ones((normed_trim_size, )) * np.complex(0.0, 0.0)
	normed_time_order_k     = np.ones((sys_vars.ndata, normed_trim_size)) * np.complex(0.0, 0.0)

	for t in range(1, sys_vars.ndata):
		theta_k_trim     	= sync_data.theta_k[t, :trim_size]
		tmp_time_order_k 	+= np.exp(theta_k_trim * 1j)
		time_order_k[t,  :] = tmp_time_order_k / t
		
		normed_theta_k_trim 	  = sync_data.normed_theta_k[t, :normed_trim_size]
		tmp_normed_time_order_k   += np.exp(normed_theta_k_trim * 1j)
		normed_time_order_k[t, :] = tmp_normed_time_order_k / t

	k_indx = [5, 10, int(trim_size/2), int(trim_size/4), int(trim_size - 1)]
	fig = plt.figure(figsize = (16, 9))
	gs  = GridSpec(2, 2, hspace = 0.6, wspace = 0.3)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(np.absolute(time_order_k[-1, 1:]))
	ax1.plot(np.absolute(normed_time_order_k[-1, 1:]))
	ax1.set_xlabel(r"$k$")
	ax1.set_ylabel(r"$R_k = |\langle e^{i\theta_k} \rangle_T|$")
	ax1.set_ylim([0, 1])
	ax1.legend(["Unnormed", "Normed"])
	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(np.absolute(time_order_k[-1, 1:]))
	ax2.plot(np.absolute(normed_time_order_k[-1, 1:]))
	ax2.set_xlabel(r"$k$")
	ax2.set_ylabel(r"$R_k = |\langle e^{i\theta_k} \rangle_T|$")
	ax2.set_xscale('log')
	ax2.set_ylim([0, 1])
	ax2.legend(["Unnormed", "Normed"])
	ax3 = fig.add_subplot(gs[1, 0])
	for k in k_indx:
		ax3.plot(run_data.time, sync_data.theta_k[:, k])
	ax3.legend([r"$k = {}$".format(k) for k in k_indx])
	ax3.set_xlabel(r"$t$")
	ax3.set_ylabel(r"$\theta_k(t)$")
	ax3.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax4 = fig.add_subplot(gs[1, 1])
	for k in k_indx:
		ax4.plot(run_data.time, np.absolute(time_order_k[:, k]))
	ax4.set_xlabel(r"$t$")
	ax4.set_ylabel(r"$R_k(t)$")
	ax4.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax4.legend([r"$k = {}$".format(k) for k in k_indx])
	ax4.set_yscale('log')
	plt.savefig(cmdargs.out_dir + "Sync.png", bbox_inches='tight')
	plt.close()




	fig = plt.figure(figsize = (16, 9))
	gs  = GridSpec(1, 3, hspace = 0.6, wspace = 0.3)
	ax1 = fig.add_subplot(gs[0, 0])
	im1 = ax1.imshow(np.fft.irfft2(run_data.w_hat[0, :, :]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") # , vmin = w_min, vmax = w_max 
	ax1.set_xlabel(r"$y$")
	ax1.set_ylabel(r"$x$")
	ax1.set_xlim(0.0, run_data.y[-1])
	ax1.set_ylim(0.0, run_data.x[-1])
	ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
	ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax1.set_title(r"$t = {:0.5f}$".format(run_data.time[0]))
	ax2 = fig.add_subplot(gs[0, 1])
	im2 = ax2.imshow(np.fft.irfft2(run_data.w_hat[int(sys_vars.ndata/2 -1), :, :]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") # , vmin = w_min, vmax = w_max 
	ax2.set_xlabel(r"$y$")
	ax2.set_ylabel(r"$x$")
	ax2.set_xlim(0.0, run_data.y[-1])
	ax2.set_ylim(0.0, run_data.x[-1])
	ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
	ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
	ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax2.set_title(r"$t = {:0.5f}$".format(run_data.time[int(sys_vars.ndata/2 -1)]))
	ax3 = fig.add_subplot(gs[0, 2])	
	im3 = ax3.imshow(np.fft.irfft2(run_data.w_hat[-1, :, :]), extent = (run_data.y[0], run_data.y[-1], run_data.x[-1], run_data.x[0]), cmap = "RdBu") # , vmin = w_min, vmax = w_max 
	ax3.set_xlabel(r"$y$")
	ax3.set_ylabel(r"$x$")
	ax3.set_xlim(0.0, run_data.y[-1])
	ax3.set_ylim(0.0, run_data.x[-1])
	ax3.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.y[-1]])
	ax3.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax3.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, run_data.x[-1]])
	ax3.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax3.set_title(r"$t = {:0.5f}$".format(run_data.time[-1]))
	plt.savefig(cmdargs.out_dir + "Vort.png", bbox_inches='tight')
	plt.close()

	fig = plt.figure(figsize = (16, 9))
	gs  = GridSpec(2, 3, hspace = 0.6, wspace = 0.3)
	ax1 = fig.add_subplot(gs[0, 0])
	kindx = int(sys_vars.Nx / 3 + 1)
	ax1.plot(spec_data.enrg_spectrum[0, :kindx]) #  / np.sum(enrg_spec[:kindx])
	ax1.set_xlabel(r"$|\mathbf{k}|$")
	ax1.set_ylabel(r"$\mathcal{K}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
	ax1.set_title(r"Energy Spectrum")
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	ax1.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(spec_data.enrg_spectrum[int(sys_vars.ndata/2 -1), :kindx]) #  / np.sum(enrg_spec[:kindx])
	ax2.set_xlabel(r"$|\mathbf{k}|$")
	ax2.set_ylabel(r"$\mathcal{K}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
	ax2.set_title(r"Energy Spectrum")
	ax2.set_yscale('log')
	ax2.set_xscale('log')
	ax2.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax3 = fig.add_subplot(gs[0, 2])
	ax3.plot(spec_data.enrg_spectrum[-1, :kindx]) #  / np.sum(enrg_spec[:kindx])
	ax3.set_xlabel(r"$|\mathbf{k}|$")
	ax3.set_ylabel(r"$\mathcal{K}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
	ax3.set_title(r"Energy Spectrum")
	ax3.set_yscale('log')
	ax3.set_xscale('log')
	ax3.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax4 = fig.add_subplot(gs[1, 0])
	ax4.plot(spec_data.enst_spectrum[0, :kindx]) #  / np.sum(enrg_spec[:kindx])
	ax4.set_xlabel(r"$|\mathbf{k}|$")
	ax4.set_ylabel(r"$\mathcal{E}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
	ax4.set_title(r"Enstorpy Spectrum")
	ax4.set_yscale('log')
	ax4.set_xscale('log')
	ax4.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax5 = fig.add_subplot(gs[1, 1])
	ax5.plot(spec_data.enst_spectrum[int(sys_vars.ndata/2 -1), :kindx]) #  / np.sum(enrg_spec[:kindx])
	ax5.set_xlabel(r"$|\mathbf{k}|$")
	ax5.set_ylabel(r"$\mathcal{E}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
	ax5.set_title(r"Enstorpy Spectrum")
	ax5.set_yscale('log')
	ax5.set_xscale('log')
	ax5.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax6 = fig.add_subplot(gs[1, 2])
	ax6.plot(spec_data.enst_spectrum[-1, :kindx]) #  / np.sum(enrg_spec[:kindx])
	ax6.set_xlabel(r"$|\mathbf{k}|$")
	ax6.set_ylabel(r"$\mathcal{E}(| \mathbf{k} |)$") #  / \sum \mathcal{K}(|k|)
	ax6.set_title(r"Enstorpy Spectrum")
	ax6.set_yscale('log')
	ax6.set_xscale('log')
	ax6.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)

	plt.savefig(cmdargs.out_dir + "Spectra.png", bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.plot(run_data.tot_enst[:])
	plt.plot(run_data.tot_palin[:])
	plt.xlabel(r"$t$")
	plt.yscale('log')
	plt.legend(["Total Enstrophy", "Total Palinstrophy"])
	plt.grid(which = "major", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	plt.savefig(cmdargs.out_dir + "TotalEnstorphy_TotalPalinstrophy.png", bbox_inches='tight')
	plt.close()