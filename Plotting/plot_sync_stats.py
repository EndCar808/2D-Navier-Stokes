import numpy as np
import h5py as h5
import sys
import os
from numba import njit
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
if mpl.__version__ > '2':    
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
import concurrent.futures as cf
from functions import tc, sim_data, import_data, import_spectra_data, import_post_processing_data, get_flux_spectrum, compute_pdf
# from plot_functions import plot_sector_phase_sync_snaps, plot_sector_phase_sync_snaps_full, plot_sector_phase_sync_snaps_full_sec, plot_sector_phase_sync_snaps_all
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
            self.phase_order = False


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:p:t:", ["par", "plot", "vid", "phase", "triads=", "full=", "phase_order"])
    except:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:

        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("\nInput Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

        elif opt in ['-p']:
            ## Read in number of processes
            cargs.num_procs = int(arg)

        elif opt in ['--par']:
            ## Read in parallel indicator
            cargs.parallel = True

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--vid']:
            ## If videos are to be made
            cargs.video = True

        elif opt in ['--phase']:
            ## If phases are to be plotted
            cargs.phases = True

        elif opt in ['--full']:
            ## If full figure is to be plotted
            cargs.full = True

            ## Read in the triad type
            cargs.triad_plot_type = str(arg)
        elif opt in ['--triads']:
            ## If triads are to be plotted
            cargs.triads = True

            ## Read in the triad type
            cargs.triad_type = str(arg)
        elif opt in ['--phase_order']:
            ## If phase_order are to be plotted
            cargs.phase_order = True

        elif opt in ['-t']:
            cargs.tag = str(arg)

    return cargs


@njit
def get_normed_collective_phase_over_t(collect_phase):

    ## Get dims
    s = collect_phase.shape
    num_t, num_sect = s[0], s[1]
    r    = np.zeros(s, dtype='float64')
    phi  = np.zeros(s, dtype='float64')
    enst = np.zeros(s, dtype='float64')
    
    for i in range(num_sect):
    	normd_const = np.amax(collect_phase[:, i])
    	for t in range(num_t):
            r[t, i]    = np.absolute(collect_phase[t, i] / normd_const)
            phi[t, i]  = np.angle(collect_phase[t, i] / normd_const)
            enst[t, i] = np.real(collect_phase[t, i])

    return r, phi, enst

@njit
def get_normed_collective_phase_2d(collect_phase, norm_const):

    ## Get dims
    s = collect_phase.shape
    r    = np.zeros(s, dtype='float64')
    phi  = np.zeros(s, dtype='float64')
    enst = np.zeros(s, dtype='float64')

    for i in range(s[0]):
        for j in range(s[1]):
            if norm_const[i, j] != 0.0:
                r[i, j]    = np.absolute(collect_phase[i, j] / norm_const[i, j])
                phi[i, j]  = np.angle(collect_phase[i, j] / norm_const[i, j])
                enst[i, j] = np.real(collect_phase[i, j])

    return r, phi, enst


@njit
def get_q_norms(q):
	num_t, num_sect = q.shape

	q_max = np.zeros((num_t, ))
	q_avg = np.zeros((num_t, ))
	q_l2  = np.zeros((num_t, ))
	for t in range(num_t):
		q_max[t] = np.amax(q[t, :])
		q_avg[t] = np.mean(q[t, :])
		q_l2[t]  = np.linalg.norm(q[t, :]) / np.sqrt(num_sect)

	return q_max, q_avg, q_l2


@njit
def get_min_max_incr(data):

	num_t, num_x, num_y = data.shape
	incrments = [1, 10, num_x//2]

	mind = 1e10
	maxd = 0.0
	for t in range(num_t):
		for x in range(num_x):
			for y in range(num_y):
				for r in incrments:
					if x + r < num_x:
						tmp = data[t, int(x + r), y] - data[t, x, y]

						maxd = np.maximum(tmp, maxd)
						mind = np.minimum(tmp, mind)
	return mind, maxd

@njit
def get_incr_hist(data, mind, maxd, nbins):

	num_t, num_x, num_y = data.shape

	ranges = np.linspace(mind, maxd, nbins + 1)
	incrments = [1, 10, num_x//2]
	hist = np.zeros((nbins, len(incrments)))

	for t in range(num_t):
		for x in range(num_x):
			for y in range(num_y):
				for r_indx, r in enumerate(incrments):
					if x + r < num_x:
						tmp = data[t, int(x + r), y] - data[t, x, y]

						for n in range(nbins):
							low = ranges[n]
							high = ranges[n + 1]
							if tmp >= low and tmp < high:
								hist[n, r_indx]+=1 

	return hist, ranges


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
	print("Reading in Data")
	## Read in simulation parameters
	sys_vars = sim_data(cmdargs.in_dir)

	# Read in solver data
	run_data = import_data(cmdargs.in_dir, sys_vars)

	## Read in spectra data
	spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)

	## Read in post processing data
	post_data = import_post_processing_data(post_file_path, sys_vars, method)


	cmdargs.out_dir_stats     = cmdargs.out_dir + "PHASE_SYNC_STATS/"
	if os.path.isdir(cmdargs.out_dir_stats) != True:
		print("Making folder:" + tc.C + " PHASE_SYNC_STATS/" + tc.Rst)
		os.mkdir(cmdargs.out_dir_stats)

	try_type = 0

	my_magma = mpl.colors.ListedColormap(cm.magma.colors[::-1])
	my_magma.set_under(color = "white")
	my_magma.set_over(color = "black")
	my_hsv = mpl.cm.hsv
	my_hsv.set_under(color = "white")
	my_hsv.set_over(color = "white")
	my_jet = mpl.cm.jet
	my_jet.set_under(color = "white") 
	my_jet.set_over(color = "white") 

	## Plot space time
	theta_k3 = post_data.theta_k3
	dtheta_k3 = theta_k3[1] - theta_k3[0]
	angticks      = [-np.pi/2, -3*np.pi/8, -np.pi/4.0, -np.pi/8, 0.0, np.pi/8, np.pi/4.0, 3*np.pi/8, np.pi/2.0]
	angtickLabels = [r"$-\frac{\pi}{2}$", r"$-\frac{3\pi}{8}$", r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
	angtickLabels_alpha = np.linspace(0, len(theta_k3) + 1, num = len(angtickLabels), endpoint = False, dtype = "int64").tolist()
	theta_k3_min     = -np.pi/2 - dtheta_k3 / 2
	theta_k3_max     = np.pi/2 + dtheta_k3 /2
	alpha_min = theta_k3_min
	alpha_max = theta_k3_max

	## Get normalized collective phase data
	if post_data.phase_order_normed_const:
		phase_order_R, phase_order_Phi, phase_order_enst = np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect))
		# phase_order_R_1D, phase_order_Phi_1D, phase_order_enst_1D = np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect))
		# phase_order_R_2D, phase_order_Phi_2D, phase_order_enst_2D = np.zeros((sys_vars.ndata, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect, post_data.num_k1_sect)), np.zeros((sys_vars.ndata, post_data.num_k3_sect, post_data.num_k1_sect))

		## Get the normalized collective phase data
		print(post_data.phase_order_C_theta_triads.shape)
		_ , phase_order_Phi, phase_order_enst = get_normed_collective_phase_over_t(post_data.phase_order_C_theta_triads[:, try_type, :])
		tmp_R = np.absolute(post_data.phase_order_C_theta_triads[:, try_type, :])
		print(np.amax(tmp_R, axis=0)[np.newaxis, :])
		phase_order_R = tmp_R / np.amax(tmp_R, axis=0)[np.newaxis, :]
	    # phase_order_R_1D[i,:], phase_order_Phi_1D[i, :], phase_order_enst_1D[i, :]          = get_normed_collective_phase(post_data.phase_order_C_theta_triads_1d[i, try_type, :], np.diag(post_data.phase_order_norm_const[i, 0, try_type, :, :]))
	    # phase_order_R_2D[i,:, :], phase_order_Phi_2D[i, :, :], phase_order_enst_2D[i, :, :] = get_normed_collective_phase_2d(post_data.phase_order_C_theta_triads_2d[i, try_type, :, :], post_data.phase_order_norm_const[i, 0, try_type, :, :])


	fig = plt.figure(figsize = (21, 13))
	gs  = GridSpec(3, 1, hspace=0.25)
	ax6 = fig.add_subplot(gs[0, 0])
	v_max = 1.0
	im6 = ax6.imshow(np.rot90(np.flipud(phase_order_R), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_magma, vmin = 0.0, vmax = v_max)
	ax6.set_yticks(angticks)
	ax6.set_yticklabels(angtickLabels)
	ax6.set_xlabel(r"$t$")
	ax6.set_ylabel(r"$\theta$")
	ax6.set_title(r"Sync Per Sector")
	div6  = make_axes_locatable(ax6)
	cbax6 = div6.append_axes("right", size = "5%", pad = 0.05)
	cb6   = plt.colorbar(im6, cax = cbax6)
	cb6.set_label(r"$| \mathcal{R}_\theta |$")
	ax7 = fig.add_subplot(gs[1, 0])
	im7 = ax7.imshow(np.rot90(np.flipud(phase_order_Phi), k=-1), aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = my_hsv, vmin = -np.pi, vmax = np.pi)
	ax7.set_yticks(angticks)
	ax7.set_yticklabels(angtickLabels)
	ax7.set_ylabel(r"$\theta$")
	ax7.set_title(r"Average Angle Per Sector")
	div7  = make_axes_locatable(ax7)
	cbax7 = div7.append_axes("right", size = "5%", pad = 0.05)
	cb7   = plt.colorbar(im7, cax = cbax7)
	cb7.set_ticks([-np.pi, -np.pi/2.0, 0., np.pi/2, np.pi])
	cb7.set_ticklabels([r"$-\pi$", r"$\frac{-\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
	cb7.set_label(r"$\arg \{ \mathcal{R}_\theta \}$")
	ax8 = fig.add_subplot(gs[2, 0])
	data = np.rot90(np.flipud(phase_order_enst), k=-1)
	im8 = ax8.imshow(data, aspect='auto', extent = (1, sys_vars.ndata, theta_k3_min, theta_k3_max), cmap = "seismic", vmin=np.amin(data), vmax=np.absolute(np.amin(data))) #vmin = flux_lims[4], vmax = flux_lims[5] #, norm = mpl.colors.SymLogNorm(linthresh = 0.1)
	ax8.set_yticks(angticks)
	ax8.set_yticklabels(angtickLabels)
	ax8.set_ylabel(r"$\theta$")
	ax8.set_title(r"Enstrophy Flux Per Sector")
	div8  = make_axes_locatable(ax8)
	cbax8 = div8.append_axes("right", size = "5%", pad = 0.05)
	cb8   = plt.colorbar(im8, cax = cbax8)
	cb8.set_label(r"$\Re \{ \mathcal{R}_\theta \}$")
	plt.savefig(cmdargs.out_dir_stats + "/Type[{}]_COLLECTIVE_SpaceTime_PerSector.png".format(try_type), bbox_inches="tight")
	plt.close()

	## Condiitonal PDFs
	Q_theta_t = phase_order_R

	q_max, q_avg, q_l2 = get_q_norms(Q_theta_t)

	fig = plt.figure(figsize = (12, 9))
	gs  = GridSpec(1, 1, hspace=0.25)
	ax6 = fig.add_subplot(gs[0, 0])
	ax6.plot(q_max, label=r"max")
	ax6.plot(q_avg, label=r"avg")
	ax6.plot(q_l2, label=r"L2")
	ax6.legend()
	plt.savefig(cmdargs.out_dir_stats + "/Q_norm.png".format(try_type), bbox_inches="tight")
	plt.close()

	sync_cutoff = 0.25

	incrments = [1, 10, 128//2]
	w_un_sync = run_data.w[q_max <= sync_cutoff, :, :]
	w_sync    = run_data.w[q_max > sync_cutoff, :, :]

	print(w_un_sync.shape[0], w_sync.shape[0])

	mind, maxd = get_min_max_incr(w_un_sync)
	print(mind, maxd)
	unsync_hist, unsync_ranges = get_incr_hist(w_un_sync, mind, maxd, 100)

	fig = plt.figure(figsize = (12, 9))
	gs  = GridSpec(1, 2, hspace=0.25)
	ax1 = fig.add_subplot(gs[0, 0])
	for i in range(len(incrments)):
		bin_cent, pdf = compute_pdf(unsync_ranges, unsync_hist[:, i], normalized = True)
		ax1.plot(bin_cent, pdf, label=r"{}".format(incrments[i]))
	ax1.set_yscale('log')
	ax1.legend()
	
	mind, maxd = get_min_max_incr(w_sync)
	print(mind, maxd)
	sync_hist, sync_ranges = get_incr_hist(w_sync, mind, maxd, 100)
	ax1 = fig.add_subplot(gs[0, 1])
	for i in range(len(incrments)):
		bin_cent, pdf = compute_pdf(sync_ranges, sync_hist[:, i], normalized = True)
		ax1.plot(bin_cent, pdf, label=r"{}".format(incrments[i]))
	ax1.set_yscale('log')
	ax1.legend()
	plt.savefig(cmdargs.out_dir_stats + "/PDF.png".format(try_type), bbox_inches="tight")
	plt.close()
	
