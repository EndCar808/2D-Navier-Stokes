import h5py as h5
import matplotlib as mpl
if mpl.__version__ > '2':
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import getopt
from functions import tc
from matplotlib.gridspec import GridSpec


def slope_fit(x, y, low, high):
    poly_output = np.polyfit(x[low:high], y[low:high], 1, full = True)
    pfit_info   = poly_output[0]
    poly_resid  = poly_output[1][0]
    pfit_slope  = pfit_info[0]
    pfit_c      = pfit_info[1]

    return pfit_slope, pfit_c, poly_resid

def parse_cml(argv):

	"""
	Parses command line arguments
	"""

	## Create arguments class
	class cmd_args:

		"""
		Class for command line arguments
		"""

		def __init__(self, in_dir = None, out_dir = None, in_file = None, plotting = False, video = False, par = False):
			self.in_dir         = in_dir
			self.in_file        = out_dir
			self.plotting       = plotting
			self.video          = video
			self.parallel       = par
			self.num_threads    = 5
			self.summ           = False 
			self.field          = False 
			self.stream         = False 


	## Initialize class
	cargs = cmd_args()

	try:
		## Gather command line arguments
		opts, args = getopt.getopt(argv, "i:o:f:")
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

		if opt in ['-f']:
			## Read input directory
			cargs.in_file = str(arg)
			print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

	return cargs

if __name__ == "__main__":

	# -------------------------------------
	# # --------- Parse Commnad Line
	# -------------------------------------
	cmdargs = parse_cml(sys.argv[1:])

	sf_output_dir = cmdargs.out_dir + "SF/"
	if os.path.isdir(sf_output_dir) != True:
	    print("Making folder:" + tc.C + " SF/" + tc.Rst)
	    os.mkdir(sf_output_dir)

	# Construct input file
	input_file = cmdargs.in_dir + "/StructureFunctions/" + cmdargs.in_file

	with h5.File(input_file, "r") as in_f:
		dsets        = list(in_f.keys())
		num_pow      = len(dsets)
		Nr_x, Nr_y = in_f[dsets[0]][:, :].shape
		sf           = np.zeros((num_pow, Nr_x, Nr_y))
		for i, dset in enumerate(dsets):
			sf[i, :, :] = in_f[dset][:, :]

	num_snaps = 160
	
	# -------------------------------------
	# # --------- Plot Data
	# -------------------------------------
	Max_r = int(np.round(np.sqrt(Nr_x**2 + Nr_y**2)))

	powers = [0.1, 0.5, 1.0, 1.5, 2.0]

	sf_avg = np.zeros((num_pow, Max_r))
	fig    = plt.figure(figsize = (12, 8))
	gs     = GridSpec(1, 2, hspace = 0.35)
	ax1    = fig.add_subplot(gs[0, 0])
	for p in range(num_pow):
		r =  np.zeros((Max_r))
		for nr_x in range(Nr_x):
			for nr_y in range(Nr_y):
				r_indx = int(np.round(np.sqrt(nr_x**2 + nr_y**2)))

				if nr_x == 0 and nr_y == 0:
					continue
				else:
					sf_avg[p, r_indx] += sf[p, nr_x, nr_y] / num_snaps / (2.0 * np.pi * r_indx)

				r[r_indx] = r_indx

	# r * np.pi/Nr_x

	## Get slopes
	inert_lim_low  = 10
	inert_lim_high = 100
	slopes = np.zeros(num_pow)
	for p in range(num_pow):
		slopes[p], c, res = slope_fit(np.log2(r[1:]), np.log2(sf_avg[p, 1:]), 10, 100)
		print("p: {} - slope = {:1.4f}".format(p, slopes[p]))

	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(1, 2, hspace = 0.35)
	ax1 = fig.add_subplot(gs[0, 0])
	for p in range(num_pow):
		ax1.plot(np.log2(r[1:]), np.log2(sf_avg[p, 1:]), label=r"$p = {}$".format(powers[p]))
	ax1.set_xlim(1, 8)
	ax1.set_xlabel(r"$\log_2 r$")
	ax1.set_ylabel(r"$\log_2 \mathcal{S}_p^{\delta\omega}(r)$")
	ax1.set_title(r"Radial Vorticity Structure Functions")
	ax1.legend()

	p_range = powers
	ax1 = fig.add_subplot(gs[0, 1])
	ax1.plot(p_range, slopes / slopes[2], marker='.')
	ax1.plot(p_range, p_range, 'k--')
	ax1.set_xlim(0, 2.0)
	ax1.set_ylim(0, 2.0)
	ax1.set_xlabel(r"$p$")
	ax1.set_ylabel(r"$\zeta_{2p} / \zeta_2$")
	ax1.set_title("Anomalous Scaling")
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	plt.savefig(sf_output_dir + "Vort_SF.png", bbox_inches="tight")
	plt.close()