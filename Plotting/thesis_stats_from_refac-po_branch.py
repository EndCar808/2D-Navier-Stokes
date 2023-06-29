#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Script to plot the weighted PDFs of triad phase types

#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import getopt
from matplotlib.pyplot import cm
from functions import tc, sim_data, import_data, import_post_processing_data, parse_cml, compute_pdf
## Choice of Back end
mpl.use('Agg') # Use this backend for writing plots to file
## Choosing latex as the font.
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
mpl.rcParams['font.size']   = 10.0 # def: 10.0
## Lines
mpl.rcParams['lines.linewidth']  = 1.0 # def: 1.5
mpl.rcParams['lines.markersize'] = 5.0 # def: 6
mpl.rcParams['lines.markersize'] = 5.0 # def: 6
## Grid lines
mpl.rcParams['grid.color']     = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha']     = 0.8
##Ticks
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top']       = True
mpl.rcParams['ytick.right']     = True
## Figsize
textwidth = 12.7
mpl.rcParams['figure.figsize'] = [textwidth, 0.75 * textwidth]



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

	## Read in post processing data
	# post_data = import_post_processing_data(post_file_path, sys_vars, method)
	with h5py.File(post_file_path, 'r') as in_f:
		# Velcoticity Structure Functions
		if 'AbsoluteVelocityLongitudinalStructureFunctions' in list(in_f.keys()):
		    vel_long_str_func_abs = in_f["AbsoluteVelocityLongitudinalStructureFunctions"][:, :] / sys_vars.ndata
		# Get the vorticity structure functions
		if 'AbsoluteVorticityLongitudinalStructureFunctions' in list(in_f.keys()):
		    vort_long_str_func_abs = in_f["AbsoluteVorticityLongitudinalStructureFunctions"][:, :] / sys_vars.ndata

		# Get the Velocity increment PDFs
		if 'LongitudinalVelIncrements_BinRanges' in list(in_f.keys()):
		    vel_long_incr_ranges = in_f["LongitudinalVelIncrements_BinRanges"][:, :]
		if 'LongitudinalVelIncrements_BinCounts' in list(in_f.keys()):
		    vel_long_incr_counts = in_f["LongitudinalVelIncrements_BinCounts"][:, :]
		# Get the Vorticity Increment PDFs
		if 'LongitudinalVortIncrements_BinRanges' in list(in_f.keys()):
		    vort_long_incr_ranges = in_f["LongitudinalVortIncrements_BinRanges"][:, :]
		if 'LongitudinalVortIncrements_BinCounts' in list(in_f.keys()):
		    vort_long_incr_counts = in_f["LongitudinalVortIncrements_BinCounts"][:, :]
		# Vorticity gradient PDFs
		if 'VorticityGradient_x_BinRanges' in list(in_f.keys()):
		    grad_w_x_ranges = in_f["VorticityGradient_x_BinRanges"][:]
		if 'VorticityGradient_x_BinCounts' in list(in_f.keys()):
		    grad_w_x_counts = in_f["VorticityGradient_x_BinCounts"][:]
		# Velocity gradient PDFs
		if 'VelocityGradient_x_BinRanges' in list(in_f.keys()):
		    grad_u_x_ranges = in_f["VelocityGradient_x_BinRanges"][:]
		if 'VelocityGradient_x_BinCounts' in list(in_f.keys()):
		    grad_u_x_counts = in_f["VelocityGradient_x_BinCounts"][:]


	# -----------------------------------------
	# # --------  Make Output Folder
	# -----------------------------------------
	cmdargs.out_dir_thesis_stats = cmdargs.out_dir + "THESIS_STATS_PLOTS/"
	if os.path.isdir(cmdargs.out_dir_thesis_stats) != True:
		print("Making folder:" + tc.C + " THESIS_STATS_PLOTS/" + tc.Rst)
		os.mkdir(cmdargs.out_dir_thesis_stats)


	# -----------------------------------------
	# # --------  Plot Data
	# -----------------------------------------
	file_format = ".pdf"
	lab_size    = 12

	# #----------------- Global parameters
	num_pow        = 5
	num_pdfs       = vel_long_incr_ranges.shape[0]
	powers         = np.asarray([0.1, 0.5, 1.0, 1.5, 2.0])
	log_func       = np.log2
	inert_lim_low  = 8
	inert_lim_high = 48

	r                = np.arange(1.0, sys_vars.Nx//2 + 1)
	str_func_colours = plt.cm.YlGnBu(np.linspace(0.3, 1.0, num_pow))
	pdf_colours      = plt.cm.YlGnBu(np.linspace(0.3, 1.0, num_pdfs))

	plot_legend = [r"$r_0 = 1 \Delta x$", r"$r_1 = 2 \Delta x$", r"$r_2 = 4 \Delta x$", r"$r_3 = 8 \Delta x$",  r"$r_{max} = \pi$"]


	# #------------------- Structure Functions & Scaling
	vort_zeta_p, vort_zeta_p_resid = [], []
	vel_zeta_p, vel_zeta_p_resid   = [], []
	fig = plt.figure()
	gs  = GridSpec(2, 2)
	ax1 = fig.add_subplot(gs[0, 0])
	for i in range(num_pow):
		p, = ax1.plot(log_func(r), log_func(vort_long_str_func_abs[i, :]), label = "$p = {}$".format(powers[i]), color = str_func_colours[i])
		poly_output = np.polyfit(log_func(r[inert_lim_low:inert_lim_high]), log_func(vort_long_str_func_abs[i, inert_lim_low:inert_lim_high]), 1, full = True)
		pfit_info   = poly_output[0]
		poly_resid  = poly_output[1][0]
		pfit_slope  = pfit_info[0]
		pfit_c      = pfit_info[1]
		vort_zeta_p.append(np.absolute(pfit_slope))
		vort_zeta_p_resid.append(poly_resid)
		print(" {}\t {:1.4f} \t {:1.4f} +/-{:0.3f}".format(i + 1, powers[i], pfit_slope, poly_resid))
		ax1.plot(log_func(r[inert_lim_low:inert_lim_high]), log_func(r[inert_lim_low:inert_lim_high])*pfit_slope + pfit_c + 0.25, '--', color = p.get_color())
	ax1.set_xlim(log_func(r[0]), log_func(r[-1]))
	ax1.set_xlabel(r"$\log_2 r$", fontsize = lab_size)
	ax1.set_ylabel(r"$\log_2 \mathcal{S}_{2p}^{\omega}(r)$", fontsize = lab_size)
	ax1.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5, alpha = 0.8)
	ax1.legend()
	ax2 = fig.add_subplot(gs[0, 1])
	for i in range(num_pow):
		p, = ax2.plot(log_func(r), log_func(vel_long_str_func_abs[i, :]), label = "$p = {}$".format(powers[i]), color = str_func_colours[i])
		poly_output = np.polyfit(log_func(r[inert_lim_low:inert_lim_high]), log_func(vel_long_str_func_abs[i, inert_lim_low:inert_lim_high]), 1, full = True)
		pfit_info   = poly_output[0]
		poly_resid  = poly_output[1][0]
		pfit_slope  = pfit_info[0]
		pfit_c      = pfit_info[1]
		vel_zeta_p.append(np.absolute(pfit_slope))
		vel_zeta_p_resid.append(poly_resid)
		print(" {}\t {:1.4f} \t {:1.4f} +/-{:0.3f}".format(i + 1, powers[i], pfit_slope, poly_resid))
		ax2.plot(log_func(r[inert_lim_low:inert_lim_high]), log_func(r[inert_lim_low:inert_lim_high])*pfit_slope + pfit_c + 0.25, '--', color = p.get_color())
	ax2.set_xlim(log_func(r[0]), log_func(r[-1]))
	ax2.set_xlabel(r"$\log_2 r$", fontsize = lab_size)
	ax2.set_ylabel(r"$\log_2 \mathcal{S}_{2p}^{\mathbf{u}}(r)$", fontsize = lab_size)
	ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5, alpha = 0.8)
	ax2.legend()
	ax3 = fig.add_subplot(gs[1, 0])
	ax3.plot(powers, powers, 'k--', label = r"$p$")
	ax3.errorbar(powers, vort_zeta_p / vort_zeta_p[2], yerr = vort_zeta_p_resid, marker='.', color = str_func_colours[2])
	ax3.set_xlim(powers[0] - 0.05, powers[-1] + 0.05)
	ax3.set_ylim(powers[0] - 0.05, powers[-1] + 0.05)
	ax3.set_xlabel(r"$p$")
	ax3.set_ylabel(r"$\zeta_{2p}^{\omega} / \zeta_2^{\omega}$")
	ax3.legend()
	ax4 = fig.add_subplot(gs[1, 1])
	ax4.plot(powers, powers, 'k--', label = r"$p$")
	ax4.errorbar(powers, vel_zeta_p / vel_zeta_p[2], yerr = vel_zeta_p_resid, marker='.', color = str_func_colours[2])
	ax4.set_xlim(powers[0] - 0.05, powers[-1] + 0.05)
	ax4.set_ylim(powers[0] - 0.05, powers[-1] + 0.05)
	ax4.set_xlabel(r"$p$")
	ax4.set_ylabel(r"$\zeta_{2p}^{\mathbf{u}} / \zeta_2^{\mathbf{u}}$")
	ax4.legend()
	plt.savefig(cmdargs.out_dir_thesis_stats + "StructureFunctions_Both_Full" + file_format, bbox_inches='tight')
	plt.close()



	# #------------------- Increment PDFs
	fig = plt.figure()
	gs  = GridSpec(1, 2) 
	ax1 = fig.add_subplot(gs[0, 0])
	for i in range(num_pdfs):
		bin_centres, pdf = compute_pdf(vel_long_incr_ranges[i, :], vel_long_incr_counts[i, :], normalized = True)
		ax1.plot(bin_centres, pdf, label = plot_legend[i], color = pdf_colours[i])
	ax1.set_xlabel(r"$\delta_r \mathbf{u}_{\parallel} / \langle (\delta_r \mathbf{u}_{\parallel})^2 \rangle^{1/2}$", fontsize=lab_size)
	ax1.set_ylabel(r"PDF", fontsize=lab_size)
	ax1.set_yscale('log')
	ax1.grid(which = "both", axis = "both", color = 'k', linewidth = .5, linestyle = ':')
	ax1.legend()
	ax2 = fig.add_subplot(gs[0, 1])
	for i in range(num_pdfs):
		bin_centres, pdf = compute_pdf(vort_long_incr_ranges[i, :], vort_long_incr_counts[i, :], normalized = True)        
		ax2.plot(bin_centres, pdf, label = plot_legend[i], color = pdf_colours[i])
	ax2.set_xlabel(r"$\delta_r \omega / \langle (\delta_r \omega)^2 \rangle^{1/2}$", fontsize=lab_size)
	ax2.set_ylabel(r"PDF", fontsize=lab_size)
	ax2.set_yscale('log')
	ax2.grid(which = "both", axis = "both", color = 'k', linewidth = .5, linestyle = ':')
	ax2.legend()    
	plt.savefig(cmdargs.out_dir_thesis_stats + "/Longitudinal_Incrmenents_PDFs" + file_format, bbox_inches='tight')
	plt.close()