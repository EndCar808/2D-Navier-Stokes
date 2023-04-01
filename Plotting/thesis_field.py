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
from functions import tc, sim_data, import_data, import_post_processing_data, parse_cml, import_spectra_data, import_sys_msr
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

@njit
def compute_grad_vort(w_hat, kx, ky):

	ny, nxf = w_hat.shape

	grad_vort_hat = np.ones((ny, nxf, 2)) * 1j
	for i  in range(ny):
		for j in range(nxf):
			grad_vort_hat[i, j, 0] = 1j * ky[i] * w_hat[i, j]
			grad_vort_hat[i, j, 1] = 1j * kx[j] * w_hat[i, j]

	return grad_vort_hat

@njit
def compute_grad_vort_msr(grad_x, grad_y):

	ny, nx = grad_x.shape

	grad_vort = np.zeros((ny, nx))
	for i  in range(ny):
		for j in range(nx):
			grad_vort[i, j] = grad_x[i, j]**2 + grad_y[i, j]**2

	return grad_vort

# @njit
def compute_frac_msr(grad_vort):

	ny, nx = grad_vort.shape

	tot_grad_vort = np.sum(grad_vort)

	grad_vort_frac = np.zeros((ny, nx))
	for i  in range(ny):
		for j in range(nx):
			tmp_grad_vort = grad_vort[grad_vort <= grad_vort[i, j]] 
			grad_vort_frac[i, j] = np.sum(tmp_grad_vort) / tot_grad_vort

	return grad_vort_frac

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

	# ## Read in solver data
	# run_data = import_data(cmdargs.in_dir, sys_vars)
	with h5py.File(cmdargs.in_dir + "/Main_HDF_Data.h5", 'r') as in_f:

		## Initialize counter
		nn = 0

		# Get time
		time      = np.zeros((sys_vars.ndata, ))
		for group in in_f.keys():
		    time[nn] = in_f[group].attrs["TimeValue"]
		    nn += 1

		## Get the indexes of the peak enstrophy dissipation
		t1 = np.where(time == 59.35)[0][0]
		t2 = np.where(time == 80.30)[0][0] + 2
		print(t1, t2)
		time_len = len(time)

		## Vorticity 
		w     = np.zeros((2, sys_vars.Nx, sys_vars.Ny))
		w_hat = np.ones((2, sys_vars.Nx, sys_vars.Nk)) * 1j
		nn = 0
		for t in [int(t1), int(t2)]:
			group = "Iter_{:05d}".format(t)
			if 'w' in list(in_f[group].keys()):
			    w[nn, :, :] = in_f[group]["w"][:, :]
			if 'w_hat' in list(in_f[group].keys()):
			    w_hat[nn, :, :] = in_f[group]["w_hat"][:, :]
			    if 'w' not in list(in_f[group].keys()):
			        w[nn, :, :] = np.fft.irfft2(w_hat[nn, :, :])
			nn+=1

	# ## Read in system measures
	# sys_msr = import_sys_msr(cmdargs.in_dir, sys_vars)
	with h5py.File(cmdargs.in_dir + "/SystemMeasures_HDF_Data.h5", 'r') as in_f:
		if 'EnstrophyDissipation' in list(in_f.keys()):
		    enst_diss = in_f['EnstrophyDissipation'][:]
		    diss_len = len(enst_diss)
		    enst_diss = enst_diss[abs(diss_len - time_len):]
		if 'x' in list(in_f.keys()):
		    x  = in_f["x"][:]
		if 'y' in list(in_f.keys()):
		    y  = in_f["y"][:]
		if 'kx' in list(in_f.keys()):
		    kx  = in_f["kx"][:]
		if 'ky' in list(in_f.keys()):
		    ky  = in_f["ky"][:]

	# ## Read in spectra data
	# spec_data = import_spectra_data(cmdargs.in_dir, sys_vars)
	with h5py.File(cmdargs.in_dir + "/Spectra_HDF_Data.h5", 'r') as in_f:
		enst_spectrum_1d = np.zeros((sys_vars.ndata, sys_vars.spec_size))
		enrg_flux_spectrum = np.zeros((sys_vars.ndata, sys_vars.spec_size))
		enst_flux_spectrum = np.zeros((sys_vars.ndata, sys_vars.spec_size))
		## Initialze counter
		nn = 0

		# Read in the spectra
		for group in in_f.keys():
		    if "Iter" in group:
		        if 'EnergyFluxSpectrum' in list(in_f[group].keys()):
		            enrg_flux_spectrum[nn, :] = in_f[group]["EnergyFluxSpectrum"][:] ##* (sys_vars.Nx * sys_vars.Ny) / (4.0 * np.pi**2)
		        if 'EnstrophySpectrum' in list(in_f[group].keys()):
		            enst_spectrum_1d[nn, :] = in_f[group]["EnstrophySpectrum"][:] ##* (sys_vars.Nx * sys_vars.Ny) / (4.0 * np.pi**2)
		        if 'EnstrophyFluxSpectrum' in list(in_f[group].keys()):
		            enst_flux_spectrum[nn, :] = in_f[group]["EnstrophyFluxSpectrum"][:] ##* (sys_vars.Nx * sys_vars.Ny)  / (4.0 * np.pi**2)
		        nn+=1

	# ## Read in post processing data
	# post_data = import_post_processing_data(post_file_path, sys_vars, method)
	with h5py.File(post_file_path, 'r') as in_f:
		enst_spectrum = np.zeros((2, int(2 * sys_vars.Nx//3 + 1), int(2 * sys_vars.Nx//3 + 1)))
		nn=0
		for t in [int(t1), int(t2)]:
			group = "Snap_{:05d}".format(t)
			if 'FullFieldEnstrophySpectrum' in list(in_f[group].keys()):
			    enst_spectrum[nn, :] = in_f[group]["FullFieldEnstrophySpectrum"][:, :] ##* (sys_vars.Nx * sys_vars.Ny) / (4.0 * np.pi**2)
			nn+=1

	# -----------------------------------------
	# # --------  Make Output Folder
	# -----------------------------------------
	cmdargs.out_dir_thesis_field = cmdargs.out_dir + "THESIS_FIELD_PLOTS/"
	if os.path.isdir(cmdargs.out_dir_thesis_field) != True:
		print("Making folder:" + tc.C + " THESIS_FIELD_PLOTS/" + tc.Rst)
		os.mkdir(cmdargs.out_dir_thesis_field)


	# -----------------------------------------
	# # --------  Plot Data
	# -----------------------------------------
	file_format = ".pdf"
	lab_size    = 12


	# #----------------- Global parameters
	# Get the time indexes
	t1_w = 0
	t2_w = 1

	kmax = sys_vars.Nx//3
	k_range = np.arange(1, kmax)
	inert_range = np.arange(10, 100)

	plot_colours = plt.cm.YlGnBu(np.linspace(0.3, 1.0, 10))

	# my_cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1])
	my_cmap = mpl.colors.ListedColormap(cm.YlGnBu(np.arange(0,cm.YlGnBu.N)))
	my_cmap.set_under(color = "white")


	# #----------------- Plot bursting of Enstrophy
	fig = plt.figure()
	gs  = GridSpec(2, 2)
	ax1  = fig.add_subplot(gs[0, 0])
	im1  = ax1.imshow(enst_spectrum[t1_w, :, :], extent = (-sys_vars.Ny / 3 + 1, sys_vars.Ny / 3, -sys_vars.Nx / 3 + 1, sys_vars.Nx / 3), cmap = my_cmap, norm = mpl.colors.LogNorm()) # vmin = spec_lims[3], vmax = spec_lims[2]   # extent = (-sys_vars.Ny / 3 + 1, sys_vars.Ny / 3, -sys_vars.Nx / 3 + 1, sys_vars.Nx / 3), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), 
	ax1.set_xlabel(r"$k_x$", fontsize = lab_size)
	ax1.set_ylabel(r"$k_y$", fontsize = lab_size)
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$\mathcal{E}_k$", fontsize= lab_size)
	ax2  = fig.add_subplot(gs[0, 1])
	im2  = ax2.imshow(enst_spectrum[t2_w, :, :], extent = (-sys_vars.Ny / 3 + 1, sys_vars.Ny / 3, -sys_vars.Nx / 3 + 1, sys_vars.Nx / 3), cmap = my_cmap, norm = mpl.colors.LogNorm()) # vmin = spec_lims[3], vmax = spec_lims[2]   # extent = (-sys_vars.Ny / 3 + 1, sys_vars.Ny / 3, -sys_vars.Nx / 3 + 1, sys_vars.Nx / 3), cmap = mpl.colors.ListedColormap(cm.magma.colors[::-1]), 
	ax2.set_xlabel(r"$k_x$", fontsize = lab_size)
	ax2.set_ylabel(r"$k_y$", fontsize = lab_size)
	div2  = make_axes_locatable(ax2)
	cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
	cb2   = plt.colorbar(im2, cax = cbax2)
	cb2.set_label(r"$\mathcal{E}_k$", fontsize= lab_size)
	ax3  = fig.add_subplot(gs[1, :])
	# ax3.plot(time, enst_diss, label = r"$\varepsilon(t)$")
	enst_flux_50 = np.fliplr(np.fliplr(enst_flux_spectrum[:, 50:]).cumsum(axis=1))[:, 0] * 100
	enst_flux_100 = np.fliplr(np.fliplr(enst_flux_spectrum[:, 75:]).cumsum(axis=1))[:, 0] * 100
	ax3.plot(time, np.absolute(enst_flux_50), label = r"$\Pi_{50}(t)$", color = plot_colours[6])
	ax3.plot(time, np.absolute(enst_flux_100), label = r"$\Pi_{75}(t)$", color = plot_colours[1])
	print(enst_flux_50)
	print(enst_flux_100)
	ax3.axvline(x=time[t1], ymin=0, ymax=np.absolute(enst_flux_50[t1]), color = 'k', linestyle = ":")
	ax3.axvline(x=time[t2], ymin=0, ymax=np.absolute(enst_flux_50[t2]), color = 'k', linestyle = ":")
	# for i in [10, 50, 100, 150, 200]:
	# 	enst_flux_50 = np.fliplr(np.fliplr(enst_flux_spectrum[:, 1:i]).cumsum(axis=1))[:, 0] * 100
	# 	ax3.plot(time, np.absolute(enst_flux_50), label = r"$\Pi_{}(t)$".format(i))
	# ax3.legend()
	ax3.set_xlim(55, 85)
	# ax3.set_ylim(0, 40)
	ax3.set_xlabel(r"$t$",fontsize= lab_size)
	ax3.legend()
	# ax3.set_ylabel(r"$\varepsilon(t)$", fontsize= lab_size)
	ax3.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5, alpha = 0.8)
	# ax2.axvline(x, ymin)
	plt.savefig(cmdargs.out_dir_thesis_field + "Enstropy_Bursting" + file_format, bbox_inches='tight')
	plt.close()


	# #----------------- Time Averaged spectra
	fig = plt.figure(figsize = (textwidth, 0.4 * textwidth))
	gs  = GridSpec(1, 2)
	ax2 = fig.add_subplot(gs[0, 0])
	for i in range(enst_spectrum_1d.shape[0]):
		ax2.plot(k_range, enst_spectrum_1d[i, 1:kmax], color = plot_colours[0], alpha = 0.15)
	ax2.plot(k_range, np.mean(enst_spectrum_1d[:, 1:kmax], axis = 0), color = plot_colours[6])
	mean_enst_spec = np.mean(enst_spectrum_1d[:, 1:int(sys_vars.Nx/3)], axis = 0)
	p_enst = np.polyfit(np.log(k_range[inert_range]), np.log(mean_enst_spec[inert_range]), 1)
	ax2.plot(k_range[inert_range], np.exp(p_enst[1] + 2.5) * k_range[inert_range]**p_enst[0], '--', color=plot_colours[-1],label=r"$k^{-(1 + \xi)}$;" + r" $\xi = {:.2f}$".format(np.absolute(p_enst[0]) - 1))
	ax2.set_xlabel(r"$k$", fontsize = lab_size)
	ax2.set_ylabel(r"$\mathcal{E}_k$", fontsize = lab_size)
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_xlim(1, kmax)
	ax2.legend()
	ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	ax2 = fig.add_subplot(gs[0, 1])
	cum_sum_enst_spec = np.fliplr(np.fliplr(enst_flux_spectrum[:, 1:kmax]).cumsum(axis=1)) / np.mean(enst_diss) # integrate from k to infinity
	for i in range(enst_flux_spectrum.shape[0]):
		ax2.plot(k_range, cum_sum_enst_spec[i, :], color = plot_colours[0], alpha = 0.15)
	ax2.plot(k_range, np.mean(cum_sum_enst_spec[:, :], axis = 0), color = plot_colours[6])
	ax2.set_xlabel(r"$k$", fontsize = lab_size)
	ax2.set_ylabel(r"$\Pi_k$", fontsize = lab_size)
	ax2.set_xscale('log')
	ax2.set_xlim(1, kmax)
	ax2.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	# Add second axes
	x0     = 0.685 
	y0     = 0.65
	width  = 0.2
	height = 0.2
	ax_inset = fig.add_axes([x0, y0, width, height])
	ax_inset.plot(k_range, np.mean(cum_sum_enst_spec[:, :], axis = 0), color = plot_colours[6])
	p_flux = np.polyfit(np.log(k_range[inert_range]), np.log(np.mean(cum_sum_enst_spec[:, :], axis = 0)[inert_range]), 1)
	print(p_flux, np.mean(enst_diss))
	ax_inset.plot(k_range[inert_range], np.exp(p_flux[1] + 1.1) * k_range[inert_range]**p_flux[0], '--', color=plot_colours[-1],label=r"$k^{-\xi}$;" + r" $\xi = {:.2f}$".format(np.absolute(p_flux[0])))
	ax_inset.set_yscale('log')
	ax_inset.set_xscale('log')
	ax_inset.set_xlim(3, kmax)
	ax_inset.set_xlabel(r"$k$", fontsize = lab_size - 2)
	ax_inset.set_ylabel(r"$\langle \Pi_k \rangle_t$", fontsize = lab_size - 2)
	ax_inset.legend()
	ax_inset.grid(which = "both", axis = "both", color = 'k', linestyle = ":", linewidth = 0.5)
	plt.savefig(cmdargs.out_dir_thesis_field + "TimeAveraged_EnstrophySpectra" + file_format, bbox_inches='tight')
	plt.close()



	# #----------------- Plot Vorticity and Vorticity Gradient
	fig = plt.figure()
	gs  = GridSpec(2, 2) 
	print()
	ax1  = fig.add_subplot(gs[0, 0])
	im1 = ax1.imshow(w[t1_w, :, :] * (sys_vars.Nx * sys_vars.Ny), extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu")
	ax1.set_ylabel(r"$y$", fontsize=lab_size)
	ax1.set_xlabel(r"$x$", fontsize=lab_size)
	ax1.set_xlim(0.0, y[-1])
	ax1.set_ylim(0.0, x[-1])
	ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
	ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$\omega(x, y)$", fontsize=lab_size)
	ax2  = fig.add_subplot(gs[1, 0])
	im2 = ax2.imshow(w[t2_w, :, :] * (sys_vars.Nx * sys_vars.Ny), extent = (y[0], y[-1], x[-1], x[0]), cmap = "RdBu")
	ax2.set_ylabel(r"$y$", fontsize=lab_size)
	ax2.set_xlabel(r"$x$", fontsize=lab_size)
	ax2.set_xlim(0.0, y[-1])
	ax2.set_ylim(0.0, x[-1])
	ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
	ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
	ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	div2  = make_axes_locatable(ax2)
	cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
	cb2   = plt.colorbar(im2, cax = cbax2)
	cb2.set_label(r"$\omega(x, y)$", fontsize=lab_size)
	ax1  = fig.add_subplot(gs[0, 1])
	grad_vort_hat = compute_grad_vort(w_hat[t1_w, :, :], kx, ky)
	grad_x = np.fft.irfft2(grad_vort_hat[:, :, 0]) * (sys_vars.Nx * sys_vars.Ny)
	grad_y = np.fft.irfft2(grad_vort_hat[:, :, 1]) * (sys_vars.Nx * sys_vars.Ny)
	grad_sqr_msr = compute_grad_vort_msr(grad_x, grad_y)
	print("Computing Gradient Snap 1")
	grad_frac = compute_frac_msr(grad_sqr_msr)
	im1 = ax1.imshow(grad_frac, extent = (y[0], y[-1], x[-1], x[0]), cmap = "bone")
	# im1 = ax1.imshow(grad_sqr_msr / np.sum(grad_sqr_msr), extent = (y[0], y[-1], x[-1], x[0]), cmap = "bone")
	ax1.set_ylabel(r"$y$", fontsize=lab_size)
	ax1.set_xlabel(r"$x$", fontsize=lab_size)
	ax1.set_xlim(0.0, y[-1])
	ax1.set_ylim(0.0, x[-1])
	ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
	ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$|\nabla \omega|^2 / \sum |\nabla \omega|^2$", fontsize=lab_size)
	ax1  = fig.add_subplot(gs[1, 1])
	grad_vort_hat = compute_grad_vort(w_hat[t2_w, :, :], kx, ky)
	grad_x = np.fft.irfft2(grad_vort_hat[:, :, 0]) * (sys_vars.Nx * sys_vars.Ny)
	grad_y = np.fft.irfft2(grad_vort_hat[:, :, 1]) * (sys_vars.Nx * sys_vars.Ny)
	grad_sqr_msr = compute_grad_vort_msr(grad_x, grad_y)
	print("Computing Gradient Snap 2")
	grad_frac = compute_frac_msr(grad_sqr_msr)
	im1 = ax1.imshow(grad_frac, extent = (y[0], y[-1], x[-1], x[0]), cmap = "bone")
	# im1 = ax1.imshow(grad_sqr_msr / np.sum(grad_sqr_msr), extent = (y[0], y[-1], x[-1], x[0]), cmap = "bone")
	ax1.set_ylabel(r"$y$", fontsize=lab_size)
	ax1.set_xlabel(r"$x$", fontsize=lab_size)
	ax1.set_xlim(0.0, y[-1])
	ax1.set_ylim(0.0, x[-1])
	ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, y[-1]])
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax1.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, x[-1]])
	ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$|\nabla \omega|^2 / \sum |\nabla \omega|^2$", fontsize=lab_size)
	plt.savefig(cmdargs.out_dir_thesis_field + "Vorticity_and_VorticityGradient" + file_format, bbox_inches='tight')
	plt.close()

