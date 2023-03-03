import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def slope_fit(x, y, low, high):

    poly_output = np.polyfit(x[low:high], y[low:high], 1, full = True)
    pfit_info   = poly_output[0]
    poly_resid  = poly_output[1][0]
    pfit_slope  = pfit_info[0]
    pfit_c      = pfit_info[1]

    return pfit_slope, pfit_c, poly_resid



if __name__ == "__main__":
	
	Nx          = sys.argv[1]
	num_threads = sys.argv[2]

	with h5.File("Test_Str_Func_Data_N{}_T{}.h5".format(Nx, num_threads), "r") as in_f:

		# print(list(in_f.keys()))

		# Run data
		run_data = in_f["RunData"][:]
		Nx = run_data[0]
		Ny = run_data[1]
		num_threads = run_data[2]
		num_pow = run_data[3]

		# Str func data
		ser_rad_vort_str_func = in_f["/Ser_Rad_Vorticity_StrFunc"][:, :, :]
		ser_vel_str_func = in_f["/Ser_Velocity_StrFunc"][:, :]
	#--------------------------------------
	# Conventional Velocity Str Function
	#--------------------------------------
	# Generate analytical solution
	# Max_r = int(np.ceil(np.sqrt((Nx/2)**2 + (Ny/2)**2)) + 1)
	Max_r = ser_vel_str_func.shape[1]
	r = np.zeros(Max_r)
	r2 = np.zeros(Max_r)
	r3 = np.zeros(Max_r)
	for i in range(Max_r):
		r_x  = i 
		r_y  = i 
		# r[i] = np.sqrt((r_x**2 + r_y**2))
		r[i] = (r_x + r_y)
		r2[i] = (r_x**2 + r_y**2)
		r3[i] = (r_x**3 + r_y**3)

	lim_low = Max_r//4
	lim_high = 3 * Max_r//4

	# Second order with fit
	plt.figure()
	# plt.plot(r, r2, 'k--', label = r"$r^2$")
	plt.plot(r, ser_vel_str_func[1, :], label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_2(r)$")
	slope, c, _ = slope_fit(r, ser_vel_str_func[1, :], lim_low, lim_high)
	plt.plot(r, r**slope + c, 'k--', label = r"$slope = {}$".format(np.around(slope, 4)))
	plt.xlabel(r"$r$")
	plt.ylabel(r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_p$")
	plt.yscale('log')
	# plt.xscale('log')
	plt.legend()
	plt.savefig("LongVel_StrFunc_2ndOrder.png")
	plt.close()

	# Third order with fit
	plt.figure()
	# plt.plot(r, r3, 'k--', label = r"$r^3$")
	plt.plot(r, ser_vel_str_func[1, :], label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_2(r)$")
	slope, c, _ = slope_fit(r, ser_vel_str_func[2, :], lim_low, lim_high)
	plt.plot(r, r**slope + c, 'k--', label = r"$slope = {}$".format(np.around(slope, 4)))
	plt.xlabel(r"$r$")
	plt.ylabel(r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_p$")
	plt.yscale('log')
	# plt.xscale('log')
	plt.legend()
	plt.savefig("LongVel_StrFunc_3rdOrder.png")
	plt.close()

	## Combined plot
	plt.figure()
	# 2nd order
	plt.plot(r, r2, 'k--', label = r"$r^2$")
	plt.plot(r, ser_vel_str_func[1, :], label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_2(r)$")
	# 3rd order
	plt.plot(r, r3, 'k--', label = r"$r^3$")
	plt.plot(r, ser_vel_str_func[2, :], label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_3(r)$")
	plt.xlabel(r"$r$")
	plt.ylabel(r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_p$")
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.savefig("LongVel_StrFunc_TestCombined.png")
	plt.close()

	# plt.plot(ser_vel_str_func)



	#--------------------------------------
	# Radial Voriticty Str Function
	#--------------------------------------
	# Generate analytical solution
	# Max_r = int(np.ceil(np.sqrt((Nx)**2 + (Ny)**2)) + 1)
	Max_vort_r = int(np.ceil(np.sqrt((Nx - 1)**2 + (Ny - 1)**2)))
	# Max_vort_r = int(np.round(np.sqrt((ser_rad_vort_str_func.shape[1])**2 + (ser_rad_vort_str_func.shape[2])**2)))
	# Max_r = int(np.round(np.sqrt((ser_rad_vort_str_func.shape[1])**2 + (ser_rad_vort_str_func.shape[2])**2)) + 1)
	r = np.zeros(Max_vort_r)
	r2 = np.zeros(Max_vort_r)
	r3 = np.zeros(Max_vort_r)
	for i in range(Max_vort_r):
		r_x  = i / (Nx - 1)
		r_y  = i / (Ny - 1)
		r[i] = np.sqrt(2) * i / (2 * Max_vort_r)
		# r[i] = np.sqrt((r_x**2 + r_y**2)) 
		# r[i] = np.sqrt((r_x**2 + r_y**2))

		# r_ind = int(np.round(np.sqrt(i**2 + j**2)))
		# r[r_ind] = np.sqrt(r_x**2 + r_y**2)

	r2 = r**2
	r3 = r**3

	
	# Shell average over vorticity increments
	shell_avg_vort_str_func = np.zeros((num_pow, Max_vort_r))
	shell_counts            = np.zeros((Max_vort_r, ))
	for i in range(ser_rad_vort_str_func.shape[1]):
		for j in range(ser_rad_vort_str_func.shape[2]):
			# Get shell index
			r_ind = int(np.ceil(np.sqrt(i**2 + j**2)))

			for p in range(num_pow):
				shell_avg_vort_str_func[p, r_ind] += ser_rad_vort_str_func[p, i, j]

			# Update count
			shell_counts[r_ind] += 1

	# Normalize shell average
	for p in range(num_pow):
		shell_avg_vort_str_func[p, :] /= shell_counts[:]

	## Combined plot
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(1, 2)
	# 2nd order
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(r[:Max_vort_r], r2[:Max_vort_r], 'k--', label = r"$r^2$")
	ax1.plot(r[:Max_vort_r], shell_avg_vort_str_func[1, :], label = r"$\mathcal{S}^{\omega}_2(r)$")
	ax1.set_xlabel(r"$r$")
	ax1.set_ylabel(r"$\mathcal{S}^{\omega}_p$")
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	ax1.legend()
	# 3rd order
	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(r[:Max_vort_r], r3[:Max_vort_r], 'k--', label = r"$r^3$")
	ax2.plot(r[:Max_vort_r], shell_avg_vort_str_func[2, :], label = r"$\mathcal{S}^{\omega}_3(r)$")
	ax2.set_xlabel(r"$r$")
	ax2.set_ylabel(r"$\mathcal{S}^{\omega}_p$")
	ax2.set_yscale('log')
	ax2.set_xscale('log')
	ax2.legend()
	plt.savefig("ShellAveragedVorticity_StrFunc_1D_TestCombined.png")
	plt.close()


	# Imshow comparison
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0, 0])
	im1 = ax1.imshow(ser_rad_vort_str_func[1, :, :], extent = (0.0, 0.5, 0.5, 0.0), cmap = "jet")
	ax1.set_xlabel(r"$r_x$")
	ax1.set_ylabel(r"$r_y$")
	ax1.set_xlim(0.0, 0.5)
	ax1.set_ylim(0.0, 0.5)
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$\mathcal{S}_2^{\omega}(r)$")
	ax1.set_title(r"2nd Order Vorticity Structure Function")

	nx = ser_rad_vort_str_func[0, :, :].shape[0]
	rx = np.zeros((nx,))
	ry = np.zeros((nx,))
	for i in range(nx):
		rx[i] = i / (Nx - 1) # These are the rx and ry I defined in the C code
		ry[i] = i / (Ny - 1)
	r_x, r_y = np.meshgrid(rx, ry)
	rr = (r_x + r_y)**2
	rel_err = np.absolute(rr - ser_rad_vort_str_func[1, :, :]) 
	abs_err = rel_err / (rr + 1e-16)
	ax2 = fig.add_subplot(gs[0, 1])
	im2 = ax2.imshow(abs_err,  extent = (0.0, 0.5, 0.5, 0.0), cmap = "jet")
	ax2.set_xlabel(r"$r_x$")
	ax2.set_ylabel(r"$r_y$")
	ax2.set_xlim(0.0, 0.5)
	ax2.set_ylim(0.0, 0.5)
	div2  = make_axes_locatable(ax2)
	cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
	cb2   = plt.colorbar(im2, cax = cbax2)
	ax2.set_title(r"Absolute Error from $(r_x + r_y)^2$")

	plt.savefig("ShellAveragedVorticity_StrFunc_2DImshow2ndOrder.png")
	plt.close()