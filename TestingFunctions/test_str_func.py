import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys


Nx          = sys.argv[1]
num_threads = sys.argv[2]

with h5.File("Test_Str_Func_Data_N{}_T{}.h5".format(Nx, num_threads), "r") as in_f:

	# print(list(in_f.keys()))

	# Run data
	run_data = in_f["RunData"][:]
	Nx = run_data[0]
	Ny = run_data[1]
	num_threads = run_data[2]

	# Str func data
	ser_rad_vort_str_func = in_f["/Ser_Rad_Vorticity_StrFunc"][:, :, :]
	ser_vel_str_func = in_f["/Ser_Velocity_StrFunc"][:, :]


# Generate analytical solution
# Max_r = int(np.ceil(np.sqrt((Nx/2)**2 + (Ny/2)**2)) + 1)
Max_r = ser_vel_str_func.shape[1]
r = np.zeros(Max_r)
r2 = np.zeros(Max_r)
r3 = np.zeros(Max_r)
for i in range(Max_r):
	r_x  = i / (Nx) 
	r_y  = i / (Nx)
	# r[i] = np.sqrt((r_x**2 + r_y**2))
	r[i] = (r_x + r_y)
	r2[i] = (r_x**2 + r_y**2)
	r3[i] = (r_x**3 + r_y**3)


plt.figure()
# 2nd order
plt.plot(r, r2, 'k--', label = r"$r^2$")
plt.plot(r, ser_vel_str_func[1, :], label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_2(r)$")
plt.plot(r, r3, 'k--', label = r"$r^3$")
plt.plot(r, ser_vel_str_func[2, :], label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_3(r)$")
plt.xlabel(r"$r$")
plt.ylabel(r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_p$")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("LongVel_StrFunc.png")
plt.close()

# plt.plot(ser_vel_str_func)

