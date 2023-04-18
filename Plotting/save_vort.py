import numpy as np
import h5py as h5



input_dir = "/home/enda/PhD/2D-Navier-Stokes/Data/Tmp/NAV_AB4_FULL_N[1024,1024]_T[0.0,0.0005,20.000]_NU[0,0,0.0]_DRAG[0,0,0.0]_FORC[NONE,2,1]_u0[DOUBLE_SHEAR_LAYER]_TAG[DBSL]/"

with h5.File(input_dir + "Main_HDF_Data.h5", 'r') as infile:

	Time = infile["Time"][:]
	num_snaps = len(Time)
	n = 0
	for i, t in enumerate(Time):
		if t >= 7.0 and t <= 9.0:
			n += 1

	w = np.zeros((n, 1024, 1024))
	n = 0
	for i, t in enumerate(Time):
		if t >= 7.0 and t <= 9.0:
			print(i, t) 
			if 'w_hat' in list(infile['Iter_00{}'.format(int(i))].keys()):
				w_hat = infile['Iter_00{}'.format(int(i))]['w_hat'][:, :]
				w[n, :, :] = np.fft.irfft2(w_hat) * (1024**2)
				n += 1


with h5.File(input_dir + "Vorticity.h5", 'w') as infile:
	infile.create_dataset("w", data=w)
