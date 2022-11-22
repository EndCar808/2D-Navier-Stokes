#!/usr/bin/env python3
import numpy as np
import pyfftw as fftw
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig


Nx = 4
Ny = 4
Nxf = Nx // 2 + 1

gamma = 1

amp   = np.zeros((Ny, Nxf))
phi   = np.zeros((Ny, Nxf))
w_hat = np.zeros((Nx, Nxf), dtype = 'complex128')
mask  = np.zeros((Nx, Nxf))
kx    = np.arange(0, Nxf)
ky    = np.append(np.arange(0, Nxf), np.arange(-Ny//2 + 1, 0))


## Generate the field
for i in range(Ny):
	for j in range(Nxf):
		## Compute |k|
		k_sqr = np.sqrt(ky[i]**2 + kx[j]**2)

		## Fill the amplitudes
		if ky[i] != 0.0 or kx[j] != 0.0:
			amp[i, j] = 1.0 / k_sqr

		## Fill the phases
		if kx[j] == 0 and ky[i] < 0: 
			phi[i, j] = -np.pi/4
		else:
			phi[i, j] = np.pi/4

		## Fill the vorticity
		w_hat[i, j] = amp[i, j] * np.exp(1j * phi[i, j])

		## Fill the dealiasing mask
		if k_sqr**2 <= 0.0 or k_sqr**2 > (Nx//3)**2:
			mask[i, j] = 0.0
		else:
			mask[i, j] = 1.0

## Amply the 2/3'rds dealiasing
# print(w_hat)
# w_hat *= mask
# print(w_hat)

w_hat_conv_direct = sig.convolve2d(w_hat, w_hat, mode = 'same')
w_hat_conv_fft    = sig.fftconvolve(w_hat, w_hat, mode = 'same')
print(np.allclose(w_hat_conv_fft, w_hat_conv_fft))
print(w_hat.shape, w_hat_conv_fft.shape, w_hat_conv_direct.shape)


for i in range(Ny):
	for j in range(Nxf):
		print("[{},{}]: {:1.15f} {:1.15f}i\t".format(ky[i], kx[j], np.real(w_hat[i, j]), np.imag(w_hat[i, j])), end = "")
	print()

convolution = np.zeros((Ny, Nxf), dtype = "complex128")

# for k_y_tmp in range(Ny):
# 	k_y = k_y_tmp - Ny//2 + 1
# 	for k_x_tmp in range(Nxf):
# 		k_x = k_x_tmp

# 		for k1_y_tmp in range(Ny):
# 			k1_y = k1_y_tmp - Ny//2 + 1
# 			for k1_x_tmp in range(Nx):
# 				k1_x = k1_x_tmp -Nx//2 + 1

# 				if k1_y < 0 and k1_x < 0 and k_y - k1_y >= 0 and k_x - k1_x >= 0: 
# 					convolution[k_y_tmp, k_x_tmp] += np.conjugate(w_hat[np.absolute(k1_y), np.absolute(k1_x)]) * w_hat[k_y - k1_y, k_x - k1_x]


# print(convolution)
# print(np.allclose(convolution, w_hat_conv_fft))