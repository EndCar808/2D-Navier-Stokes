import numpy as np



def Dft_2d(real, Nx, Ny):

    comp = np.ones((Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)

    for kx in range(Nx):
        # kkx = kx - Nx / 2
        for ky in range(int(Ny / 2 + 1)):
            for n in range(Nx):
                for m in range(Ny):
                    comp[kx, ky] += real[n, m] * np.exp(np.complex(0.0 -1.0) * (kx * (2 * np.pi * n / Nx) + ky * (2 * np.pi * m / Ny))) 

    return comp / (Nx * Ny)


def Dealias(data, kx, ky, Nx, Ny):

    Nx3 = int(np.ceil(Nx / 3))
    Ny3 = int(np.ceil(Ny / 3))

    for i in range(Nx):
        for j in range(int(Ny/ 2 + 1)):
            if np.absolute(kx[i]) < Nx3 and np.absolute(ky[j]) < Ny3:
                data[i, j] = data[i, j]
            else:
                data[i, j] = np.complex(0.0, 0.0)

    return data 

Nx = 8
Ny = 8

x  = np.zeros((Nx, ))
y  = np.zeros((Ny, ))
kx = np.zeros((Nx, ))
ky = np.zeros((Ny, ))
for i in range(Nx):
    if i <= Nx /2:
        kx[i] = i
    else:
        kx[i] = i - Nx
    x[i] = i * 2 * np.pi / Nx
for i in range(Nx):
    if i <= Nx /2:
        ky[i] = i
    else:
        ky[i] = i - Nx
    y[i] = i * 2 * np.pi / Nx

# for i in range(Nx):
#     if i < int(Ny /2  + 1):
#         print("x[{}]: {:0.16f}\ty[{}]: {:0.16f}\tkx[{}]:{}\tky[{}]:{}".format(i, x[i], i, y[i], i, kx[i], i, ky[i]))
#     else:
#         print("x[{}]: {:0.16f}\ty[{}]: {:0.16f}\tkx[{}]:{}".format(i, x[i], i, y[i], i, kx[i]))
# print()
# print()


## Fill Real Space velocities
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        u[i, j] = np.cos(x[i]) * np.sin(y[j])
#         print("u0[{}]: {:+0.16f}\t".format(i * Ny + j, np.cos(x[i]) * np.sin(y[j])), end = "")
#     print()
# print()
# print()
for i in range(Nx):
    for j in range(Ny):
        v[i, j] =  -np.sin(x[i]) * np.cos(y[j])
#         print("v0[{}]: {:+0.16f}\t".format(i * Ny + j, -np.sin(x[i]) * np.cos(y[j])), end = "")
#     print()
# print()
# print()
# print()
# print()

## Transform both to Fourier Space
u_hat = np.fft.rfft2(u)
v_hat = np.fft.rfft2(v)
# print(u_hat)
# print(v_hat)
# for i in range(Nx):
#     for j in range(int(Ny/ 2 + 1)):
#         print("uh[{}]: {:+0.16f} {:+0.16f} I\t".format(i * int(Ny/ 2 + 1) + j, np.real(u_hat[i, j] ), np.imag(u_hat[i, j] )), end = "")
#     print()
# print()
# print()
# for i in range(Nx):
#     for j in range(int(Ny/ 2 + 1)):
#         print("vh[{}]: {:+0.16f} {:+0.16f} I\t".format(i * int(Ny/ 2 + 1) + j, np.real(v_hat[i, j] ), np.imag(v_hat[i, j] )), end = "")
#     print()
# print()
# print()

# u_t = np.fft.irfft2(u_hat)
# v_t = np.fft.irfft2(v_hat)
# for i in range(Nx):
#     for j in range(int(Ny)):
#         print("ut[{}]: {:+0.16f}\t".format(i * int(Ny/ 2 + 1) + j, u_t[i, j]), end = "")
#     print()
# print()
# print()
# for i in range(Nx):
#     for j in range(int(Ny)):
#         print("vt[{}]: {:+0.16f}\t".format(i * int(Ny/ 2 + 1) + j, v_t[i, j]), end = "")
#     print()
# print()
# print()

w_hat = np.ones((Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)
for i in range(Nx):
    for j in range(int(Ny / 2 + 1)):
        w_hat[i, j] = np.complex(0.0, 1.0) * (kx[i] * v_hat[i, j]  - ky[j] * u_hat[i, j])
#         print("wh[{}]: {:+0.16f} {:+0.16f} I\t".format(i * int(Ny/ 2 + 1) + j, np.real(w_hat[i, j] ), np.imag(w_hat[i, j] )), end = "")
#     print()
# print()
# print()


Dealias(w_hat, kx, ky, Nx, Ny)


w = np.zeros((Nx, int(Ny)))
for i in range(Nx):
    for j in range(int(Ny)):
        w[i, j] = 2 * np.sin(x[i]) * np.sin(x[j])
w_hat_tmp = np.fft.rfft2(w)

# for i in range(Nx):
#     for j in range(int(Ny / 2 + 1)):
#         print("wh[{}]: {:+0.16f} {:+0.16f} I\t".format(i * int(Ny/ 2 + 1) + j, np.real(w_hat_tmp[i, j] ), np.imag(w_hat_tmp[i, j] )), end = "")
#     print()
# print()
# print()

for i in range(Nx):
    for j in range(int(Ny / 2 + 1)):
        k_sqr = kx[i] * kx[i] + ky[j] * ky[j] + 1e-50
        u_hat[i, j] = np.complex(0.0, 1.0) * ky[j] * w_hat[i, j] / k_sqr 
        v_hat[i, j] = np.complex(0.0, -1.0) * kx[i] * w_hat[i, j] / k_sqr


u_tmp = np.fft.irfft2(u_hat) * (Nx * Ny)
v_tmp = np.fft.irfft2(v_hat) * (Nx * Ny)
for i in range(Nx):
    for j in range(int(Ny)):
        print("u0[{}]: {:+0.16f}\t".format(i * int(Ny) + j, u_tmp[i, j]), end = "")
    print()
print()
print()
for i in range(Nx):
    for j in range(int(Ny)):
        print("v0[{}]: {:+0.16f}\t".format(i * int(Ny) + j, u_tmp[i, j]), end = "")
    print()
print()
print()

dwhat_dx = np.ones((Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)
dwhat_dy = np.ones((Nx, int(Ny / 2 + 1))) * np.complex(0.0, 0.0)
for i in range(Nx):
    for j in range(int(Ny / 2 + 1)):
        dwhat_dx[i, j] = np.complex(0.0, 1.0) * kx[i] * w_hat[i, j]
        dwhat_dy[i, j] = np.complex(0.0, 1.0) * ky[j] * w_hat[i, j]


dw_dx = np.fft.irfft2(dwhat_dx) * (Nx * Ny)
dw_dy = np.fft.irfft2(dwhat_dy) * (Nx * Ny)
for i in range(Nx):
    for j in range(int(Ny)):
        print("whx[{}]: {:+0.16f}\t".format(i * int(Ny/ 2 + 1) + j, dw_dx[i, j]), end = "")
    print()
print()
print()
for i in range(Nx):
    for j in range(int(Ny)):
        print("why[{}]: {:+0.16f}\t".format(i * int(Ny/ 2 + 1) + j, dw_dy[i, j]), end = "")
    print()
print()
print()

dw_dt = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        dw_dt[i, j] = -1.0 * ((u_tmp[i, j] * dw_dx[i, j]) + (v_tmp[i, j] * dw_dy[i, j]))
for i in range(Nx):
    for j in range(int(Ny)):
        print("dwdt[{}]: {:+0.16f}\t".format(i * int(Ny/ 2 + 1) + j, dw_dt[i, j]), end = "")
    print()
print()
print()
dw_hat_dt = np.fft.rfft2(dw_dt) / (Nx * Ny)
dw_hat_dt = Dealias(dw_hat_dt, kx, ky, Nx, Ny)
for i in range(Nx):
    for j in range(int(Ny/ 2 + 1)):
        print("dwhdt[{}]: {:+0.16f} {:+0.16f}I\t".format(i * int(Ny/ 2 + 1) + j, np.real(dw_hat_dt[i, j] ), np.imag(dw_hat_dt[i, j] )), end = "")
    print()
print()
print()












# u_hat1 = Dft_2d(u, Nx, Ny)
# print(u_hat1)
# print()
# for i in range(Nx):
#     for j in range(int(Ny/ 2 + 1)):
#         print("uh1[{}]: {:0.10f} {:0.10f} I\t".format(i * int(Ny/ 2 + 1) + j, np.real(u_hat1[i, j] ), np.imag(u_hat1[i, j] )), end = "")
#     print()
# print()
# print()




# ## Fill Fourier space vorticity
# w_hat = np.ones((Nx, int(Ny/ 2 + 1))) * np.complex(0.0, 0.0)
# for i in range(Nx):
#     for j in range(int(Ny / 2 + 1)):
#         w_hat[i, j] = np.complex(0.0, 1.0) * (kx[i] * v_hat[i, j] - ky[j] * u_hat[i, j])
#         # print("what[{}]: {:0.10f} {:0.10f} I\t".format(i * (Ny / 2 + 1) + j, np.real(w_hat[i, j]), np.imag(w_hat[i, j])), end = "")
#         print("what[{}]: {} \t".format(i * (Ny / 2 + 1) + j, w_hat[i, j]), end = "")
#     print()


