import numpy as np 
import numpy.fft as fft
import pyfftw as fftw



def empty_real_array(shape, fft):

    if fft == "pyfftw":
        out = pyfftw.empty_aligned(shape, dtype='float64')
        out.flat[:] = 0.
        return out
    else:
        return np.zeros(shape, dtype='float64')


def empty_cmplx_array(shape, fft):

    if fft == "pyfftw":
        out = pyfftw.empty_aligned(shape, dtype='complex128')
        out.flat[:] = 0. + 0.*1.0j
        return out
    else:
        return np.zeros(shape, dtype='complex128')




if __name__ == "__main__":
    Nx = 16
    Ny = 16
    Nyf = int(Ny / 2 + 1)


    u     = empty_real_array((Nx, Ny), "pyfft")
    v     = empty_real_array((Nx, Ny), "pyfft")
    u_hat = empty_cmplx_array((Nx, Nyf), "pyfft")
    v_hat = empty_cmplx_array((Nx, Nyf), "pyfft")


    u_to_uh = fftw.FFTW(u,  u_hat, threads = 6, axes = (-2, -1))
    uh_to_u = fftw.FFTW(u_hat,  u, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
    v_to_vh = fftw.FFTW(v,  v_hat, threads = 6, axes = (-2, -1))
    vh_to_v = fftw.FFTW(v_hat,  v, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        

    for i in range(Nx):
        for j in range(Ny):
            u[i, j] = -np.sin(np.pi * i * (2.0 * np.pi / Nx)) * np.cos(np.pi * j * (2.0 * np.pi / Ny))
            v[i, j] = np.cos(np.pi * i * (2.0 * np.pi / Nx)) * np.sin(np.pi * j * (2.0 * np.pi / Ny))

            print("u[{}]: {} \t v[{}]: {}".format(i * Ny + j, u[i, j], i * Ny + j, v[i, j]))
    print()
    print()
    uh_fft = fft.fft2(u)
    vh_fft = fft.fft2(v)

    u_to_uh()
    v_to_vh()

    for i in range(Nx):
        for j in range(Nyf):
            # print("uh[{}]: {} {} I \t\t\t uhfft[{}]: {} {} I \t Diff: {}".format(i * Nyf + j, np.real(u_hat[i, j]), np.imag(u_hat[i, j]), i * Nyf + j, np.real(uh_fft[i, j]), np.imag(uh_fft[i, j]), np.absolute(u_hat[i, j] - uh_fft[i, j])))
            print("uh[{}]: {:1.16f} {:1.16f} I \t\t\t vh[{}]: {:1.16f} {:1.16f} I".format(i * Nyf + j, np.real(u_hat[i, j]), np.imag(u_hat[i, j]), i * Nyf + j, np.real(v_hat[i, j]), np.imag(v_hat[i, j])))


    for i in range(Nx):
        for j in range(Nyf):
            if i <= Nx / 2:
                u_hat[i, j] = 1.0j * (i) * u_hat[i, j]
            else:
                u_hat[i, j] = 1.0j * (-Nx + i) * u_hat[i, j]
            v_hat[i, j] = 1.0j * j * v_hat[i, j]

    u_fft = fft.irfft2(u_hat)
    v_fft = fft.irfft2(v_hat)

    uh_to_u()
    vh_to_v()

    print()
    print()
    for i in range(Ny):
        for j in range(Nx):
            print("du[{}]: {} \tdv[{}]: {} \t duf[{}]: {} \tdvf[{}]: {}".format(i * Ny + j, u[i, j], i * Ny + j, v[i, j], i * Ny + j, u_fft[i, j], i * Ny + j, v_fft[i, j]))



    print()
    print()
    u_err = 0.
    v_err = 0.
    for i in range(Nx):
        for j in range(Ny):
            u_err += np.absolute(u[i, j] - (-np.pi * np.cos(np.pi * i * (2.0 * np.pi / Nx)) * np.cos(np.pi * j * (2.0 * np.pi / Ny))))**2
            v_err += np.absolute(v[i, j] - (np.pi * np.cos(np.pi * i * (2.0 * np.pi / Nx)) * np.cos(np.pi * j * (2.0 * np.pi / Ny))))**2

    x = np.arange(0, Nx) * (2.0 * np.pi / Nx)
    y = np.arange(0, Ny) * (2.0 * np.pi / Ny)
    exact = -np.pi * np.cos(np.pi * x) * np.cos(np.pi *  y[:, np.newaxis])
    err = np.absolute(u - exact)
    print("My L2: {} \t Numpy L2: {}".format(np.sqrt(1 / (Nx * Ny) * u_err), np.linalg.norm(err.flat)))
            