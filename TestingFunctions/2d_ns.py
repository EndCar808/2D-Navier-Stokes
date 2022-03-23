#!/usr/bin/env python3
import numpy as np
import pyfftw as fftw
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def InitialConditions(Nx, Ny, Nyf, x, y, u0, filt_mask):

    w     = empty_real_array((Nx, Ny), "pyfft")
    w_hat = empty_cmplx_array((Nx, Nyf), "pyfft")

    if u0 == "TAYLOR_GREEN":
        ## Set the real vorticity
        w[:, :] = 2.0 * KAPPA * np.cos(KAPPA * x) * np.cos(KAPPA * y[:, np.newaxis]) 

        ## Transform to Fourier space
        w_hat[:, :] = w_to_wh(w) * filt_mask
    elif u0 == "DOUBLE_SHEAR_LAYER":
        ## Set the real vorticity
        w[:, :] = DELTA * np.cos(x[:]) - SIGMA * np.cosh(SIGMA * (y[:, np.newaxis] - 0.5 * np.pi))**(-2)
        w[:, :] += DELTA * np.cos(x[:]) + SIGMA * np.cosh(SIGMA * (1.5 * np.pi - y[:, np.newaxis]))**(-2)

        ## Transform to Fourier space
        w_hat[:, :] = w_to_wh(w) * filt_mask
    else:
        ## Set random initial conditions
        w_hat[:, :] = np.random.rand(Nx, Nyf) * np.exp(1.0j * np.random.rand(Nx, Nyf) * np.pi) * filt_mask
    

    return w_hat


def TestTaylorGreen(x, y, t):

    ## Vorticty in real space
    w = 2. * KAPPA * np.cos(KAPPA * x) * np.cos(KAPPA * y[:, np.newaxis]) * np.exp(- 2 * KAPPA**2 * nu * t)

    return w

def L2_norm(err, Nx, Ny):

    return np.sqrt(1. / (Nx * Ny) * np.sum(err**2))

def Linf_norm(err):

    return np.amax(err)

def TotalEnergy(w_h, k_sqr, k_sqr_i):

    ## Get psi
    psi = k_sqr_i * w_h[:, :]

    tens = np.real(.5 * k_sqr * psi * np.conj(psi)) / (Nx * Ny)

    return tens.mean()

def TotalEnstrophy(w_h, k_sqr, k_sqr_i):

    ## Get psi
    psi = k_sqr_i * w_h[:, :]

    tens = np.real(.5 * k_sqr**2 * psi * np.conj(psi)) / (Nx * Ny)

    return tens.mean()


def UpdateTimestep(w_h, dx, dy, k_sqr_i):

    u_hat = empty_cmplx_array((Nx, Nyf), "pyfft")
    v_hat = empty_cmplx_array((Nx, Nyf), "pyfft")

    ## Get the Fourier velocities
    u_hat[:, :] = 1.0j * k_sqr_i * ky[:, np.newaxis] * w_h[:, :]
    v_hat[:, :] = -1.0j * k_sqr_i * kx[:Nyf] * w_h[:, :]

    ## Transform back to real space velocities
    u = uh_to_u(u_hat)
    v = vh_to_v(v_hat)

    ## Convective and Diffusive Scales
    D_c  = np.amax(np.pi * ((np.absolute(u)) / dx + (np.absolute(v)) / dy))
    D_mu = np.amax(np.pi**2 * (nu / dx**2 + nu / dy**2)) # * nu

    ## New Timestep
    dt = CFL_NUM / (D_c + D_mu)

    return dt


def ComputeEnergySpectrum(w_h, k_sqr, k_sqr_i, Nyf, res = 200):

    ## Get psi
    psi = k_sqr_i * w_h[:, :]

    tke = np.real(.5 * k_sqr * psi * np.conj(psi))
    kmod   = np.sqrt(k_sqr)

    k = np.arange(1, Nyf, 1, dtype = np.float64) # nyquist limit for this grid
    E = np.zeros_like(k)
    dk = (np.max(k) - np.min(k)) / res  ##0.5

    #  binning energies with wavenumber modulus in threshold
    for i in range(len(k)):
        E[i] += np.sum(tke[(kmod < k[i] + dk) & (kmod >= k[i] - dk)])

    return E

def ComputeEnstrophySpectrum(w_h, k_sqr, k_sqr_i, Nyf, res = 200):

    ## Get psi
    psi = k_sqr_i * w_h[:, :]

    tens = np.real(.5 * k_sqr**2 * psi * np.conj(psi))
    kmod   = np.sqrt(k_sqr)

    k = np.arange(1, Nyf, 1, dtype = np.float64) # nyquist limit for this grid
    E = np.zeros_like(k)
    dk = 0.5 ## (np.max(k) - np.min(k)) / res

    #  binning energies with wavenumber modulus in threshold
    for i in range(len(k)):
        E[i] += np.sum(tens[(kmod < k[i] + dk) & (kmod >= k[i] - dk)])

    return E


def NonlinearRHS(w_h, Nx, Ny, Nyf, Mx, My, Myf, kx, ky, k_sqr_i, padder, filt_mask):

    ## Allocate memory
    if PADDED == 1:
        u_hat    = empty_cmplx_array((Mx, Myf), "pyfft")
        v_hat    = empty_cmplx_array((Mx, Myf), "pyfft")
        w_hat_dx = empty_cmplx_array((Mx, Myf), "pyfft")
        w_hat_dy = empty_cmplx_array((Mx, Myf), "pyfft")

        ## Get the Fourier velocities
        u_hat[padder, :Nyf] = 1.0j * k_sqr_i * ky[:Nyf] * w_h[:, :]
        v_hat[padder, :Nyf] = -1.0j * k_sqr_i * kx[:, np.newaxis] * w_h[:, :]

        ## Transform back to real space velocities
        u = uh_to_u_pad(u_hat)
        v = vh_to_v_pad(v_hat)

        ## Get the Fourier derivatives of vorticity
        w_hat_dx[padder, :Nyf] = 1.0j * kx[:, np.newaxis] * w_h[:, :]
        w_hat_dy[padder, :Nyf] = 1.0j * ky[:Nyf] * w_h[:, :]

        ## Transform back to real space vorticity derivatives
        w_dx = w_dx_h_to_w_dx_pad(w_hat_dx)
        w_dy = w_dy_h_to_w_dy_pad(w_hat_dy)

        ## Multiply in real space
        rhs = u*w_dx + v*w_dy 

        ## Transform back the Fourier Space and dealias
        rhs_hat = w_to_wh_pad(rhs) 

        # print(rhs_hat[padder, :Nyf] * PAD**2)
        return rhs_hat[padder, :Nyf] * PAD**2
    else:
        u_hat    = empty_cmplx_array((Nx, Nyf), "pyfft")
        v_hat    = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_hat_dx = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_hat_dy = empty_cmplx_array((Nx, Nyf), "pyfft")
        
        ## Get the Fourier velocities
        u_hat[:, :] = 1.0j * k_sqr_i * ky[:Nyf] * w_h[:, :]
        v_hat[:, :] = -1.0j * k_sqr_i * kx[:, np.newaxis] * w_h[:, :]

        ## Get the Fourier derivatives of vorticity
        w_hat_dx = 1.0j * kx[:, np.newaxis] * w_h[:, :]
        w_hat_dy = 1.0j * ky[:Nyf] * w_h[:, :]

        # for i in range(Nx):
        #     for j in range(Nyf):
        #         print("-k_sqr[{}, {}]: {:0.6f} {:0.6f} I\t k_sqr[{}]: {} \t wh[{}, {}]: {:0.6f} {:0.6f} I".format(i, j, np.real(-1.0j * k_sqr_i[i, j]), np.imag(-1.0j * k_sqr_i[i, j]), i, kx[i], i, j, np.real(w_h[i, j]), np.imag(w_h[i, j])))
        # print()

        # for i in range(Nx):
        #     for j in range(Nyf):
        #         print("uh[{}]: {:0.20f} {:0.20f} I".format(i * Nyf + j, np.real(u_hat[i, j]), np.imag(u_hat[i, j])))
        # print()

        # for i in range(Nx):
        #     for j in range(Nyf):
        #         print("vh[{}]: {:0.20f} {:0.20f} I".format(i * Nyf + j, np.real(v_hat[i, j]), np.imag(v_hat[i, j])))
        # print()

        # for i in range(Nx):
        #     for j in range(Nyf):
        #         print("wh_dx[{}]: {:0.20f} {:0.20f} I".format(i * Nyf + j, np.real(w_hat_dx[i, j]), np.imag(w_hat_dx[i, j])))
        # print()

        # for i in range(Nx):
        #     for j in range(Nyf):
        #         print("wh_dy[{}]: {:0.20f} {:0.20f} I".format(i * Nyf + j, np.real(w_hat_dy[i, j]), np.imag(w_hat_dy[i, j])))
        # print()

        ## Transform back to real space velocities
        u = uh_to_u(u_hat)
        v = vh_to_v(v_hat)

        ## Transform back to real space vorticity derivatives
        w_dx = w_dx_h_to_w_dx(w_hat_dx)
        w_dy = w_dy_h_to_w_dy(w_hat_dy)

        # for i in range(Nx):
        #     for j in range(Ny):
        #         print("u[{}]: {:0.20f}".format(i * Ny + j, np.real(u[i, j])* (Nx * Ny), np.imag(u[i, j])* (Nx * Ny)))
        # print()

        # for i in range(Nx):
        #     for j in range(Ny):
        #         print("v[{}]: {:0.20f}".format(i * Ny + j, np.real(v[i, j])* (Nx * Ny), np.imag(v[i, j])* (Nx * Ny)))
        # print()

        # for i in range(Nx):
        #     for j in range(Ny):
        #         print("w_dx[{}]: {:0.20f}".format(i * Ny + j, np.real(w_dx[i, j])* (Nx * Ny), np.imag(w_dx[i, j])* (Nx * Ny)))
        # print()

        # for i in range(Nx):
        #     for j in range(Ny):
        #         print("w_dy[{}]: {:0.20f}".format(i * Ny + j, np.real(w_dy[i, j])* (Nx * Ny), np.imag(w_dy[i, j])* (Nx * Ny)))
        # print()

        ## Multiply in real space
        rhs = u * w_dx + v * w_dy

        # for i in range(Nx):
        #     for j in range(Ny):
        #         print("rhs[{}]: {:g}".format(i * Ny + j, np.real(rhs[i, j])* (Nx * Ny)**2, np.imag(rhs[i, j])* (Nx * Ny)**2))
        # print()

        ## Transform back the Fourier Space and dealias
        rhs_hat = rhs_to_rhsh(rhs) 

        # for i in range(Nx):
        #     for j in range(Nyf):
        #         print("rhs_h[{}]: {:0.20f} {:0.20f} I".format(i * Nyf + j, np.real(rhs_hat[i, j]), np.imag(rhs_hat[i, j])))
        # print()


        return rhs_hat[:, :] * filt_mask

if __name__ == "__main__":

    
    ### ---------------------------
    ### System Parameters
    ### ---------------------------
    EULER_SOLVE = 0
    TESTING     = 1
    PADDED      = 0
    PAD = 3./2.

    ## Space variables
    Nx    = int(8)
    Ny    = int(8)
    Nyf   = int(Ny / 2 + 1)
    Mx  = int(PAD * Nx)
    My  = int(PAD * Ny)
    Myf = int(PAD * Nyf)
    x, dx = np.linspace(0., 2. * np.pi, Nx, endpoint = False, retstep = True)
    y, dy = np.linspace(0., 2. * np.pi, Ny, endpoint = False, retstep = True)
    # kx    = np.fft.fftfreq(Nx, d = 1. / Nx)
    # ky    = np.fft.fftfreq(Ny, d = 1. / Ny)
    kx = np.append(np.arange(0, Nyf), np.linspace(-Ny//2 + 1, -1, Ny//2 - 1))
    ky = np.append(np.arange(0, Nyf), np.linspace(-Ny//2 + 1, -1, Ny//2 - 1))


    # for i in range(Nx):
    #         print("kx[{}]: {} \t ky[{}]: {}".format(i, kx[i], i, ky[i]))
    # print()
    # print()


    ## Pre-compute arrays
    k_sqr        = kx[:Nyf]**2 + ky[:, np.newaxis]**2    ## NOTE: This has y along the rows and x along the columns which is opposite to C code but values are exact same
    non_zer_indx = k_sqr != 0.0
    k_sqr_inv    = empty_cmplx_array((Nx, Nyf), "pyfft")
    k_sqr_inv[non_zer_indx] = 1. / k_sqr[non_zer_indx]

    ## Dealias Mask
    if PADDED == 1:
        # for easier slicing when padding
        padder = np.ones(Mx, dtype = bool)
        padder[int(Nx / 2):int(Nx * (PAD - 0.5)):] = False

        filt_mask = empty_real_array((Nx, Nyf), "pyfft")
        filt_mask.flat[:] = 1.
    else:
        filt_mask = empty_real_array((Nx, Nyf), "pyfft")
        Nk_23     = int(Nx / 3 + 1)
        filt_indx = np.absolute(ky) < Nk_23 
        filt_mask[filt_indx, :Nk_23] = 1.
        padder = np.ones(Mx, dtype = bool)

    ## Time variables
    t0 = 0.0
    t  = t0
    dt = 0.0001
    T  = 1.0 
    print_iters = 100 #int(T / dt * 0.1)


    ## Eqn parameters
    nu    = 1.
    KAPPA = 1.0
    SIGMA = 15. / np.pi
    DELTA = 0.005   
    CFL_NUM = np.sqrt(3.)


    # for i in range(Nx):
    #     for j in range(Nyf):
    #         print("k[{}, {}]: {}".format(i, j, np.real(k_sqr_inv[i, j])))
    # print()
    # print()

    ### ---------------------------
    ### Allocate Memory
    ### ---------------------------
    if PADDED == 1:
        w         = empty_real_array((Nx, Ny), "pyfft")
        u         = empty_real_array((Nx, Ny), "pyfft")
        v         = empty_real_array((Nx, Ny), "pyfft")
        v_hat     = empty_cmplx_array((Nx, Nyf), "pyfft")
        u_hat     = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_hat     = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_pad     = empty_real_array((Mx, My), "pyfft")
        u_pad     = empty_real_array((Mx, My), "pyfft")
        v_pad     = empty_real_array((Mx, My), "pyfft")
        u_hat_pad = empty_cmplx_array((Mx, Myf), "pyfft")
        v_hat_pad = empty_cmplx_array((Mx, Myf), "pyfft")
        w_hat_pad = empty_cmplx_array((Mx, Myf), "pyfft")
        w_dx      = empty_real_array((Mx, My), "pyfft")
        w_dy      = empty_real_array((Mx, My), "pyfft")
        w_dx_h    = empty_cmplx_array((Mx, Myf), "pyfft")
        w_dy_h    = empty_cmplx_array((Mx, Myf), "pyfft")
    else:
        w       = empty_real_array((Nx, Ny), "pyfft")
        u       = empty_real_array((Nx, Ny), "pyfft")
        v       = empty_real_array((Nx, Ny), "pyfft")
        rhs     = empty_real_array((Nx, Ny), "pyfft")
        rhs_hat = empty_cmplx_array((Nx, Nyf), "pyfft") 
        u_hat   = empty_cmplx_array((Nx, Nyf), "pyfft")
        v_hat   = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_hat   = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_dx    = empty_real_array((Nx, Ny), "pyfft")
        w_dy    = empty_real_array((Nx, Ny), "pyfft")
        w_dx_h  = empty_cmplx_array((Nx, Nyf), "pyfft")
        w_dy_h  = empty_cmplx_array((Nx, Nyf), "pyfft")
    RK1 = empty_cmplx_array((Nx, Nyf), "pyfft")
    RK2 = empty_cmplx_array((Nx, Nyf), "pyfft")
    RK3 = empty_cmplx_array((Nx, Nyf), "pyfft")
    RK4 = empty_cmplx_array((Nx, Nyf), "pyfft")
    if TESTING:
        ww = empty_real_array((Nx, Ny), "pyfft")
        wh = empty_cmplx_array((Nx, Nyf), "pyfft")

    ### ---------------------------
    ### Set Up Transforms
    ### ---------------------------
    fftw.interfaces.cache.enable()
    if PADDED == 1:
        w_to_wh            = fftw.FFTW(w,  w_hat, threads = 6, axes = (-2, -1))
        wh_to_w            = fftw.FFTW(w_hat,  w, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        u_to_uh            = fftw.FFTW(u,  u_hat, threads = 6, axes = (-2, -1))
        uh_to_u            = fftw.FFTW(u_hat,  u, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        v_to_vh            = fftw.FFTW(v,  v_hat, threads = 6, axes = (-2, -1))
        vh_to_v            = fftw.FFTW(v_hat,  v, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        w_to_wh_pad        = fftw.FFTW(w_pad,  w_hat_pad, threads = 6, axes = (-2, -1))
        w_dx_h_to_w        = fftw.FFTW(w_hat_pad,  w_pad, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        uh_to_u_pad        = fftw.FFTW(u_hat_pad,  u_pad, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        vh_to_v_pad        = fftw.FFTW(v_hat_pad,  v_pad, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        w_dx_h_to_w_dx_pad = fftw.FFTW(w_dx_h,  w_dx, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        w_dy_h_to_w_dy_pad = fftw.FFTW(w_dy_h,  w_dy, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
    else:
        w_to_wh        = fftw.FFTW(w,  w_hat, threads = 6, axes = (-2, -1))
        wh_to_w        = fftw.FFTW(w_hat,  w, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        u_to_uh        = fftw.FFTW(u,  u_hat, threads = 6, axes = (-2, -1))
        uh_to_u        = fftw.FFTW(u_hat,  u, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        v_to_vh        = fftw.FFTW(v,  v_hat, threads = 6, axes = (-2, -1))
        vh_to_v        = fftw.FFTW(v_hat,  v, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        rhs_to_rhsh    = fftw.FFTW(rhs,  rhs_hat, threads = 6, axes = (-2, -1))
        rhsh_to_rhs    = fftw.FFTW(rhs_hat,  rhs, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        w_dx_h_to_w_dx = fftw.FFTW(w_dx_h,  w_dx, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
        w_dy_h_to_w_dy = fftw.FFTW(w_dy_h,  w_dy, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
    if TESTING:
        ifft = fftw.FFTW(wh,  ww, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
    

    ### ---------------------------
    ### Get Initial Conditions
    ### ---------------------------
    if TESTING:
        w_hat = InitialConditions(Nx, Ny, Nyf, x, y, "TAYLOR_GREEN", filt_mask)
    else:
        w_hat = InitialConditions(Nx, Ny, Nyf, x, y, "DOUBLE_SHEAR_LAYER", filt_mask)
    wh_to_w()

    # energy_spec = ComputeEnergySpectrum(w_hat, k_sqr, k_sqr_inv, Nyf)
    
        
    # for i in range(Nx):
    #     for j in range(Nyf):
    #         print("wh[{}]: {:0.20f} {:0.20f} I".format(i * Nyf + j, np.real(w_hat[i, j]), np.imag(w_hat[i, j])))
    # print()


    # RK1 = NonlinearRHS(w_hat, Nx, Ny, Nyf, Mx, My, Myf, kx, ky, k_sqr_inv, padder, filt_mask)


    # for i in range(Nx):
    #     for j in range(Nyf):
    #         print("RHS[{}]: {}".format(i * Nyf + j, RK1[i, j] ))
    # print()

    print(kx)

    print(ky)

    print(np.meshgrid(kx, ky))

    for i in range(len(kx)):
        for j in range(len(ky)):
            print("({}, {}) ".format(kx[i], ky[j]), end = "")
        print()


    iters = 0
    while t <= T:


        # ## Stage 1
        RK1 = NonlinearRHS(w_hat, Nx, Ny, Nyf, Mx, My, Myf, kx, ky, k_sqr_inv, padder, filt_mask)
                
        ## Stage 2
        RK2 = NonlinearRHS(w_hat + dt * 0.5 * RK1, Nx, Ny, Nyf, Mx, My, Myf, kx, ky, k_sqr_inv, padder, filt_mask)
        
        ## Stage 3
        RK3 = NonlinearRHS(w_hat + dt * 0.5 * RK2, Nx, Ny, Nyf, Mx, My, Myf, kx, ky, k_sqr_inv, padder, filt_mask)

        ## Stage 4
        RK4 = NonlinearRHS(w_hat + dt * RK3, Nx, Ny, Nyf, Mx, My, Myf, kx, ky, k_sqr_inv, padder, filt_mask)
        

        ## Update Stage
        if EULER_SOLVE:
            w_hat[:, :] = w_hat[:, :] + 1./6.0 * dt * RK1 + 1./3. * dt * RK2 + 1./3. * dt * RK3 + 1./6. * dt * RK4
        else:
            D = dt * nu * k_sqr
            w_hat[:, :] = w_hat[:, :] * ((2. - D) / (2. + D)) + (2 * dt / (2. + D)) * (1./6. * RK1 + 1./3. * RK2 + 1./3. * RK3 + 1./6. * RK4)

        ## Update for next iteration
        t  += dt
        dt = UpdateTimestep(w_hat, dx, dy, k_sqr_inv)
        iters += 1

        ## Print Update to screen
        if np.mod(iters, print_iters) == 0:
            if TESTING:
                wh = w_hat.copy()
                ww = ifft(wh)
                abs_err = np.absolute(ww - TestTaylorGreen(x, y, t))
                print("Iter: {} \t t: {:0.5f} \t dt: {:0.5f} \t KE: {:0.5f} \t ENS: {:0.5f} \t L2: {:0.5g} \t Linf: {:0.5g}".format(iters, t, dt, TotalEnergy(w_hat, k_sqr, k_sqr_inv), TotalEnstrophy(w_hat, k_sqr, k_sqr_inv), L2_norm(abs_err, Nx, Ny), Linf_norm(abs_err)))

                plt.figure()
                plt.imshow(w[:, :])
                plt.colorbar()
                plt.savefig("../Testing_Functions/wh.png[{}].png".format(iters))

                for i in range(5):
                    for j in range(5):
                        print("Er[{}, {}]: {} ".format(i, j, abs_err[i, j]), end = "")
                    print()
                print()
                print()
                plt.figure()
                plt.imshow(abs_err)
                plt.colorbar()
                plt.savefig("../Testing_Functions/wh_err[{}].png".format(iters))
            else:
                print("Iter: {} \t t: {:0.5f} \t dt: {:0.5f} \t KE: {:0.5f} \t ENS: {:0.5f}".format(iters, t, dt, TotalEnergy(w_hat, k_sqr_inv), TotalEnstrophy(w_hat)))




