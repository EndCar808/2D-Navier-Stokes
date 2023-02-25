import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


N = 16
Nf = N//2 + 1
kx = np.arange(0, Nf, dtype="int64")
ky = np.arange(-N//2 +1, Nf + 1, dtype="int64")

kf = 2 
kdelta = 1.0

power = 0
for kdelta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for i in range(N):
        for j in range(Nf):
        #    print("({}, {})".format(kx[j], ky[i]), end=" ")
            
            kabs = kx[j]**2 + ky[i]**2 

            if kabs <= kf + kdelta and kabs >= kf - kdelta and kx[j] != 0:
                power += np.exp(-(kabs - kf)**2 / 2.0)

    print("kdelta = {}\tkmin = {} - kmax = {}".format(kdelta, kf - kdelta, kf + kdelta))
    count = 0
    for i in range(N):
        for j in range(Nf):
            
            kabs = kx[j]**2 + ky[i]**2 
            
            if kabs == 0:
                continue
            elif kabs <= kf + kdelta and kabs >= kf - kdelta and kx[j] != 0 and ky[i] >= 0:
                count += 2
                
                ## Define random angles
                psi = np.random.rand() * 2.0 * np.pi
                phi = np.random.rand() * 2.0 * np.pi

                ## Define zeta_1 and zeta_2
                zeta_1 = np.random.rand() * np.cos(phi)
                zeta_2 = np.random.rand() * np.sin(phi)

                ## Get theta_1 and theta_2
                theta_1 = np.arctan((np.real(zeta_1) + np.real(zeta_2) * np.cos(psi) - np.imag(zeta_2) * np.sin(psi)) / (np.real(zeta_2) * np.sin(psi) + np.imag(zeta_1) + np.imag(zeta_2) * np.cos(psi)))
                theta_2 = theta_1 + psi
                
                ## Get the forcing 
                k1 = np.sqrt(72 * np.exp(-(kabs - kf)**2 / 2.0) / (np.pi * kabs * power)) * np.exp(-1j * theta_1) * np.cos(phi)
                k2 = np.sqrt(72 * np.exp(-(kabs - kf)**2 / 2.0) / (np.pi * kabs * power)) * np.exp(-1j * theta_2) * np.sin(phi)
                
                print("{}, {} - c {}".format(kx[j], ky[i], count))
                print("{}, {} - c {}".format(ky[i], -kx[j], count))

    print()
    print()
