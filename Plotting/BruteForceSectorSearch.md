---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
```

```python
R = 2
angle = np.linspace(0.0, 2.0 * np.pi, 256)

x = R * np.cos(angle)
y = R * np.sin(angle)
```

```python
## Theta angles
n_theta   = 12
dtheta    = np.pi / n_theta
theta     = np.arange(-np.pi/2.0, np.pi/2.0 + dtheta, dtheta)
mid_theta = (theta[1:] + theta[:-1]) * 0.5
theta = mid_theta[:]


Nx = 32
Ny = 32
kx = np.arange(-Nx//2 + 1, Nx//2 + 1, 1)
ky = np.arange(0, Nx//2 + 1, 1)
kmax = Nx//3
kmax_sqr = kmax**2
```

```python jupyter={"outputs_hidden": true, "source_hidden": true}
a = 6
theta_a = theta[a]
l = 3
print(a)
for l in range(0, len(k1_sect_angles)):
    theta_k1 = theta[a] + k1_sect_angles[l] * dtheta

    ## Get the k3 sector boundary angles
    k3_theta_lwr = theta_a - dtheta/2.0
    k3_theta_upr = theta_a + dtheta/2.0
    ## Get the k1 sector boundary angles
    k1_theta_lwr = theta_k1 - dtheta/2.0
    k1_theta_upr = theta_k1 + dtheta/2.0

    for k3_x in kx:
        for k3_y in ky:
            ## Get polar coords
            k3_sqr   = k3_x**2 + k3_y**2
            k3_angle = np.arctan2(k3_x, k3_y)

            if (k3_sqr > 0 and k3_sqr < kmax_sqr) and (k3_angle >= k3_theta_lwr and k3_angle < k3_theta_upr):
                for k1_x in kx:
                    for k1_y in ky:
                        ## Get k1 polar coords
                        k1_sqr   = k1_x**2 + k1_y**2
                        k1_angle = np.arctan2(k1_x, k1_y)

                        if (k1_sqr > 0 and k1_sqr < kmax_sqr) and (k1_angle >= k1_theta_lwr and k1_angle < k1_theta_upr):
                            ## Get k2 
                            k2_x = k3_x - k1_x
                            k2_y = k3_y - k1_y

                            ## Get k2 polar coords
                            k2_sqr   = k2_x**2 + k2_y**2
                            k2_angle = np.arctan2(k2_x, k2_y)
    #                         if k2_y < 0:
                            neg_k2_angle = np.arctan2(-k2_x, -k2_y)

                            ## Get the sector angle for k2
                            k3_vec = np.exp(1j * theta_a)
                            k1_vec = np.exp(1j * (theta_k1))
                            k2_theta = np.real((1.0/ (2.0 * 1j) * (np.log(k3_vec - k1_vec) - np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec)))))
                            k2_theta_lwr = k2_theta - dtheta/2.0
                            k2_theta_upr = k2_theta + dtheta/2.0

    #                         print("a: {} l: {} \t | k1: ({}, {}) \t k2: ({}, {}) \t k3: ({}, {}) \t | k1_a: {:0.6f} - \t lwr: {:0.6f} k2_a: {:0.6f} -k2_a: {:0.6f} upr: {:0.6f} \t k3_a: {:0.6f}".format(a, l, k1_x, k1_y, k2_x, k2_y, k3_x, k3_y, k1_angle, k2_theta_lwr, k2_angle, neg_k2_angle, k2_theta_upr, k3_angle))
                            if (k2_sqr > 0 and k2_sqr < kmax_sqr) and ((k2_angle >= k2_theta_lwr and k2_angle < k2_theta_upr) or (k2_y < 0 and neg_k2_angle >= k2_theta_lwr and neg_k2_angle < k2_theta_upr)):
                                print("a: {} l: {} \t | k1: ({}, {}) \t k2: ({}, {}) \t k3: ({}, {}) \t | k1_a: {:0.6f} \t k2_a: {:0.6f} \t k3_a: {:0.6f}".format(a, l, k1_x, k1_y, k2_x, k2_y, k3_x, k3_y, k1_angle, k2_angle, k3_angle))

    plt.figure(figsize = (13, 12))
    plt.plot(x, y, 'k--', alpha = 0.2)
    plt.plot([0, 0], [-R, R], 'k--', alpha = 0.2)
    plt.plot([R, -R], [0, 0], 'k--', alpha = 0.2)
    for aa in range(theta.shape[0]):
        ## Plot sectors
        plt.plot([0, R*np.cos(theta[aa] - dtheta/2)], [0, R*np.sin(theta[aa] - dtheta/2)], 'r--', alpha = 0.1)
        plt.plot([0, R*np.cos(theta[aa] + dtheta/2)], [0, R*np.sin(theta[aa] + dtheta/2)], 'r--', alpha = 0.1)
    ## Plot sector k3
    plt.plot([0, np.cos(theta_a)], [0, np.sin(theta_a)], 'b')
    plt.plot([0, R*np.cos(theta_a - dtheta/2)], [0, R*np.sin(theta_a - dtheta/2)], 'b--')
    plt.plot([0, R*np.cos(theta_a + dtheta/2)], [0, R*np.sin(theta_a + dtheta/2)], 'b--')
    ## Plot sector k1
    plt.plot([0, np.cos(theta_k1)], [0, np.sin(theta_k1)], 'r')  ## -pi/6
    plt.plot([0, -np.cos(theta_k1)], [0, -np.sin(theta_k1)], 'r')  ## -pi/6
    plt.plot([0, R*np.cos(theta_k1 - dtheta/2)], [0, R*np.sin(theta_k1 - dtheta/2)], 'r--')  ## -pi/6
    plt.plot([0, R*np.cos(theta_k1 + dtheta/2)], [0, R*np.sin(theta_k1 + dtheta/2)], 'r--')  ## -pi/6
    ## Plot sector k2 = k3 - k1
    k3_vec = np.exp(1j * theta_a)
    k1_vec = np.exp(1j * (theta_k1))
    k2_theta = np.real((1.0/ (2.0 * 1j) * (np.log(k3_vec - k1_vec) - np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec)))))
    plt.plot([0, np.cos(k2_theta)], [0, np.sin(k2_theta)], 'g')
    plt.plot([0, R*np.cos(k2_theta - dtheta/2)], [0, R*np.sin(k2_theta - dtheta/2)], 'g--')
    plt.plot([0, R*np.cos(k2_theta + dtheta/2)], [0, R*np.sin(k2_theta + dtheta/2)], 'g--')
    k3 = [R*np.cos(theta_a), R*np.sin(theta_a)]
    k1 = [R*np.cos(theta_k1), R*np.sin(theta_k1)]
    print(np.arccos(np.dot([R*np.cos(theta_a), R*np.sin(theta_a)], [R*np.cos(theta_k1), R*np.sin(theta_k1)]) / R**2))
    plt.show()
```

```python

```

```python

```

```python

```

```python
## Theta angles
n_theta   = 24
dtheta    = 2.0 * np.pi / n_theta
theta     = np.arange(-np.pi, np.pi + dtheta, dtheta)
mid_theta = (theta[1:] + theta[:-1]) * 0.5
theta     = mid_theta[:]


## k1 sector angles
k1_sect_angles = [-(n_theta/2), -(n_theta/3), -(n_theta/4), -(n_theta/6) , (n_theta/6), (n_theta/4), (n_theta/3), (n_theta/2)]

Nx = 32
Ny = 32
kx = np.arange(-Nx//2 + 1, Nx//2 + 1, 1)
ky = np.arange(-Ny//2 + 1, Nx//2 + 1, 1)
kmax = Nx//3
kmax_sqr = kmax**2

C_frac = 0.75
kmax_C = int(np.ceil(C_frac * kmax))
kmax_C_sqr = kmax_C ** 2

print(kmax_C_sqr)
R = kmax

x = R * np.cos(angle)
y = R * np.sin(angle)


for i in range(n_theta):
    print("thete[{}]: {} \t dtheta: {}".format(i, theta[i], dtheta))
```

```python jupyter={"outputs_hidden": true, "source_hidden": true}
plt.figure(figsize = (13, 12))
plt.plot(x, y, 'k--', alpha = 0.2)
plt.plot([0, 0], [-R, R], 'k--', alpha = 0.2)
plt.plot([R, -R], [0, 0], 'k--', alpha = 0.2)
for aa in range(theta.shape[0]):
    ## Plot sectors
    plt.plot([0, R*np.cos(theta[aa] - dtheta/2)], [0, R*np.sin(theta[aa] - dtheta/2)], 'r--', alpha = 0.1)
    plt.plot([0, R*np.cos(theta[aa] + dtheta/2)], [0, R*np.sin(theta[aa] + dtheta/2)], 'r--', alpha = 0.1)
#     plt.plot([0, np.cos(theta[aa])], [0, np.sin(theta[aa])], 'r', alpha = 0.21)

```

```python jupyter={"outputs_hidden": true, "source_hidden": true}
for a in range( int(n_theta)):
    theta_a = theta[a]
    
    plt.figure(figsize = (13, 12))
    plt.plot(x, y, 'k--', alpha = 0.2)
    plt.plot([0, 0], [-R, R], 'k--', alpha = 0.2)
    plt.plot([R, -R], [0, 0], 'k--', alpha = 0.2)
    for aa in range(theta.shape[0]):
        ## Plot sectors
        plt.plot([0, R*np.cos(theta[aa] - dtheta/2)], [0, R*np.sin(theta[aa] - dtheta/2)], 'r--', alpha = 0.1)
        plt.plot([0, R*np.cos(theta[aa] + dtheta/2)], [0, R*np.sin(theta[aa] + dtheta/2)], 'r--', alpha = 0.1)
    
    ## Get the k3 sector boundary angles
    k3_theta_lwr = theta_a - dtheta/2.0
    k3_theta_upr = theta_a + dtheta/2.0
    
    for k3_x in kx:
        for k3_y in ky:
            ## Get polar coords
            k3_sqr   = k3_x**2 + k3_y**2
            k3_angle = np.arctan2(k3_y, k3_x)

            if (k3_sqr > 0 and k3_sqr < kmax_sqr) and (k3_angle >= k3_theta_lwr and k3_angle < k3_theta_upr):
                
                plt.plot([0, k3_x], [0, k3_y], 'b')
                plt.plot([0, R*np.cos(theta_a - dtheta/2)], [0, R*np.sin(theta_a - dtheta/2)], 'b--')
                plt.plot([0, R*np.cos(theta_a + dtheta/2)], [0, R*np.sin(theta_a + dtheta/2)], 'b--')
    plt.xticks(np.arange(-10, 11))
    plt.yticks(np.arange(-10, 11))
    plt.grid(which = 'both')
    plt.show()
```

```python
## Get full field data
phi_k = np.random.uniform(-np.pi, np.pi, (Nx, int(Ny/2 + 1)))
a_k   = np.random.uniform(0.0, 10, (Nx, int(Ny/2 + 1))) + 0.5

phases = np.zeros((2 * kmax - 1, 2 * kmax - 1))
amps   = np.zeros((2 * kmax - 1, 2 * kmax - 1))

for k_x in kx:
    if np.absolute(k_x) < kmax:
        for k_y in ky:
            if np.absolute(k_y) < kmax:
                
                k_sqr = k_x**2 + k_y**2
                
                if k_sqr < kmax_sqr:
                    if k_y == 0:
                        phases[kmax - 1 - k_x, kmax - 1 + k_y] = np.mod(phi_k[k_x, k_y], 2.0 * np.pi)
                        amps[kmax - 1 - k_x, kmax - 1 + k_y]   = a_k[k_x, k_y]
                    else:
                        phases[kmax - 1 - k_x, kmax - 1 + k_y] = np.mod(phi_k[k_x, k_y], 2.0 * np.pi)
                        phases[kmax - 1 + k_x, kmax - 1 - k_y] = np.mod(-phi_k[k_x, k_y], 2.0 * np.pi)
                        amps[kmax - 1 - k_x, kmax - 1 + k_y]   = a_k[k_x, k_y]
                        amps[kmax - 1 + k_x, kmax - 1 - k_y]   = a_k[k_x, k_y]
                        
plt.figure(figsize=(16, 9))      
plt.imshow(phases, extent = [-kmax + 1, kmax, -kmax + 1, kmax])
plt.colorbar()
plt.show()
plt.figure(figsize=(16, 9))      
plt.imshow(amps, extent = [-kmax + 1, kmax, -kmax + 1, kmax])
plt.colorbar()
plt.show()
```

```python
with h5py.File("/home/ecarroll/PhD/2D_Navier_Stokes/Data/Tmp/SIM_DATA_NAVIER_RK4_FULL_N[32,32]_T[0-40]_NU[0.000000]_CFL[1.73]_u0[DECAY_TURB_ALT]_TAG[Decay-Test-Alt]/PostProcessing_HDF_Data_SECTORS[24]_KFRAC[0.75]_TAG[NO_TAG].h5", 'r') as data_file:
    ndata = len([g for g in data_file.keys() if 'Snap' in g])
    
    phases = np.zeros((ndata, 2 * kmax - 1, 2 * kmax - 1))
    amps   = np.zeros((ndata, 2 * kmax - 1, 2 * kmax - 1))
    
    nn = 0
    for group in data_file.keys():
        if "Snap" in group:
            if 'FullFieldPhases' in list(data_file[group].keys()):
                phases[nn, :, :] = data_file[group]['FullFieldPhases'][:, :]
            if 'FullFieldAmplitudes' in list(data_file[group].keys()):
                amps[nn, :, :] = data_file[group]['FullFieldAmplitudes'][:, :]
            nn += 1
```

```python jupyter={"outputs_hidden": true, "source_hidden": true}
# a = 0
for a in range(n_theta):
    theta_a = theta[a]
    for l in range(0, len(k1_sect_angles)):
        theta_k1 = np.mod(theta[a] + k1_sect_angles[l] * dtheta + np.pi, 2.0 * np.pi) - np.pi

        ## Get the k3 sector boundary angles
        k3_theta_lwr = theta_a - dtheta/2.0
        k3_theta_upr = theta_a + dtheta/2.0
        ## Get the k1 sector boundary angles
        k1_theta_lwr = theta_k1 - dtheta/2.0
        k1_theta_upr = theta_k1 + dtheta/2.0
        
        plt.figure(figsize = (13, 12))
        plt.plot(x, y, 'k--', alpha = 0.2)
        plt.plot([0, 0], [-R, R], 'k--', alpha = 0.2)
        plt.plot([R, -R], [0, 0], 'k--', alpha = 0.2)
        for aa in range(theta.shape[0]):
            ## Plot sectors
            plt.plot([0, R*np.cos(theta[aa] - dtheta/2)], [0, R*np.sin(theta[aa] - dtheta/2)], 'r--', alpha = 0.1)
            plt.plot([0, R*np.cos(theta[aa] + dtheta/2)], [0, R*np.sin(theta[aa] + dtheta/2)], 'r--', alpha = 0.1)

        for k3_x in kx:
            for k3_y in ky:
                ## Get polar coords
                k3_sqr   = k3_x**2 + k3_y**2
                k3_angle = np.arctan2(k3_y, k3_x)

                if (k3_sqr > 0 and k3_sqr < kmax_sqr) and (k3_angle >= k3_theta_lwr and k3_angle < k3_theta_upr):
                    
                    plt.plot([0, k3_x], [0, k3_y], 'b')
                    plt.plot([0, R*np.cos(theta_a - dtheta/2)], [0, R*np.sin(theta_a - dtheta/2)], 'b--')
                    plt.plot([0, R*np.cos(theta_a + dtheta/2)], [0, R*np.sin(theta_a + dtheta/2)], 'b--')
                
                    for k1_x in kx:
                        for k1_y in ky:
                            ## Get k1 polar coords
                            k1_sqr   = k1_x**2 + k1_y**2
                            k1_angle = np.arctan2(k1_y, k1_x)

                            if (k1_sqr > 0 and k1_sqr < kmax_sqr) and (k1_angle >= k1_theta_lwr and k1_angle < k1_theta_upr):
                                ## Get k2 
                                k2_x = k3_x - k1_x
                                k2_y = k3_y - k1_y

                                ## Get k2 polar coords
                                k2_sqr   = k2_x**2 + k2_y**2
                                k2_angle = np.arctan2(k2_y, k2_x)

                                ## Get the sector angle for k2
                                k3_vec = np.exp(1j * theta_a)
                                k1_vec = np.exp(1j * (theta_k1))
                                k2_theta = np.real((1.0/ (2.0 * 1j) * (np.log(k3_vec - k1_vec) - np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec)))))
                                k2_theta_lwr = k2_theta - dtheta/2.0
                                k2_theta_upr = k2_theta + dtheta/2.0

                                if (k2_sqr > 0 and k2_sqr < kmax_sqr) and ((k2_angle >= k2_theta_lwr and k2_angle < k2_theta_upr)):
                                    print("a: {} l: {} \t | k1: ({}, {}) \t k2: ({}, {}) \t k3: ({}, {}) \t | k1_a: {:0.6f} \t k2_a: {:0.6f} \t k3_a: {:0.6f}".format(a, l, k1_x, k1_y, k2_x, k2_y, k3_x, k3_y, k1_angle, k2_angle, k3_angle))
                                    
                                    ## Flux pre factor term
                                    flux_pre_fac = (k1_x * k2_y - k1_y * k2_x) * (1.0 / k1_sqr - 1.0 / k2_sqr)
                                    
                                    ## Flux weight term
                                    flux_wght = flux_pre_fac * amps[kmax - 1 + k1_x, kmax - 1 + k1_y] * amps[kmax - 1 + k2_x, kmax - 1 + k2_y] * amps[kmax - 1 + k3_x, kmax - 1 + k3_y]
                                    
                                    ## Triad Phase
                                    triad_phase = phases[kmax - 1 + k1_x, kmax - 1 + k1_y] + phases[kmax - 1 + k2_x, kmax - 1 + k2_y] - phases[kmax - 1 + k3_x, kmax - 1 + k3_y]

                                    k3 = [np.cos(theta_a), np.sin(theta_a)]
                                    k1 = [np.cos(theta_k1), np.sin(theta_k1)]
                                    k2 = [np.cos(k2_theta), np.sin(k2_theta)]
                                    plt.plot([0, k1_x], [0, k1_y], 'r')
                                    plt.plot([0, k2_x], [0, k2_y], 'g')
        #                             plt.plot([0, np.cos(k2_theta)], [0, np.sin(k2_theta)], 'g')
                                    plt.plot([0, R*np.cos(k2_theta - dtheta/2)], [0, R*np.sin(k2_theta - dtheta/2)], 'g--')
                                    plt.plot([0, R*np.cos(k2_theta + dtheta/2)], [0, R*np.sin(k2_theta + dtheta/2)], 'g--')

        plt.xticks(np.arange(-10, 11))
        plt.yticks(np.arange(-10, 11))
        plt.grid(which = 'both')
        plt.show()

```

```python
triad_phase_order_across_sec = np.ones((6, n_theta, len(k1_sect_angles))) * np.complex(0.0, 0.0)
triad_R_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
triad_Phi_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
num_triads_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
enst_flux_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
enst_flux = np.zeros((6, n_theta, len(k1_sect_angles)))
num_triads = np.zeros((6, n_theta, len(k1_sect_angles)))

angles = [-np.pi/2, -np.pi/3, -np.pi/4, -np.pi/6, np.pi/6, np.pi/4, np.pi/3, np.pi/2]

def mid_angle(a, k1):
    
    k3_vec = np.exp(1j * a)
    k1_vec = np.exp(1j * (k1))
    k2_theta = np.real((1.0/ (2.0 * 1j) * (np.log(k3_vec - k1_vec) - np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec)))))
    
    return k2_theta

def sync_across_sec(a, l):
    
    ## Phase order data
    theta_a  = theta[a]
    theta_k1 = np.mod(theta[a] + k1_sect_angles[l] * dtheta / 2.0 + np.pi, 2.0 * np.pi) - np.pi
#     if theta_k1 < -np.pi:
#         theta_k1 = np.pi - np.mod(theta_k1 + np.pi, 2.0 * np.pi)
#     elif theta_k1 > np.pi:
#         theta_k1 = -np.pi + np.mod(theta_k1 + np.pi, 2.0 * np.pi)
    print("l: {}".format(l))
   

    ## Get the k3 sector boundary angles
    k3_theta_lwr = theta_a - dtheta/2.0
    k3_theta_upr = theta_a + dtheta/2.0
    ## Get the k1 sector boundary angles
    k1_theta_lwr = theta_k1 - dtheta/2.0
    k1_theta_upr = theta_k1 + dtheta/2.0

    plt.figure(figsize = (13, 12))
    plt.plot(x, y, 'k--', alpha = 0.2)
    plt.plot([0, 0], [-R, R], 'k--', alpha = 0.2)
    plt.plot([R, -R], [0, 0], 'k--', alpha = 0.2)
    for aa in range(theta.shape[0]):
        ## Plot sectors
        plt.plot([0, R*np.cos(theta[aa] - dtheta/2)], [0, R*np.sin(theta[aa] - dtheta/2)], 'r--', alpha = 0.1)
        plt.plot([0, R*np.cos(theta[aa] + dtheta/2)], [0, R*np.sin(theta[aa] + dtheta/2)], 'r--', alpha = 0.1)
    
    ## Plot the k3 sector
    plt.plot([0, R*np.cos(theta_a)], [0, R*np.sin(theta_a)], 'b')
    plt.plot([0, R*np.cos(theta_a - dtheta/2)], [0, R*np.sin(theta_a - dtheta/2)], 'b--')
    plt.plot([0, R*np.cos(theta_a + dtheta/2)], [0, R*np.sin(theta_a + dtheta/2)], 'b--')

    ## Plot the k1 sector
    plt.plot([0, R*np.cos(theta_k1)], [0, R*np.sin(theta_k1)], 'r')
    plt.plot([0, -R*np.cos(theta_k1)], [0, -R*np.sin(theta_k1)], 'r')
    plt.plot([0, R*np.cos(theta_k1 - dtheta/2)], [0, R*np.sin(theta_k1 - dtheta/2)], 'r--')
    plt.plot([0, R*np.cos(theta_k1 + dtheta/2)], [0, R*np.sin(theta_k1 + dtheta/2)], 'r--')
    
    ## Get the sector angle for k2
    print(theta_a)
    print(theta_k1, np.mod(theta_k1, 2.0 * np.pi))
    
    k2_theta = mid_angle(theta_a, theta_k1)
    plt.plot([0, R * np.cos(k2_theta)], [0, R * np.sin(k2_theta)], 'k')
    plt.plot([0, R*np.cos(k2_theta - dtheta/2)], [0, R*np.sin(k2_theta - dtheta/2)], 'k--')
    plt.plot([0, R*np.cos(k2_theta + dtheta/2)], [0, R*np.sin(k2_theta + dtheta/2)], 'k--')
    
    print(k2_theta)
    
    for aa in range(n_theta):
        if k2_theta >= theta[aa] - dtheta/2.0 and k2_theta < theta[aa] + dtheta/2.0:
            k2_theta = theta[aa]
    k2_theta_lwr = k2_theta - dtheta/2.0
    k2_theta_upr = k2_theta + dtheta/2.0
    ## Plot k2 sector
    plt.plot([0, R / 2 * np.cos(k2_theta)], [0, R / 2 * np.sin(k2_theta)], 'g')
    plt.plot([0, R*np.cos(k2_theta - dtheta/2)], [0, R*np.sin(k2_theta - dtheta/2)], 'g--')
    plt.plot([0, R*np.cos(k2_theta + dtheta/2)], [0, R*np.sin(k2_theta + dtheta/2)], 'g--')
    
#     plt.plot([0, R*np.cos(theta[correct_sect] - dtheta/2)], [0, R*np.sin(theta[correct_sect] - dtheta/2)], 'c--')
#     plt.plot([0, R*np.cos(theta[correct_sect] + dtheta/2)], [0, R*np.sin(theta[correct_sect] + dtheta/2)], 'c--')
    
    
    plt.plot([0, R * np.cos(theta_a + angles[l])], [0, R *np.sin(theta_a + angles[l])], 'g')
    
    
    print(k2_theta)
    
    for k3_x in kx:
        for k3_y in ky:
            ## Get polar coords
            k3_sqr   = k3_x**2 + k3_y**2
            k3_angle = np.arctan2(k3_x, k3_y)

            if (k3_sqr > 0 and k3_sqr < kmax_sqr) and (k3_angle >= k3_theta_lwr and k3_angle < k3_theta_upr):
                for k1_x in kx:
                    for k1_y in ky:
                        ## Get k1 polar coords
                        k1_sqr   = k1_x**2 + k1_y**2
                        k1_angle = np.arctan2(k1_x, k1_y)
                        
                        if (k1_sqr > 0 and k1_sqr < kmax_sqr) and (k1_angle >= k1_theta_lwr and k1_angle < k1_theta_upr):
                            ## Get k2 
                            k2_x = k3_x - k1_x
                            k2_y = k3_y - k1_y

                            ## Get k2 polar coords
                            k2_sqr   = k2_x**2 + k2_y**2
                            k2_angle = np.arctan2(k2_x, k2_y)

                            if (k2_sqr > 0 and k2_sqr < kmax_sqr) and ((k2_angle >= k2_theta_lwr and k2_angle < k2_theta_upr)):
                                ## Flux pre factor term
                                flux_pre_fac = (k1_x * k2_y - k1_y * k2_x) * (1.0 / k1_sqr - 1.0 / k2_sqr)

                                ## Flux weight term
                                flux_wght = flux_pre_fac * amps[0, kmax - 1 + k1_x, kmax - 1 + k1_y] * amps[0, kmax - 1 + k2_x, kmax - 1 + k2_y] * amps[0, kmax - 1 + k3_x, kmax - 1 + k3_y]

                                ## Triad Phase and Generalized Triad Phase
                                triad_phase = phases[0, kmax - 1 + k1_x, kmax - 1 + k1_y] + phases[0, kmax - 1 + k2_x, kmax - 1 + k2_y] - phases[0, kmax - 1 + k3_x, kmax - 1 + k3_y]
                                if k3_sqr > k1_sqr and k3_sqr > k2_sqr:
                                    gen_triad = np.mod(triad_phase + 2.0 * np.pi + np.angle(np.complex(flux_wght, 0.0)), 2.0 * np.pi) - np.pi
                                elif k3_sqr < k1_sqr and k3_sqr < k2_sqr:
                                    gen_triad = np.mod(triad_phase + 2.0 * np.pi + np.angle(-np.complex(flux_wght, 0.0)), 2.0 * np.pi) - np.pi
                                else: 
                                    gen_triad = np.mod(triad_phase + 2.0 * np.pi, 2.0 * np.pi) - np.pi
                                    
                                    
                                print("a: {} l: {} \t | k1: ({}, {}) \t k2: ({}, {}) \t k3: ({}, {}) \t | k1_sq: {:0.1f}   k2_sq: {:0.1f}   k3_sq: {:0.1f}   kmax_C_sq: {:0.1f} kmax_sq: {:0.1f} \t | k1_a: {:0.6f}   k2_a: {:0.6f}   k3_a: {:0.6f} \t | \t flux_pre_fac: {:0.5f}   flux_w: {:0.5f}   triad: {:0.5f}   gen_triad: {} ".format(a, l, k1_x, k1_y, k2_x, k2_y, k3_x, k3_y, k1_sqr, k2_sqr, k3_sqr, kmax_C_sqr, kmax_sqr, k1_angle, k2_angle, k3_angle, flux_pre_fac, flux_wght, triad_phase, gen_triad))

                                
                                ## Get the Fux contribution conditions
                                if k3_sqr > kmax_C_sqr and (k1_sqr <= kmax_C_sqr and k2_sqr <= kmax_C_sqr):
                                    ## Update the combined triads
                                    triad_phase_order_across_sec[0][a][l] += np.exp(1j * gen_triad) 
                                    
                                    ## Update the number of combined triads
                                    num_triads_across_sec[0][a][l] += 1
                                    
                                    ## Update the flux contribution
                                    enst_flux_across_sec[0][a][l] += flux_wght * np.cos(triad_phase)
                                    
                                    ## TRIAD TYPE 1
                                    if flux_pre_fac < 0:
                                        ## Update Type 1 triad order
                                        triad_phase_order_across_sec[1][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 1
                                        num_triads_across_sec[1][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 1 triads
                                        enst_flux_across_sec[1][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                    ## TRIAD TYPE 2
                                    elif flux_pre_fac > 0:
                                        ## Update Type 2 triad order
                                        triad_phase_order_across_sec[2][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 2
                                        num_triads_across_sec[2][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 2 triads
                                        enst_flux_across_sec[2][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                elif k3_sqr <= kmax_C_sqr and (k1_sqr > kmax_C_sqr and k2_sqr > kmax_C_sqr):
                                    ## Update the combined triads
                                    triad_phase_order_across_sec[0][a][l] += np.exp(1j * gen_triad) 
                                    
                                    ## Update the number of combined triads
                                    num_triads_across_sec[0][a][l] += 1
                                    
                                    ## Update the flux contribution
                                    enst_flux_across_sec[0][a][l] += flux_wght * np.cos(triad_phase)
                                    
                                    ## TRIAD TYPE 3
                                    if flux_pre_fac < 0:
                                        ## Update Type 3 triad order
                                        triad_phase_order_across_sec[3][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 3
                                        num_triads_across_sec[3][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 3 triads
                                        enst_flux_across_sec[3][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                    ## TRIAD TYPE 4
                                    elif flux_pre_fac > 0:
                                        ## Update Type 4 triad order
                                        triad_phase_order_across_sec[4][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 4
                                        num_triads_across_sec[4][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 4 triads
                                        enst_flux_across_sec[4][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                else:
                                    ## Update the phase order for triad type 5
                                    triad_phase_order_across_sec[5][a][l] += np.exp(1j * gen_triad)
                                    
                                    ## Update num of triads of type 5
                                    num_triads_across_sec[5][a][l] += 1
                                    
                                plt.plot([0, k3_y], [0, k3_x], 'b')
                                plt.plot([0, k1_y], [0, k1_x], 'r')
                                plt.plot([0, k2_y], [0, k2_x], 'g')
                                
    plt.xticks(np.arange(-10, 11))
    plt.yticks(np.arange(-10, 11))
    plt.grid(which = 'both')
    plt.show()    
    
    ## Record the sync parameters & reset triad arrays
    for i in range(6):
        ## Record sync parameters
        if num_triads_across_sec[i][a][l] != 0: 
            triad_R_across_sec[i][a][l] = np.absolute(triad_phase_order_across_sec[i][a][l] / num_triads_across_sec[i][a][l])
            triad_Phi_across_sec[i][a][l] = np.angle(triad_phase_order_across_sec[i][a][l] / num_triads_across_sec[i][a][l])
        enst_flux[i][a][l] = enst_flux_across_sec[i][a][l]
        num_triads[i][a][l] = num_triads_across_sec[i][a][l]
        
        ## Reset triad parameters for next iteration
        triad_phase_order_across_sec[i][a][l] = 0.0
        num_triads_across_sec[i][a][l] = 0
        enst_flux_across_sec[i][a][l] = 0.0
```

```python jupyter={"outputs_hidden": true}
for l in range(len(k1_sect_angles)):
    sync_across_sec(0, l)
```

```python jupyter={"outputs_hidden": true}
for a in range(n_theta):
    for l in range(len(k1_sect_angles)):
        sync_across_sec(a, l)
```

```python jupyter={"outputs_hidden": true}
for a in range(n_theta):
    for l in range(len(k1_sect_angles)):
        theta_k1 = np.mod(theta[a] + k1_sect_angles[l] * dtheta /2.0 + np.pi, 2.0 * np.pi) - np.pi
        k3_vec   = np.exp(1j * theta[a])
        k1_vec   = np.exp(1j * (theta_k1))
        k2_theta = np.real((1.0/ (2.0 * 1j) * (np.log(k3_vec - k1_vec) - np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec)))))
        
#         print("k3: {} {} I \t k1: {} {} I".format(np.real(k3_vec), np.imag(k3_vec), np.real(k1_vec), np.imag(k1_vec)))
        print("theta[{}]: {:0.16f} \t k1: {:0.16f} \t k1_theta[{}]: {:0.16f} \t mid[{}, {}]: {:0.16f}".format(a, theta[a], np.mod(theta[a] + k1_sect_angles[l] * dtheta /2.0 + np.pi, 2.0 * np.pi) - np.pi, l, theta_k1, a, l, k2_theta))
#         print()
#         print("mid[{}, {}]: {:0.16f} \t k3: {} \t k1: {} \t k3 - k1: {} \t conjk3 -conjk1: {} \t first: {} \t second: {}".format(a, l, k2_theta, k3_vec, k1_vec, k3_vec - k1_vec, np.conjugate(k3_vec) - np.conjugate(k1_vec), np.log(k3_vec - k1_vec), np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec))))

    print()
```

```python jupyter={"outputs_hidden": true}
for a in range(n_theta):
    for l in range(len(k1_sect_angles)):
        num = 0
        for type in range(6):
            print("a: {} \t l: {} \t type: {} \t Num: {} \t R: {} Phi: {}".format(a, l, type, num_triads[type][a][l], triad_R_across_sec[type][a][l], triad_Phi_across_sec[type][a][l]))
            num += num_triads[type][a][l]
        print("Num: {}".format(num))
        print()
    print()
    print("----------------------------------------------------------")
    print()
```

```python jupyter={"outputs_hidden": true}
for k3_x in kx:
    for k3_y in ky:
        print("k3: ({}, {}) - k3_a: {}".format(k3_x, k3_y, np.arctan2(k3_x, k3_y)))
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
## Theta angles
n_theta   = 24
dtheta    = 2.0 * np.pi / n_theta
theta     = np.arange(-np.pi, np.pi + dtheta, dtheta)
mid_theta = (theta[1:] + theta[:-1]) * 0.5
theta     = mid_theta[:]


## k1 sector angles
k1_sect_angles = theta

Nx = 32
Ny = 32
kx = np.arange(-Nx//2 + 1, Nx//2 + 1, 1)
ky = np.arange(-Ny//2 + 1, Nx//2 + 1, 1)
kmax = Nx//3
kmax_sqr = kmax**2

C_frac = 0.75
kmax_C = int(np.ceil(C_frac * kmax))
kmax_C_sqr = kmax_C ** 2

print(kmax_C_sqr)
R = kmax

x = R * np.cos(angle)
y = R * np.sin(angle)


for i in range(n_theta):
    print("thete[{}]: {} \t dtheta: {}".format(i, theta[i], dtheta))
```

```python
triad_phase_order_across_sec = np.ones((6, n_theta, len(k1_sect_angles))) * np.complex(0.0, 0.0)
triad_R_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
triad_Phi_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
num_triads_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
enst_flux_across_sec = np.zeros((6, n_theta, len(k1_sect_angles)))
enst_flux = np.zeros((6, n_theta, len(k1_sect_angles)))
num_triads = np.zeros((6, n_theta, len(k1_sect_angles)))

# angles = [-np.pi/2, -np.pi/3, -np.pi/4, -np.pi/6, np.pi/6, np.pi/4, np.pi/3, np.pi/2]

def mid_angle(a, k1):
    
    k3_vec = np.exp(1j * a)
    k1_vec = np.exp(1j * (k1))
    k2_theta = np.real((1.0/ (2.0 * 1j) * (np.log(k3_vec - k1_vec) - np.log(np.conjugate(k3_vec) - np.conjugate(k1_vec)))))
    
    return k2_theta

def sync_across_sec_full(a, l):
    
    ## Phase order data
    theta_a  = theta[a]
    if l == 0:
        theta_k1 = theta[a]
    else:
        theta_k1 = theta[np.mod((a + l), n_theta)]  ##np.mod(theta[a] + k1_sect_angles[l] * dtheta / 2.0 + np.pi, 2.0 * np.pi) - np.pi
#     if theta_k1 < -np.pi:
#         theta_k1 = np.pi - np.mod(theta_k1 + np.pi, 2.0 * np.pi)
#     elif theta_k1 > np.pi:
#         theta_k1 = -np.pi + np.mod(theta_k1 + np.pi, 2.0 * np.pi)
    print("l: {}".format(l))
   

    ## Get the k3 sector boundary angles
    k3_theta_lwr = theta_a - dtheta/2.0
    k3_theta_upr = theta_a + dtheta/2.0
    ## Get the k1 sector boundary angles
    k1_theta_lwr = theta_k1 - dtheta/2.0
    k1_theta_upr = theta_k1 + dtheta/2.0

    plt.figure(figsize = (13, 12))
    plt.plot(x, y, 'k--', alpha = 0.2)
    plt.plot([0, 0], [-R, R], 'k--', alpha = 0.2)
    plt.plot([R, -R], [0, 0], 'k--', alpha = 0.2)
    for aa in range(theta.shape[0]):
        ## Plot sectors
        plt.plot([0, R*np.cos(theta[aa] - dtheta/2)], [0, R*np.sin(theta[aa] - dtheta/2)], 'r--', alpha = 0.1)
        plt.plot([0, R*np.cos(theta[aa] + dtheta/2)], [0, R*np.sin(theta[aa] + dtheta/2)], 'r--', alpha = 0.1)
    
    ## Plot the k3 sector
    plt.plot([0, R*np.cos(theta_a)], [0, R*np.sin(theta_a)], 'b')
    plt.plot([0, R*np.cos(theta_a - dtheta/2)], [0, R*np.sin(theta_a - dtheta/2)], 'b--')
    plt.plot([0, R*np.cos(theta_a + dtheta/2)], [0, R*np.sin(theta_a + dtheta/2)], 'b--')

    ## Plot the k1 sector
    plt.plot([0, R*np.cos(theta_k1)], [0, R*np.sin(theta_k1)], 'r')
    plt.plot([0, -R*np.cos(theta_k1)], [0, -R*np.sin(theta_k1)], 'r')
    plt.plot([0, R*np.cos(theta_k1 - dtheta/2)], [0, R*np.sin(theta_k1 - dtheta/2)], 'r--')
    plt.plot([0, R*np.cos(theta_k1 + dtheta/2)], [0, R*np.sin(theta_k1 + dtheta/2)], 'r--')
    
    ## Get the sector angle for k2
    print(theta_a)
    print(theta_k1, np.mod(theta_k1, 2.0 * np.pi))
    
    if l == 0:
        k2_theta = theta[a]
    else:
        k2_theta = mid_angle(theta_a, theta_k1)
#     plt.plot([0, R * np.cos(k2_theta)], [0, R * np.sin(k2_theta)], 'k')
#     plt.plot([0, R*np.cos(k2_theta - dtheta/2)], [0, R*np.sin(k2_theta - dtheta/2)], 'k--')
#     plt.plot([0, R*np.cos(k2_theta + dtheta/2)], [0, R*np.sin(k2_theta + dtheta/2)], 'k--')
    
    print(k2_theta)
    
    for aa in range(n_theta):
        if k2_theta >= theta[aa] - dtheta/2.0 and k2_theta < theta[aa] + dtheta/2.0:
            k2_theta = theta[aa]
    k2_theta_lwr = k2_theta - dtheta/2.0
    k2_theta_upr = k2_theta + dtheta/2.0
    ## Plot k2 sector
    plt.plot([0, R / 2 * np.cos(k2_theta)], [0, R / 2 * np.sin(k2_theta)], 'g')
    plt.plot([0, R*np.cos(k2_theta - dtheta/2)], [0, R*np.sin(k2_theta - dtheta/2)], 'g--')
    plt.plot([0, R*np.cos(k2_theta + dtheta/2)], [0, R*np.sin(k2_theta + dtheta/2)], 'g--')
    
    plt.plot([0, R*np.cos(theta[correct_sect] - dtheta/2)], [0, R*np.sin(theta[correct_sect] - dtheta/2)], 'c--')
    plt.plot([0, R*np.cos(theta[correct_sect] + dtheta/2)], [0, R*np.sin(theta[correct_sect] + dtheta/2)], 'c--')
    
    
    plt.plot([0, R * np.cos(theta_a + angles[l])], [0, R *np.sin(theta_a + angles[l])], 'g')
    
    
    print(k2_theta)
    
    for k3_x in kx:
        for k3_y in ky:
            ## Get polar coords
            k3_sqr   = k3_x**2 + k3_y**2
            k3_angle = np.arctan2(k3_x, k3_y)

            if (k3_sqr > 0 and k3_sqr < kmax_sqr) and (k3_angle >= k3_theta_lwr and k3_angle < k3_theta_upr):
                for k1_x in kx:
                    for k1_y in ky:
                        ## Get k1 polar coords
                        k1_sqr   = k1_x**2 + k1_y**2
                        k1_angle = np.arctan2(k1_x, k1_y)
                        
                        if (k1_sqr > 0 and k1_sqr < kmax_sqr) and (k1_angle >= k1_theta_lwr and k1_angle < k1_theta_upr):
                            ## Get k2 
                            k2_x = k3_x - k1_x
                            k2_y = k3_y - k1_y

                            ## Get k2 polar coords
                            k2_sqr   = k2_x**2 + k2_y**2
                            k2_angle = np.arctan2(k2_x, k2_y)

                            if (k2_sqr > 0 and k2_sqr < kmax_sqr) and ((k2_angle >= k2_theta_lwr and k2_angle < k2_theta_upr)):
                                ## Flux pre factor term
                                flux_pre_fac = (k1_x * k2_y - k1_y * k2_x) * (1.0 / k1_sqr - 1.0 / k2_sqr)

                                ## Flux weight term
                                flux_wght = flux_pre_fac * amps[0, kmax - 1 + k1_x, kmax - 1 + k1_y] * amps[0, kmax - 1 + k2_x, kmax - 1 + k2_y] * amps[0, kmax - 1 + k3_x, kmax - 1 + k3_y]

                                ## Triad Phase and Generalized Triad Phase
                                triad_phase = phases[0, kmax - 1 + k1_x, kmax - 1 + k1_y] + phases[0, kmax - 1 + k2_x, kmax - 1 + k2_y] - phases[0, kmax - 1 + k3_x, kmax - 1 + k3_y]
                                if k3_sqr > k1_sqr and k3_sqr > k2_sqr:
                                    gen_triad = np.mod(triad_phase + 2.0 * np.pi + np.angle(np.complex(flux_wght, 0.0)), 2.0 * np.pi) - np.pi
                                elif k3_sqr < k1_sqr and k3_sqr < k2_sqr:
                                    gen_triad = np.mod(triad_phase + 2.0 * np.pi + np.angle(-np.complex(flux_wght, 0.0)), 2.0 * np.pi) - np.pi
                                else: 
                                    gen_triad = np.mod(triad_phase + 2.0 * np.pi, 2.0 * np.pi) - np.pi
                                    
                                    
                                print("a: {} l: {} \t | k1: ({}, {}) \t k2: ({}, {}) \t k3: ({}, {}) \t | k1_sq: {:0.1f}   k2_sq: {:0.1f}   k3_sq: {:0.1f}   kmax_C_sq: {:0.1f} kmax_sq: {:0.1f} \t | k1_a: {:0.6f}   k2_a: {:0.6f}   k3_a: {:0.6f} \t | \t flux_pre_fac: {:0.5f}   flux_w: {:0.5f}   triad: {:0.5f}   gen_triad: {} ".format(a, l, k1_x, k1_y, k2_x, k2_y, k3_x, k3_y, k1_sqr, k2_sqr, k3_sqr, kmax_C_sqr, kmax_sqr, k1_angle, k2_angle, k3_angle, flux_pre_fac, flux_wght, triad_phase, gen_triad))

                                
                                ## Get the Fux contribution conditions
                                if k3_sqr > kmax_C_sqr and (k1_sqr <= kmax_C_sqr and k2_sqr <= kmax_C_sqr):
                                    ## Update the combined triads
                                    triad_phase_order_across_sec[0][a][l] += np.exp(1j * gen_triad) 
                                    
                                    ## Update the number of combined triads
                                    num_triads_across_sec[0][a][l] += 1
                                    
                                    ## Update the flux contribution
                                    enst_flux_across_sec[0][a][l] += flux_wght * np.cos(triad_phase)
                                    
                                    ## TRIAD TYPE 1
                                    if flux_pre_fac < 0:
                                        ## Update Type 1 triad order
                                        triad_phase_order_across_sec[1][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 1
                                        num_triads_across_sec[1][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 1 triads
                                        enst_flux_across_sec[1][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                    ## TRIAD TYPE 2
                                    elif flux_pre_fac > 0:
                                        ## Update Type 2 triad order
                                        triad_phase_order_across_sec[2][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 2
                                        num_triads_across_sec[2][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 2 triads
                                        enst_flux_across_sec[2][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                elif k3_sqr <= kmax_C_sqr and (k1_sqr > kmax_C_sqr and k2_sqr > kmax_C_sqr):
                                    ## Update the combined triads
                                    triad_phase_order_across_sec[0][a][l] += np.exp(1j * gen_triad) 
                                    
                                    ## Update the number of combined triads
                                    num_triads_across_sec[0][a][l] += 1
                                    
                                    ## Update the flux contribution
                                    enst_flux_across_sec[0][a][l] += flux_wght * np.cos(triad_phase)
                                    
                                    ## TRIAD TYPE 3
                                    if flux_pre_fac < 0:
                                        ## Update Type 3 triad order
                                        triad_phase_order_across_sec[3][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 3
                                        num_triads_across_sec[3][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 3 triads
                                        enst_flux_across_sec[3][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                    ## TRIAD TYPE 4
                                    elif flux_pre_fac > 0:
                                        ## Update Type 4 triad order
                                        triad_phase_order_across_sec[4][a][l] += np.exp(1j * gen_triad)
                                        
                                        ## Update num of triad type 4
                                        num_triads_across_sec[4][a][l] += 1
                                        
                                        ## Update the enstrophy flux contribution for type 4 triads
                                        enst_flux_across_sec[4][a][l] += flux_wght * np.cos(triad_phase)
                                        
                                else:
                                    ## Update the phase order for triad type 5
                                    triad_phase_order_across_sec[5][a][l] += np.exp(1j * gen_triad)
                                    
                                    ## Update num of triads of type 5
                                    num_triads_across_sec[5][a][l] += 1
                                    
                                plt.plot([0, k3_y], [0, k3_x], 'b')
                                plt.plot([0, k1_y], [0, k1_x], 'r')
                                plt.plot([0, k2_y], [0, k2_x], 'g')
                                
    plt.xticks(np.arange(-10, 11))
    plt.yticks(np.arange(-10, 11))
    plt.grid(which = 'both')
    plt.show()    
    
#     ## Record the sync parameters & reset triad arrays
#     for i in range(6):
#         ## Record sync parameters
#         if num_triads_across_sec[i][a][l] != 0: 
#             triad_R_across_sec[i][a][l] = np.absolute(triad_phase_order_across_sec[i][a][l] / num_triads_across_sec[i][a][l])
#             triad_Phi_across_sec[i][a][l] = np.angle(triad_phase_order_across_sec[i][a][l] / num_triads_across_sec[i][a][l])
#         enst_flux[i][a][l] = enst_flux_across_sec[i][a][l]
#         num_triads[i][a][l] = num_triads_across_sec[i][a][l]
        
#         ## Reset triad parameters for next iteration
#         triad_phase_order_across_sec[i][a][l] = 0.0
#         num_triads_across_sec[i][a][l] = 0
#         enst_flux_across_sec[i][a][l] = 0.0
```

```python
for l in range(len(k1_sect_angles)):
    sync_across_sec_full(n_theta - 1, l)
```

```python

```
