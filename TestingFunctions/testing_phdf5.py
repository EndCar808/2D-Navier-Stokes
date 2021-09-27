import h5py
import numpy as np


file = h5py.File("/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/2D_NavierStokes/Data/Test/Test_N[8,8]_ITERS[100].h5", 'r')

w = file["w"][:, :]

w_r = np.reshape(w, (w.shape[0], 8, 8))


for t in range(w.shape[0]):
    for i in range(8):
        for j in range(8):
            print("w[{}]: {:0.16f}\t".format(i * 8 + j, w_r[t, i, j]), end = "")
        print()
    print()
    print()