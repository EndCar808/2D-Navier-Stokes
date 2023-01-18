import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


N = 8
Nf = N//2
kx = np.arange(-Nf +1, Nf + 1, dtype="int64")
ky = np.arange(-Nf +1, Nf + 1, dtype="int64")


for i in range(N):
    for j in range(N):
        print("({}, {})".format(kx[j], ky[i]), end=" ")
    print()
