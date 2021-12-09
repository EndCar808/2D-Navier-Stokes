import numpy as np
from numpy.linalg import norm

kmax   = 4
krange  = np.arange(-kmax + 1, kmax + 1, 1)
kxrange = np.arange(-kmax + 1, kmax + 1, 1)
kyrange = np.arange(1, kmax + 1, 1)
for kx in kxrange:
	if np.absolute(kx) > 0:
		for ky in kyrange:
			# if abs(ky) > 0:
			print("\n----> ({}, {})".format(kx, ky))
			for k1x in kxrange:
				if np.absolute(k1x) > 0:
					for k1y in kyrange:
						for k2x in kxrange:
							if np.absolute(k2x) > 0:
								for k2y in kyrange:
									k1_vec = np.array([k1x, k1y])
									k2_vec = np.array([k2x, k2y])
									k_vec = np.array([kx, ky])
							
									if np.array_equal(k1_vec + k2_vec, k_vec):  # norm(k1_vec) < norm(k_vec) and norm(k1_vec) <= norm(k2_vec) and
										print("({}, {}) ({}, {}) ({}, {})".format(k1x, k1y, k2x, k2y, kx, ky), end = "  ")
print()
print()


for kx in krange:
	# print(kx, int(np.ceil(kx / 2)), np.arange(1, int(np.ceil(kx / 2)) + 1))
	for ky in krange:
		for k1x in np.arange(1, int(np.ceil(kx / 2)) + 1):
			for k1y in np.arange(1, int(np.ceil(ky / 2)) + 1):
				k_vec  = np.array([kx, ky])
				k1_vec = np.array([k1x, k1y])
				k2_vec = np.array(k_vec - k1_vec)		
				if k2_vec[0] > 0 and k2_vec[1] > 0: # norm(k1_vec) < norm(k_vec) and norm(k1_vec) <= norm(k2_vec) and
					print("({}, {}) ({}, {}) ({}, {})".format(k1x, k1y, k2_vec[0], k2_vec[1], kx, ky), end = "  ")
		print()
print()