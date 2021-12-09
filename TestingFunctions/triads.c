#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int main (int argc, char* argv) {

	int kmin = 1;
	int kmax = 4;
	int k2_x, k2_y;
	double k_sqr, k1_sqr, k2_sqr;

	int k_x, k1_x;

	double k_angle, k1_angle, k2_angle;

	for (int tmp_k_x = 0; tmp_k_x <= 2 * kmax - 1; ++tmp_k_x) {

		// Get the real k_x value
		k_x = tmp_k_x - kmax + 1;

		// Ignore the zero mode 
		if (abs(k_x) > 0) {
			// printf("----> kx: %d -- kx/2: %d\n", k_x, (int)(k_x / 2) +1);
			for (int k_y = kmin; k_y <= kmax; ++k_y) {
				printf("\n-----> (%d, %d)\n", k_x, k_y);
				// Get polar coords of this wavevector
				k_sqr   = (double) (k_x * k_x + k_y * k_y);
				// k_angle = k_x / k_y;

				for (int tmp_k1_x = 0; tmp_k1_x <= 2 * kmax - 1; ++tmp_k1_x) {

					// Get real k1_x value
					k1_x = tmp_k1_x - kmax + 1;

					// Ignore the zero mode
					if (abs(k1_x) > 0) {
						for (int k1_y = kmin; k1_y <= kmax; ++k1_y) {
							// // Compute |k1|^2
							k1_sqr   = (double) (k1_x * k1_x + k1_y * k1_y);
							// k1_angle = k1_x / k1_y; 

							// // Compute k2		
							k2_x = k_x - k1_x;
							k2_y = k_y - k1_y;


							// // Compute |k2|^2
							k2_sqr   = (double) (k2_x * k2_x + k2_y * k2_y);
							// k2_angle = k2_x / k2_y;

							if ((k1_sqr <= kmax * kmax) && (k2_sqr <= kmax * kmax) && (abs(k2_x) > 0) && (abs(k2_y) > 0)) {   // k1_sqr < k_sqr && k1_sqr <= k2_sqr && k2_x > 0 && k2_y > 0
								printf("{(%d, %d) (%d, %d) (%d, %d)}  ", k1_x, k1_y, k2_x, k2_y, k_x, k_y);
							}
						}
					}
				}
			// printf("\n");
			}

		}
			
	}
	printf("\n");


	return 1;
}