// gcc --std=c99 -Wall -fopenmp -c jacobi_omp.c

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#include "jacobi_omp.h"

/* x should be initialised to [1., ..., 1.] */
/* Became obsolete jacobi_parallel with n_thread = 1 give the same
 * results with n high enough */
int jacobi(double *A, double *b, double *x, int size,
	double epsilon2) {

	double *y = calloc(size, sizeof(double));
	double last_xi, norm = epsilon2 + 1;
	int idx1, idx2, n_iter = 0;

	while (norm > epsilon2) {
		// deriving y[]
		for (idx1=0; idx1 < size; idx1++) {
			y[idx1] = 0.;
			for (idx2=0; idx2 < size; idx2++)
				y[idx1] += A[size*idx1 + idx2] * x[idx2];
			// not to break the pipeline
			y[idx1] -= A[size*idx1 + idx1] * x[idx1];
		}

		norm = 0.;
		for (idx1=0; idx1 < size; idx1++) {
			last_xi = x[idx1];
			x[idx1] = (b[idx1] - y[idx1]) / A[size*idx1 + idx1];
			norm += (last_xi - x[idx1]) * (last_xi - x[idx1]);
		}
		n_iter ++;
	}
	free(y);

	return n_iter;
}

int jacobi_parallel(double *A, double *b, double *x, int size,
	double epsilon2) {

	double *y = calloc(size, sizeof(double));
	double last_xi, norm = epsilon2 + 1;
	int idx1, idx2, n_iter = 0;

	while (norm > epsilon2) {
		// deriving y[]
		#pragma omp parallel for default(none) private(idx1, idx2) \
			shared(A, b, size, x, y)
		for (idx1=0; idx1 < size; idx1++) {
			y[idx1] = 0.;
			for (idx2=0; idx2 < size; idx2++)
				y[idx1] += A[size*idx1 + idx2] * x[idx2];
			// not to break the pipeline
			y[idx1] -= A[size*idx1 + idx1] * x[idx1];
		}

		norm = 0.;
		for (idx1=0; idx1 < size; idx1++) {
			last_xi = x[idx1];
			x[idx1] = (b[idx1] - y[idx1]) / A[size*idx1 + idx1];
			norm += (last_xi - x[idx1]) * (last_xi - x[idx1]);
		}
		n_iter ++;

	}

	free(y);

	return n_iter;
}

int gauss_seidel(double *A, double *b, double *x,	int size, double epsilon2) {

	double last_xi, pseudo_y, norm = epsilon2 + 1.;
	int idx1, idx2, n_iter = 0;

	while (norm > epsilon2) {
		norm = 0.;
		// deriving y[]
		for (idx1=0; idx1 < size; idx1++) {
			last_xi = x[idx1];
			pseudo_y = 0.;
			for (idx2=0; idx2 < size; idx2++)
				pseudo_y += A[size*idx1 + idx2] * x[idx2];
			// not to break the pipeline
			pseudo_y -= A[size*idx1 + idx1] * x[idx1];
			x[idx1] = (b[idx1] - pseudo_y) / A[size*idx1 + idx1];
			norm += (last_xi - x[idx1]) * (last_xi - x[idx1]);
		}
		n_iter += 1;

	}
	return n_iter;
}
