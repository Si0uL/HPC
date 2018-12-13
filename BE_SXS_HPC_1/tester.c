// after compiling jacobi.o
// gcc --std=c99 -Wall -fopenmp jacobi_omp.o tester.c -o tester.out

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#include "jacobi_omp.h"

void run_test (int size, int method, int n_threads, double epsilon, double *A,
	double *b, double *x) {

	// Re-init x
	for (int i=0; i < size; i++) x[i] = 1.;

	double exec_time = omp_get_wtime();

	int niter;

	if (method == 1) {
		niter = jacobi(A, b, x, size, epsilon*epsilon);
	} else if (method == 2) {
		omp_set_num_threads(n_threads);
		niter = jacobi_parallel(A, b, x, size, epsilon*epsilon);
	} else if (method == 3) {
		niter = gauss_seidel(A, b, x, size, epsilon*epsilon);
	} else {
		printf("Invalid method number (use 1 / 2 / 3)\n");
		return ;
	}

	exec_time = omp_get_wtime() - exec_time;

	printf("%d, %12.11f, %d, %d, %f\n", size, epsilon, n_threads, niter,
		exec_time);

}


int main (int argc, char *argv[]) {

	/****************************************************************************/
	// Init of parameters
	int METHOD = 2; // 2 for Jacobi_parallel & 3 for Gauss-Seidel
	int fixed_size = 1024;
	double fixed_epsilon = 1./100000000;

	// /!\ WARNING: change both l_x and x not to have undetermined behaviour
	int l_sizes = 6, l_n_threads = 7, l_epsilons = 7;
	int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
	int n_threads[] = {1, 2, 4, 8, 16, 32, 64};
	double epsilons[] = {1./100000, 1./1000000, 1./10000000,
		1./100000000, 1./1000000000, 1./10000000000, 1./100000000000};
	/****************************************************************************/

	// Allocation for fixed size tests
	double *A = malloc(fixed_size * fixed_size * sizeof(double));
	double *b = malloc(fixed_size * sizeof(double));
	double *x = malloc(fixed_size * sizeof(double));

	for (int i=0; i < fixed_size; i++) {

		for (int j=0; j < fixed_size; j++)
			*(A + i*fixed_size + j) = 1.;

		*(A + i*fixed_size + i) = 2. * fixed_size + 1.;
		x[i] = 1.;
		b[i] = 3.;
	}

	printf("-----------Epsilon---------------\n");

	// Loop on epsilon & threads, size fixed to fixed_size
	for (int eidx=0; eidx < l_epsilons; eidx++) {
		for (int tidx=0; tidx < l_n_threads; tidx ++)
			run_test(fixed_size, METHOD, n_threads[tidx], epsilons[eidx], A, b, x);
	}

	free(A);
	free(b);
	free(x);

	printf("-----------Size---------------\n");

	// Loop on N & threads, Epsilon=10^-8
	for (int nidx; nidx < l_sizes; nidx++) {
		double *A = malloc(sizes[nidx] * sizes[nidx] * sizeof(double));
		double *b = malloc(sizes[nidx] * sizeof(double));
		double *x = malloc(sizes[nidx] * sizeof(double));

		for (int i=0; i < sizes[nidx]; i++) {

			for (int j=0; j < sizes[nidx]; j++)
				A[i*sizes[nidx] + j] = 1.;

			A[i * sizes[nidx] + i] = 2. * sizes[nidx] + 1.;
			x[i] = 1.;
			b[i] = 3.;
		}

		// Tests
		for (int tidx=0; tidx < l_sizes; tidx ++)
			run_test(sizes[nidx], METHOD, n_threads[tidx], fixed_epsilon, A, b, x);

		free(A);
		free(b);
		free(x);

	}

	return 0;
}
