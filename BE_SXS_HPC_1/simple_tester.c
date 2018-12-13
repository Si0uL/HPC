// after compiling jacobi.o
//gcc --std=c99 -Wall -fopenmp jacobi_omp.o simple_tester.c -o simple_tester.out

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#include "jacobi_omp.h"

int main (int argc, char *argv[]) {

	if (argc < 4) {
		printf("Usage: %s [size] [method(1/2/3)] [n_threads]\n", argv[0]);
		return 1;
	}

	int N, n_threads, method;
	N = atoi(argv[1]);
	method = atoi(argv[2]);
	n_threads = atoi(argv[3]);

	omp_set_num_threads(n_threads);

	// Init of parameters
	double *A = calloc(N * N, sizeof(double));
	double *b = calloc(N, sizeof(double));
	double *x = calloc(N, sizeof(double));
	double epsilon2 = 1./10000000000000000; // 10^-16 -> epsilon = 10^-8

	for (int i=0; i < N; i++) {

		for (int j=0; j < N; j++) {
			A[i*N + j] = 1.;
		}

		A[i*N + i] = 2. * N + 1.;
		x[i] = 1.;
		b[i] = 3.;
	}
	// EO init

	double exec_time = omp_get_wtime();

	int niter;

	if (method == 1) {
		niter = jacobi(A, b, x, N, epsilon2);
	} else if (method == 2) {
		niter = jacobi_parallel(A, b, x, N, epsilon2);
	} else if (method == 3) {
		niter = gauss_seidel(A, b, x, N, epsilon2);
	} else {
		printf("Usage: %s [size] [method(1/2/3)] [n_threads]\n", argv[0]);
		return 2;
	}

	exec_time = omp_get_wtime() - exec_time;

	printf("Method %d\n", method);
	printf("N = %d\nEpsilon^2 = %20.19f\n", N, epsilon2);
	printf("Execution time: %f\n", exec_time);
	printf("Number of iterations: %d\n", niter);

	double norm_error = 0.;
	for (int i=0; i < N; i++)
		norm_error += (x[i] - 1./N) * (x[i] - 1./N);
	printf("Error compared to sol.: %25.24f\n\n", norm_error);

	free(A);
	free(b);
	free(x);

	return 0;
}
