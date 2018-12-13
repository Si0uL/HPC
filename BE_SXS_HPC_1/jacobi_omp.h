// gcc --std=c99 -Wall -fopenmp jacobi.c -o jacobi.out

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

/* x should be initialised to [1., ..., 1.] */
int jacobi(double *A, double *b, double *x, int size,
	double epsilon2);

/* x should be initialised to [1., ..., 1.] */
int jacobi_parallel(double *A, double *b, double *x, int size,
	double epsilon2);

/* x should be initialised to [1., ..., 1.] */
int gauss_seidel(double *A, double *b, double *x,	int size, double epsilon2);
