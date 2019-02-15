// nvcc jcaobi.cu -o jacobi

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NBLOCKS 32

__global__ void compute_xnext(double *r, double *d, double *x, double *x_next,
	double *b, int size) {

	// Size defined as third arg in <<< >>> thing
	extern __shared__ double x_copy[];

	// Copy x to multiply in shared memory for quicker access
	int idx, row_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int upBound = NBLOCKS * (threadIdx.x + 1);
  if (upBound > size)
    upBound = size;
  for (idx=NBLOCKS*threadIdx.x; idx < upBound; idx++)
    x_copy[idx] = x[idx];

	if (row_idx >= size) return;

	__syncthreads();

	double y_k = 0.;
	for (idx=0; idx < size; idx++)
		y_k += x_copy[idx] * r[row_idx * size + idx];

	x_next[row_idx] = (b[row_idx] - y_k) / d[row_idx];
}

int main (int argc, char *argvs[]) {

	if (argc != 4) {
		printf("usage: %s [size] [epsilon] [verbose: 1/2]\n", argvs[0]);
		return 1;
	}

	// init constants & variables
	int nBlocks, nThreadsPerBlock, size, idx1, idx2, niter, epsilon_pow, verbose;
	size = atoi(argvs[1]);
	epsilon_pow = atoi(argvs[2]);
	verbose = atoi(argvs[3]);
	nBlocks = NBLOCKS;
	nThreadsPerBlock = (int) size / nBlocks + 1;
	niter = 0;
	double epsilon = 1.;
	for (idx1=0; idx1 < epsilon_pow; idx1++) epsilon = epsilon / 10.;
	double epsilon2 = epsilon * epsilon;
	double norm2 = epsilon2 + 1.; // to be init over espilon

	struct timeval t0, t1, t2, t3;
	size_t size_mat = size * size * sizeof(double);
	size_t size_vect = size * sizeof(double);

	// init matrices
	double *r_mat, *d_vect, *x, *x_next, *b;
	double *d_r, *d_d, *d_x, *d_xnext, *d_b;
	r_mat = (double *) calloc(size * size, sizeof(double));
	d_vect = (double *) calloc(size, sizeof(double));
	x = (double *) calloc(size, sizeof(double));
	x_next = (double *) calloc(size, sizeof(double));
	b = (double *) calloc(size, sizeof(double));

	for (idx1=0; idx1<size; idx1++) {
		for (idx2=0; idx2<size; idx2++)
			r_mat[idx1*size + idx2] = 1.;
		r_mat[idx1*size + idx1] = 0.;
		d_vect[idx1] = 2. * size + 1.;
		x[idx1] = 1.;
		b[idx1] = 3.;
	}

	// Allocation & copy on device
	gettimeofday(&t0,NULL);
	cudaMalloc(&d_r, size_mat);
	cudaMalloc(&d_d, size_vect);
	cudaMalloc(&d_x, size_vect);
	cudaMalloc(&d_xnext, size_vect);
	cudaMalloc(&d_b, size_vect);

	gettimeofday(&t1,NULL);
	cudaMemcpy(d_r, r_mat, size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d_vect, size_vect, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size_vect, cudaMemcpyHostToDevice);

	gettimeofday(&t2,NULL);

	// Main Loop
	while (norm2 > epsilon2) {
		niter ++;
		// Send x_k
		cudaMemcpy(d_x, x, size_vect, cudaMemcpyHostToDevice);
		compute_xnext <<<nBlocks, nThreadsPerBlock, size_vect >>> (d_r, d_d,
			d_x, d_xnext, d_b, size);
			// Gather x_k+1
		cudaMemcpy(x_next, d_xnext, size_vect, cudaMemcpyDeviceToHost);

		norm2 = 0.;
		for (idx1=0; idx1 < size; idx1++) {
			norm2 += (x[idx1] - x_next[idx1]) * (x[idx1] - x_next[idx1]);
			x[idx1] = x_next[idx1];
		}
	}

	gettimeofday(&t3,NULL);

	cudaFree(d_r);
	cudaFree(d_d);
	cudaFree(d_x);
	cudaFree(d_xnext);
	cudaFree(d_b);

	// Verbose
	double t_alloc = (double)(t1.tv_sec-t0.tv_sec) + \
		(double)(t1.tv_usec-t0.tv_usec)/1000000;
	double t_trans = (double)(t2.tv_sec-t1.tv_sec) + \
		(double)(t2.tv_usec-t1.tv_usec)/1000000;
	double t_calc  = (double)(t3.tv_sec-t2.tv_sec) + \
		(double)(t3.tv_usec-t2.tv_usec)/1000000;

	if (verbose == 1) {
		printf("N = %d\nEpsilon = %10.9f\n", size, epsilon);
		printf("Nombre d'iterations: %d\n", niter);
		double norm_error = 0.;
		for (int i=0; i < size; i++)
			norm_error += (x[i] - 1./size) * (x[i] - 1./size);
		printf("Error compared to sol.: %25.24f\n\n", norm_error);
		printf("Temps d'alloc. device   : %f s\n", t_alloc);
		printf("Temps de transfert init.: %f s\n", t_trans);
		printf("Temps de calcul:          %f s\n", t_calc);
	} else if (verbose == 2)
		printf("%d, %12.11f, %d, %f\n", size, epsilon, niter, t_calc);

	free(r_mat);
	free(d_vect);
	free(x);
	free(x_next);
	free(b);

	return 0;
}
