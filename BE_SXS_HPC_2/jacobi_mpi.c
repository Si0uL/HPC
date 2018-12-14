/*
 * mpicc jacobi_mpi.c -o jacobi_mpi.out --std=c99 -Wall
 * mpirun -np n_proc ./jacobi_mpi.out
 * scp jacobi_mpi.c sxs1@serv-prol1:~/Desktop/be-mpi/jacobi_mpi.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

int main (int argc, char *argvs[]) {

	if (argc != 4) {
		printf("Usage: %s [size] [epsilon] [verbose: 0/1]\n", argvs[0]);
		return 1;
	}

	int real_size, verbose, epsilon_power;
	real_size = atoi(argvs[1]);
	epsilon_power = atoi(argvs[2]);
	verbose = atoi(argvs[3]);

	// Init of parameters
	struct timeval t0, t1, t2;
	double epsilon = 1.;
	for (int i=0; i < epsilon_power; i++) epsilon = epsilon / 10.;
	double epsilon2 = epsilon * epsilon;
	double norm2 = epsilon2 + 1.; // to be > epsilon2
	int idx1, idx2, diag_idx, n_iter = 0;
	// Only for master
	double *A, *b, *next_x;
	// Common
	double *x, *loc_A, *loc_b, *loc_x, *loc_y;

	MPI_Init(&argc, &argvs);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n_proc;
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	// Closest multiple of n_proc over real_size
	int size = real_size;
	if (real_size % n_proc != 0)
		size = real_size + n_proc - real_size % n_proc;

	int block_length = size / n_proc;

	x = calloc(size, sizeof(double));
	loc_A = calloc(size * block_length, sizeof(double));
	loc_b = calloc(block_length, sizeof(double));
	loc_x = calloc(block_length, sizeof(double));
	loc_y = calloc(block_length, sizeof(double));

	if (rank == 0) {

		A = calloc(size * size, sizeof(double));
		b = calloc(size, sizeof(double));
		next_x = calloc(size, sizeof(double));

		for (idx1=0; idx1 < real_size; idx1++) {

			for (idx2=0; idx2 < real_size; idx2++)
				A[idx1*size + idx2] = 1.;

			for (idx2=real_size; idx2 < size; idx2++)
				A[idx1*size + idx2] = 0.;

			A[idx1*size + idx1] = 2. * real_size + 1.;
			x[idx1] = 1.;
			b[idx1] = 3.;
		}

		/* Complete with zeros except on the diagonal of A to reach the real
		 * problem size (whilst not affecting the norm2 calculation)
		 */
		 for (idx1=real_size; idx1 < size; idx1++) {

 			for (idx2=0; idx2 < size; idx2++)
 				A[idx1*size + idx2] = 0.;

 			A[idx1*size + idx1] = 1.;
 			x[idx1] = 0.;
 			b[idx1] = 0.;
 		}
	}
	// EO init

	if (rank == 0)
		gettimeofday(&t0, NULL);

	// Split A & b amongst processes
	MPI_Scatter(A, size * block_length, MPI_DOUBLE,
							loc_A, size * block_length, MPI_DOUBLE,
							0, MPI_COMM_WORLD);
	MPI_Scatter(b, block_length, MPI_DOUBLE,
							loc_b, block_length, MPI_DOUBLE,
							0, MPI_COMM_WORLD);

	if (rank == 0) gettimeofday(&t1, NULL);

	while (norm2 > epsilon2) {

		// Send new x_k to everybody
		MPI_Bcast(x, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Compute my chunk of y_k
		for (idx1=0; idx1 < block_length; idx1++) {
			loc_y[idx1] = 0.;
			for (idx2=0; idx2 < size; idx2++)
				loc_y[idx1] += loc_A[size*idx1 + idx2] * x[idx2];
			diag_idx = rank*block_length + idx1;
			// Not to break the pipeline
			loc_y[idx1] -= loc_A[size*idx1 + diag_idx] * x[diag_idx];
			// Use it to compute my chunk of x_(k+1)
			loc_x[idx1] = (loc_b[idx1] - loc_y[idx1]) /	loc_A[size*idx1 + diag_idx];
		}

		// Gather chunks of x_k+1
		MPI_Gather(loc_x, block_length, MPI_DOUBLE,
				   next_x, block_length, MPI_DOUBLE,
				   0, MPI_COMM_WORLD);

		// Computes norm^2
		if (rank == 0) {

			norm2 = 0.;
			for (idx1=0; idx1 < real_size; idx1++)
				norm2 += (next_x[idx1] - x[idx1]) * (next_x[idx1] - x[idx1]);

		}

		// Share norm2
		MPI_Bcast(&norm2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			// Copy next_x into next_x (further use in norm(delta))
			for (idx1=0; idx1 < size; idx1++)
				x[idx1] = next_x[idx1];
		}

		n_iter ++;

	}

	if (rank == 0)
		gettimeofday(&t2, NULL);

	if (rank == 0) {

		// Verbose
		float t_send = (float)(t1.tv_sec - t0.tv_sec) + \
			(float)(t1.tv_usec - t0.tv_usec) / 1000000;
		float t_calc = (float)(t2.tv_sec - t1.tv_sec) + \
			(float)(t2.tv_usec - t1.tv_usec) / 1000000;
		if (verbose) {
			printf("N = %d\nEpsilon = %10.9f\n", size, epsilon);
			printf("Number of iterations: %d\n", n_iter);

			double norm_error = 0.;
			for (int i=0; i < real_size; i++)
				norm_error += (x[i] - 1./real_size) * (x[i] - 1./real_size);
			printf("Error compared to sol.: %25.24f\n\n", norm_error);

			printf("Temps de transfert init.: %f s\nTemps de calcul:          %f s\n",
				t_send, t_calc);
		} else {
			printf("%d, %12.11f, %d, %d, %f\n", real_size, epsilon, n_proc, n_iter,
				t_calc);
		}

		// Deallocation
		free(A);
		free(b);
		free(next_x);
	}

	free(x);
	free(loc_A);
	free(loc_b);
	free(loc_x);
	free(loc_y);

	MPI_Finalize();

	return 0;
}
