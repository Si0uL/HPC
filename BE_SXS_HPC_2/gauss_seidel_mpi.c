/*
 * mpicc gauss_seidel_mpi.c -o gauss_seidel_mpi.out --std=c99 -Wall
 * mpirun -np n_proc ./gauss_seidel_mpi.out
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
	int idx1, idx2, ridx, x_idx, diag_idx, n_iter = 0;
	// Only for master
	double *A, *b;
	// Common
	double *x, *new_x, *loc_A, *loc_b;

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
	new_x = calloc(size, sizeof(double));
	loc_A = calloc(size * block_length, sizeof(double));
	loc_b = calloc(block_length, sizeof(double));

	if (rank == 0) {

		A = calloc(size * size, sizeof(double));
		b = calloc(size, sizeof(double));

		for (idx1=0; idx1 < real_size; idx1++) {

			for (idx2=0; idx2 < real_size; idx2++)
				A[idx1*size + idx2] = 1.;

			for (idx2=real_size; idx2 < size; idx2++)
				A[idx1*size + idx2] = 0.;

			A[idx1*size + idx1] = 2. * real_size + 1.;
			b[idx1] = 3.;
		}

		/* Complete with zeros except on the diagonal of A to reach the real
		 * problem size (whilst not affecting the norm2 calculation)
		 */
		 for (idx1=real_size; idx1 < size; idx1++) {

 			for (idx2=0; idx2 < size; idx2++)
 				A[idx1*size + idx2] = 0.;

 			A[idx1*size + idx1] = 1.;
 			b[idx1] = 0.;
 		}
	}

	// Init x for everybody
	for (idx1=0; idx1 < real_size; idx1++)
		x[idx1] = 1.;
	for (idx1=real_size; idx1 < size; idx1++)
		x[idx1] = 0.;
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

		// Compute my chunk of x_k+1 (only the term using previous results)
		// idx1 in [0, block_length[ ; idx2 in ]diag_idx, size[
		for (idx1=0; idx1 < block_length; idx1++) {
			diag_idx = rank * block_length + idx1;
			new_x[diag_idx] = 0.;
			for (idx2=diag_idx + 1; idx2 < size; idx2++)
				new_x[diag_idx] += loc_A[size*idx1 + idx2] * x[idx2];
		}

		// Compute the other term of x_k+1 (using new results) and send it to other
		// workers
		for (ridx=0; ridx < n_proc; ridx++) {
			if (rank == ridx) {
				// Compute missing part of x_k+1 (using previous x_k+1 terms)
				// idx1 in [0, block_length[ / idx2 in [rank * block_length, diag_idx[
				for (idx1=0; idx1 < block_length; idx1++) {
					diag_idx = rank * block_length + idx1;
					for (idx2=rank * block_length; idx2 < diag_idx; idx2++)
						new_x[diag_idx] += loc_A[size*idx1 + idx2] * new_x[idx2];

					new_x[diag_idx] = (loc_b[idx1] - new_x[diag_idx]) /
						loc_A[size*idx1 + diag_idx];
				}
			}
			// Send this to other to unlock them
			MPI_Bcast(&new_x[ridx*block_length], block_length, MPI_DOUBLE, ridx,
				MPI_COMM_WORLD);
			// Use received part to compute part of f term
			if (rank > ridx) {
				// idx1 in [0, block_length[
				// idx2 in [ridx * block_length, (ridx + 1) * block_length[
				for (idx1=0; idx1 < block_length; idx1++) {
					x_idx = rank * block_length + idx1;
					for (idx2=ridx * block_length; idx2 < (ridx+1) * block_length; idx2++)
						new_x[x_idx] += loc_A[size*idx1 + idx2] * new_x[idx2];
				}
			}
		}

		// Computes norm^2
		if (rank == 0) {
			norm2 = 0.;
			for (idx1=0; idx1 < real_size; idx1++)
				norm2 += (new_x[idx1] - x[idx1]) * (new_x[idx1] - x[idx1]);
		}

		// Share norm2
		MPI_Bcast(&norm2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Copy next_x into x
		for (idx1=0; idx1 < size; idx1++)
			x[idx1] = new_x[idx1];

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
	}

	free(x);
	free(new_x);
	free(loc_A);
	free(loc_b);

	MPI_Finalize();

	return 0;
}
