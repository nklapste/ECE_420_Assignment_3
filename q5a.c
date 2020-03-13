#include <stdlib.h>
#include "mpi.h"

RowMatrixVectorMultiply(int n, double *a, double *b, double *x, MPI_Comm comm) {
    int i, j;
    int nlocal;        /* Number of locally stored rows of A */
    double *fb;        /* Will point to a buffer that stores the entire vector b */
    int npes, myrank;
    MPI_Status status;

    /* Get information about the communicator */
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    /* Allocate the memory that will store the entire vector b */
    fb = (double *) malloc(n * sizeof(double));
    nlocal = n / npes;

    /* Gather the entire vector b on each processor using MPI's ALLGATHER operation */
    MPI_Allgather(b, nlocal, MPI_DOUBLE, fb, nlocal, MPI_DOUBLE, comm);

    /* Perform the matrix-vector multiplication involving the locally stored sub */
    for (i = 0; i < nlocal; i++) {
        x[i] = 0.0;
        for (j = 0; j < n; j++)
            x[i] += a[i * n + j] * fb[j];
    }
    free(fb);
}
