#include <stdlib.h>

ColMatrixVectorMultiply(int n, double *a, double *b, double *x, MPI_Comm comm) {
    int i, j;
    int nlocal;
    double *px;
    double *fx;
    int npes, myrank;
    MPI_Status status;

    nlocal = n / npes;

    /* Get identity and size information from the communicator */
    MPI_Comm_size(comm, &npes);

    /* Allocate memory for arrays storing intermediate results. */
    MPI_Comm_rank(comm, &myrank);
    px = (double *) malloc(n * sizeof(double));
    fx = (double *) malloc(n * sizeof(double));

    /* Compute the partial-dot products that correspond to the local columns of A.*/
    for (i = 0; i < n; i++) {
        px[i] = 0.0;
        for (j = 0; j < nlocal; j++)
            px[i] += a[i * nlocal + j] * b[j];
    }

    /* Sum-up the results by performing an element-wise reduction operation */
    MPI_Reduce(px, fx, n, MPI_DOUBLE, MPI_SUM, 0, comm);

    /* Redistribute fx in a fashion similar to that of vector b */
    MPI_Scatter(fx, nlocal, MPI_DOUBLE, x, nlocal, MPI_DOUBLE, 0, comm);
    free(px);
    free(fx);
}
