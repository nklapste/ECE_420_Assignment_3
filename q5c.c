#include <stdlib.h>
#include <math.h>

void MatrixVectorMultiply_2D(int n, double *a, double *b, double *x, MPI_Comm comm) {
    int ROW = 0, COL = 1; /* Improve readability */
    int i, j, nlocal;
    double *px; /* Will store partial dot products */

    int npes, dims[2], periods[2], keep_dims[2];
    int myrank, my2drank, mycoords[2];
    int other_rank, coords[2];

    MPI_Status status;
    MPI_Comm comm_2d, comm_row, comm_col;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    dims[ROW] = dims[COL] = sqrt(npes);

    nlocal = n / dims[ROW];

    px = malloc(nlocal * sizeof(double));
    periods[ROW] = periods[COL] = 1;

    //Create a 2D Cartesian topology and get the rank and coordinates of the process
    MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);
    MPI_Comm_rank(comm_2d, &my2drank); /* Get my rank in the new topology */
    MPI_Cart_coords(comm_2d, my2drank, 2, mycoords); /* Get my coordinates */

    /* Create the row-based sub-topology*/
    keep_dims[ROW] = 0;
    keep_dims[COL] = 1; /* Column is still connected*/
    MPI_Cart_sub(comm_2d, keep_dims, &comm_row);

    /* Create the column-based sub-topology*/
    keep_dims[ROW] = 1;
    keep_dims[COL] = 0;
    MPI_Cart_sub(comm_2d, keep_dims, &comm_col);

    /* Redistribute the b vector. The vector b is in the first column*/
    /* Step 1. The processes along the 0th column send their data to the diagonal processes. */
    if (mycoords[COL] == 0 && mycoords[ROW] != 0) {
        coords[ROW] = mycoords[ROW];
        coords[COL] = mycoords[ROW];
        MPI_Cart_rank(comm_2d, coords, &other_rank);
        MPI_Send(b, nlocal, MPI_DOUBLE, other_rank, 1, comm_2d);
    }
    if (mycoords[ROW] == mycoords[COL] && mycoords[ROW] != 0) {
        coords[ROW] = mycoords[ROW];
        coords[COL] = 0;
        MPI_Cart_rank(comm_2d, coords, &other_rank);
        MPI_Recv(b, nlocal, MPI_DOUBLE, other_rank, 1, comm_2d, &status);
    }

    /* Step 2. The diagonal processes perform a column-wise broadcast*/
    coords[0] = mycoords[COL];
    MPI_Cart_rank(comm_col, coords, &other_rank);
    MPI_Bcast(b, nlocal, MPI_DOUBLE, other_rank, comm_col);

    for (i = 0; i < nlocal; i++) {
        px[i] = 0.0;
        for (j = 0; j < nlocal; j++) {
            px[i] += a[i * nlocal + j] * b[j];
        }
    }

    /* Perform the sum-reduction along the rows to add up the partial dot products */
    coords[0] = mycoords[ROW];
    MPI_Cart_rank(comm_row, coords, &other_rank);
    MPI_Reduce(px, x, nlocal, MPI_DOUBLE, MPI_SUM, other_rank, comm_row);

    MPI_Comm_free(&comm_2d); /* Free up communicator */
    MPI_Comm_free(&comm_row); /* Free up communicator */
    MPI_Comm_free(&comm_col); /* Free up communicator */
    free(px);
}
