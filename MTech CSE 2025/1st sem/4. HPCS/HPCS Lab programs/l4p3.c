#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 4

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int matrix[n][n];
    int local_row[n], result_row[n];
    int gathered[n][n];

    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        fflush(stdout);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    // Scatter rows of the matrix to each process
    MPI_Scatter(matrix, n, MPI_INT, local_row, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform prefix sum across processes for each row
    MPI_Scan(local_row, result_row, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Gather the result rows back to root
    MPI_Gather(result_row, n, MPI_INT, gathered, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Root prints the final matrix
    if (rank == 0) {
        printf("The output matrix is:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", gathered[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
