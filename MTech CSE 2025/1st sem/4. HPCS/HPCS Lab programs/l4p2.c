#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 3

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int matrix[n][n];
    int key;
    int local_row[n];
    int local_count = 0, total_count = 0;
    fflush(stdout);
    if (rank == 0) {
        printf("Enter the 3x3 matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        
        printf("Enter the element to be searched: ");
        fflush(stdout);
        scanf("%d", &key);
    }

    // Broadcast the key to all processes
    MPI_Bcast(&key, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of the matrix (each process gets 1 row)
    MPI_Scatter(matrix, n, MPI_INT, local_row, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts occurrences of 'key' in its row
    for (int j = 0; j < n; j++) {
        if (local_row[j] == key) {
            local_count++;
        }
    }

    // Reduce to get total count at root
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    fflush(stdout);
    if (rank == 0) {
        printf("Total occurrences of %d = %d\n", key, total_count);
    }

    MPI_Finalize();
    return 0;
}
