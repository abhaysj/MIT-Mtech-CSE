#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10; // Define the number of elements in the array
    int numberToFind = 5; // The number to search for
    int* arr = NULL;

    if (rank == 0) {
        arr = (int*)malloc(N * sizeof(int));
        printf("Enter %d elements:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &arr[i]);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numberToFind, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localResult = 0;
    if (rank != 0) {
        arr = (int*)malloc(N * sizeof(int));
    }

    MPI_Bcast(arr, N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N; i++) {
        if (arr[i] == numberToFind) {
            localResult = 1;
            break;
        }
    }

    int globalResult;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (globalResult == 1) {
            printf("Number %d found in the array.\n", numberToFind);
        } else {
            printf("Number %d not found in the array.\n", numberToFind);
        }
        free(arr);
    }

    MPI_Finalize();
    return 0;
}
