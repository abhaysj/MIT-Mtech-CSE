#include <stdio.h>
#include <mpi.h>

int fact(int number) {
    int fact = 1;
    while (number != 0) {
        fact = fact * number;
        number -= 1;
    }
    return fact;
}

int main(int argc, char *argv[]) {
    int size, rank, number;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[size];

    if (rank == 0) {
        printf("Enter the array:\n");
        for (int i = 0; i < size; i++)
            scanf("%d", &arr[i]);
    }

    MPI_Scatter(arr, 1, MPI_INT, &number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // printf("rank %d recv number %d\n", rank, number);

    printf("rank %d, number: %d, fact: %d\n", rank, number, fact(number));

    MPI_Finalize();
    return 0;
}

