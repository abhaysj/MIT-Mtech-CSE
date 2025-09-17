#include <stdio.h>
#include <mpi.h>

int is_prime(int n) {
    if (n <= 1)
        return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0)
            return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            printf("Please run with exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    int start = rank * 49 + 2; // Start from 2 or 51
    int end = start + 48;      // End at 50 or 100

    for (int num = start; num <= end; num++) {
        if (is_prime(num)) {
            printf("Process %d found prime: %d\n", rank, num);
        }
    }

    MPI_Finalize();
    return 0;
}
