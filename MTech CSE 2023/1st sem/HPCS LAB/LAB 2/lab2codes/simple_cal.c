#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int size, rank;
    int a = 4, b = 2, c = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        c = a + b;
        printf("\nPrinting from process: %d|#|", rank);
        printf(" a: %d, b: %d, c = a + b: %d\n", a, b, c);
    } else if (rank == 1) {
        c = a - b;
        printf("\nPrinting from process: %d|#|", rank);
        printf(" a: %d, b: %d, c = a - b: %d\n", a, b, c);
    } else if (rank == 2) {
        c = a * b;
        printf("\nPrinting from process: %d|#|", rank);
        printf(" a: %d, b: %d, c = a * b: %d\n", a, b, c);
    } else if (rank == 3) {
        c = a / b;
        printf("\nPrinting from process: %d|#|", rank);
        printf(" a: %d, b: %d, c = a / b: %d\n", a, b, c);
    }

    MPI_Finalize();
    return 0;
}