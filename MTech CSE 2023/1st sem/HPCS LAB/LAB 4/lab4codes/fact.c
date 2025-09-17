#include <stdio.h>
#include <mpi.h>

int factorial(int number) {
    int fact = 1;
    while(number != 0) {
        fact = fact * number;
        number -= 1;
    }
    return fact;
}

int main(int argc, char *argv[]) {
    int size, rank;
    int sum_fact = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1 = MPI_Wtime();
    int fact = factorial(rank + 1);
    MPI_Scan(&fact, &sum_fact, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    printf("time taken by %d rank is %f\n", rank, t2 - t1);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == size - 1) {
        printf("fact sum : %d\n", sum_fact);
    }
    MPI_Finalize();
}