#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int factorial(int n) {
    int res = 1;
    for (int i = 1; i <= n; i++)
        res *= i;
    return res;
}

int main(int argc, char* argv[]) {
    int rank, size, x, prefix_sum_factorial;
    double t1, t2, local_time, total_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    t1 = MPI_Wtime();

    // Each process computes its factorial(rank+1)
    x = factorial(rank + 1);

    // Prefix sum of factorials using MPI_Scan
    MPI_Scan(&x, &prefix_sum_factorial, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    t2 = MPI_Wtime();
    local_time = t2 - t1;

    // Print prefix sum result from each process
    printf("Process %d: Prefix sum of factorials up to %d! = %d\n", 
           rank, rank+1, prefix_sum_factorial);

    // Reduce max time to root (slowest process = system time)
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Each process prints its own time
    printf("Process %d: Time taken = %lf seconds\n", rank, local_time);

    if (rank == 0) {
        printf("Total system time (max over processes): %lf seconds\n", total_time);
    }

    MPI_Finalize();
    return 0;
}
