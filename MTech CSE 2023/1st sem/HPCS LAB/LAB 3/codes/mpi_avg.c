#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = atoi(argv[1]);
    int brr[M], arr[M];
    double avg = 0, total_avg = 0;

    if (rank == 0)
    {
        printf("Enter %d values:\n", M);
        for (int i = 0; i < M; i++)
            scanf("%d", &arr[i]);
    }

    MPI_Scatter(arr, M, MPI_INT, brr, M, MPI_INT, 0, MPI_COMM_WORLD);

    for (int j = 0; j < M; ++j)
        avg += brr[j];

    avg /= M;

    MPI_Gather(&avg, 1, MPI_DOUBLE, &total_avg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        total_avg /= M;
        printf("Total avg: %lf\n", total_avg);
    }

    MPI_Finalize();
    return 0;
}
