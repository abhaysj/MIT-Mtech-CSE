#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int myrank, size, m;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0)
    {
        printf("Enter the value of M: ");
        scanf("%d", &m);
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int a[m];
    MPI_Scatter(a, m, MPI_INT, a, m, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < m; i++)
    {
        a[i] = pow(a[i], myrank + 2);
    }

    MPI_Gather(a, m, MPI_INT, a, m, MPI_INT, 0, MPI_COMM_WORLD);

    if (myrank == 0)
    {
        printf("The gathered elements are:");
        for (int i = 0; i < size * m; i++)
            printf(" %d", a[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
