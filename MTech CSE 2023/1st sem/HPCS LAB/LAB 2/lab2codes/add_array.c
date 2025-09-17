#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int size, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size != 2)
    {
        fprintf(stderr, "This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("Enter the number of elements: ");
        int N;
        scanf("%d", &N);
        int arr[N];

        for (int i = 0; i < N; i++)
            scanf("%d", &arr[i]);

        // Send the size of the array to process 1
        MPI_Send(&N, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Send the second half of the array to process 1
        MPI_Send(&arr[N / 2], N / 2, MPI_INT, 1, 1, MPI_COMM_WORLD);

        int sum1 = 0, sum2 = 0;

        // Calculate the sum of the first half of the array
        for (int i = 0; i < N / 2; i++)
        {
            sum1 += arr[i];
        }

        printf("sum 1: %d\n", sum1);

        // Receive the sum of the second half from process 1
        MPI_Recv(&sum2, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);

        int final_sum = sum1 + sum2;
        printf("final sum: %d\n", final_sum);
    }
    else if (rank == 1)
    {
        int N;
        // Receive the size of the array from process 0
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int brr[N / 2];

        // Receive the second half of the array from process 0
        MPI_Recv(&brr[0], N / 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        int sum2 = 0;
        // Calculate the sum of the received array elements
        for (int i = 0; i < N / 2; ++i)
        {
            sum2 += brr[i];
        }

        printf("sum 2: %d\n", sum2);

        // Send the sum of the second half to process 0
        MPI_Send(&sum2, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}