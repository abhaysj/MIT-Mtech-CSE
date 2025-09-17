#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int size, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        char message[8] = "manipal";
        char rec_message[8];
        int len = 8;

        printf("Sending message: %s from rank %d\n", message, rank);

        // Send length of message to rank 1
        MPI_Ssend(&len, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Send the message itself to rank 1
        MPI_Ssend(&message, len, MPI_CHAR, 1, 1, MPI_COMM_WORLD);

        // Receive manipulated message from rank 1
        MPI_Recv(&rec_message, len, MPI_CHAR, 1, 2, MPI_COMM_WORLD, &status);
        printf("Received: %s in rank %d\n", rec_message, rank);
    }
    else if (rank == 1)
    {
        int len;
        char message[8];

        // Receive length of message from rank 0
        MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Receive the message from rank 0
        MPI_Recv(&message, len, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);
        printf("Received: %s in rank %d\n", message, rank);

        // Manipulate the message by toggling case
        for (int i = 0; i < len; i++)
        {
            if (message[i] >= 97) // ASCII value of 'a'
            {
                message[i] -= 32; // Convert to uppercase
            }
            else
            {
                message[i] += 32; // Convert to lowercase
            }
        }

        // Send the manipulated message back to rank 0
        MPI_Ssend(&message, len, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
        printf("Sending message: %s from rank %d\n", message, rank);
    }

    MPI_Finalize();
    return 0;
}