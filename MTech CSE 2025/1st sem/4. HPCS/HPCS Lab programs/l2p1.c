#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int np, rank, ierr;
    double start_time, end_time;

    // Initialize MPI with error handler
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, ierr);
        return -1;
    }

    // Attach error handler so MPI calls return error codes instead of aborting
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ierr != MPI_SUCCESS) {
        printf("Error getting rank!\n");
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &np);
    if (ierr != MPI_SUCCESS) {
        printf("Error getting size!\n");
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }

    int n1, n2;
    MPI_Status status;

    if (rank == 0) {
        printf("Enter the two numbers: \n");
        scanf("%d %d", &n1, &n2);

        start_time = MPI_Wtime();  // start timing

        for (int i = 1; i < np; i++) {
            ierr = MPI_Send(&n1, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) printf("Error in MPI_Send by rank 0!\n");

            ierr = MPI_Send(&n2, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) printf("Error in MPI_Send by rank 0!\n");
        }
    } 
    else {
        ierr = MPI_Recv(&n1, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) printf("Error in MPI_Recv by rank %d!\n", rank);

        ierr = MPI_Recv(&n2, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) printf("Error in MPI_Recv by rank %d!\n", rank);

        if (rank == 1) {
            int n3 = n1 + n2;
            printf("Addition by process %d = %d\n", rank, n3);
        } 
        else if (rank == 2) {
            int n3 = n1 - n2;
            printf("Subtraction by process %d = %d\n", rank, n3);
        } 
        else if (rank == 3) {
            int n3 = n1 * n2;
            printf("Multiplication by process %d = %d\n", rank, n3);
        } 
        else {
            if (n2 != 0) {
                int n3 = n1 / n2;
                printf("Division by process %d = %d\n", rank, n3);
            } else {
                printf("Division by zero attempted by process %d!\n", rank);
            }
        }
    }

    // Only rank 0 measures time
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Total time taken = %f seconds\n", end_time - start_time);
    }

    ierr = MPI_Finalize();
    return 0;
}
