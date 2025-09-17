#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char word[100];
    int n;
    char local_result[100];
    char *final_result = NULL;

    if (rank == 0) {
        printf("Enter a word: ");
        fflush(stdout);
        scanf("%s", word);
        n = strlen(word);

        if (n != size) {
            printf("Error: Run with %d processes!\n", n);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the length and word to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(word, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Each process creates its local repeated string
    int repeat = rank + 1;
    for (int i = 0; i < repeat; i++) {
        local_result[i] = word[rank];
    }
    local_result[repeat] = '\0';

    // Gather results in root process
    if (rank == 0) {
        final_result = (char *)malloc(1000 * sizeof(char));
    }
    //char recvbuf[100];
    char *recvbuf = NULL;
    if (rank == 0) {
        recvbuf = (char *)malloc(size * 100 * sizeof(char));
    }

    MPI_Gather(local_result, 100, MPI_CHAR,
            recvbuf, 100, MPI_CHAR,
            0, MPI_COMM_WORLD);

    //MPI_Gather(local_result, 100, MPI_CHAR, recvbuf, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process concatenates
    if (rank == 0) {
        final_result[0] = '\0';
        for (int i = 0; i < n; i++) {
            strcat(final_result, recvbuf + i*100);
        }
        printf("Output: %s\n", final_result);
        fflush(stdout);
        free(final_result);
    }

    MPI_Finalize();
    return 0;
}
