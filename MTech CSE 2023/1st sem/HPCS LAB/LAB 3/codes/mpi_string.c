#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int n;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char s1[100], s2[100], b[200];

    if (rank == 0) {
        printf("Enter the first string: ");
        scanf("%s", s1);
        printf("Enter the second string: ");
        scanf("%s", s2);
        n = strlen(s1);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(s1, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(s2, n, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0, j = 0; i < n; i++, j += 2) {
        b[j] = s1[i];
        b[j + 1] = s2[i];
    }
    b[2 * n] = '\0';

    if (rank == 0) {
        printf("Resultant String: %s\n", b);
    }

    MPI_Finalize();
    return 0;
}
