#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int arr[3][3];
    int brr;
    int element, counter = 0;
    
    if(rank == 0) {
        printf("Enter the arr\n");
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                scanf("%d", &arr[i][j]);
            }
        }
        printf("Enter the element\n");
        scanf("%d", &element);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(arr, 9, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    for(int i=0; i<3; i++) {
        if(arr[rank][i] == element) {
            counter++;
        }
    }
    
    MPI_Reduce(&counter, &brr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        printf("Found no. of occurences : %d\n", brr);
    }
    
    MPI_Finalize();
    return 0;
}