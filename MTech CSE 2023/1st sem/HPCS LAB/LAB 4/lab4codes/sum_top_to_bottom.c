#include<stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Status status;
    int arr[4][4], recvArr1[4], recvArr2[4];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0) {
        printf("Enter 4X4 Matrix values\n");
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                scanf("%d", &arr[i][j]);
            }
        }
        printf("\n");
    }
    
    MPI_Scatter(&arr[0][0], 4, MPI_INT, &recvArr1, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scan(&recvArr1, &recvArr2, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    for(int i=0; i<size; i++) {
        printf("%d ", recvArr2[i]);
    }
    printf("\n");
    
    MPI_Finalize();
    return 0;
}