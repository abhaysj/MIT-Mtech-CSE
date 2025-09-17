#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void pigeonhole_sort(int local_arr[], int local_n, int min_val, int range_size, int* sorted_arr) {
    int local_pigeonholes[range_size];
    for (int i = 0; i < range_size; i++) {
        local_pigeonholes[i] = 0;
    }

    for (int i = 0; i < local_n; i++) {
        local_pigeonholes[local_arr[i] - min_val]++;
    }

    int index = 0;
    for (int i = 0; i < range_size; i++) {
        while (local_pigeonholes[i] > 0) {
            sorted_arr[index++] = i + min_val;
            local_pigeonholes[i]--;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n;
    if (world_rank == 0) {
        printf("Enter the number of elements: ");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / world_size;
    int local_arr[local_n];

    int* sendbuf = NULL;
    if (world_rank == 0) {
        sendbuf = (int*)malloc(n * sizeof(int));
        printf("Enter %d elements: ", n);
        for (int i = 0; i < n; i++) {
            scanf("%d", &sendbuf[i]);
        }
    }

    MPI_Scatter(sendbuf, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        free(sendbuf);
    }

    int min_val = local_arr[0];
    int max_val = local_arr[0];
    for (int i = 1; i < local_n; i++) {
        if (local_arr[i] < min_val) {
            min_val = local_arr[i];
        }
        if (local_arr[i] > max_val) {
            max_val = local_arr[i];
        }
    }

    int global_min, global_max;
    MPI_Allreduce(&min_val, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_val, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    int range_size = global_max - global_min + 1;
    int sorted_arr[n];
    pigeonhole_sort(local_arr, local_n, global_min, range_size, sorted_arr);

    int* gathered_sorted_arr = NULL;
    if (world_rank == 0) {
        gathered_sorted_arr = (int*)malloc(n * sizeof(int));
    }

    MPI_Gather(sorted_arr, local_n, MPI_INT, gathered_sorted_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Sorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", gathered_sorted_arr[i]);
        }
        printf("\n");

        free(gathered_sorted_arr);
    }

    MPI_Finalize();

    return 0;
}




