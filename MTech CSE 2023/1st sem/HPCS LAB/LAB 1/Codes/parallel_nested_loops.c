#include <stdio.h>
#include <omp.h>

int main() {
    // OpenMP parallel region
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads(); // Get the total number of threads
        int thread_id = omp_get_thread_num();     // Get the current thread ID

        for (int i = thread_id; i < 4; i += num_threads) {
            // Print the thread ID and outer loop index
            printf("Thread %d: i=%d\n", thread_id, i);

            for (int j = 0; j < 4; ++j) {
                // Print the thread ID, outer loop index, and inner loop index
                printf("Thread %d: i=%d, j=%d\n", thread_id, i, j);
            }
        }
    }

    return 0;
}




