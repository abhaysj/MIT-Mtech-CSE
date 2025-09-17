#include <stdio.h>
#include <omp.h>

int main() {
    int i, n = 5;
    int shared_sum = 0;   // Shared variable
    int private_var;      // Will be private inside the loop

    // Parallel region with a for loop
    #pragma omp parallel for private(private_var) shared(shared_sum)
    for (i = 0; i < n; i++) {
        private_var = i * i;   // Each thread has its own private copy
        printf("Thread %d: i=%d, private_var=%d\n", omp_get_thread_num(), i, private_var);

        // Update shared variable - use atomic to avoid race condition
        #pragma omp atomic
        shared_sum += private_var;
    }

    printf("Final shared_sum = %d\n", shared_sum);

    return 0;
}
