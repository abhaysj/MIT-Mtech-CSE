#include <stdio.h>
#include <omp.h>

int main() {
    int N = 100; // Change N to the desired value
    int sum = 0;

    // Parallelize the for loop and perform reduction on the 'sum' variable
    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= N; i++) {
        sum += i; // Each thread adds its own partial sum to the 'sum' variable
    }

    // Print the final sum calculated by all threads
    printf("Sum of integers from 1 to %d is %d\n", N, sum);

    return 0;
}






