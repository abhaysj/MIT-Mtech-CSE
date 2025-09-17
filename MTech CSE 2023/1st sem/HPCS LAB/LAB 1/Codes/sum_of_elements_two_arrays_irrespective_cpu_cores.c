#include <stdio.h> // Include the standard input-output library
#include <omp.h>   // Include the OpenMP library for parallel programming

int main() {

    int num_threads = 6; // Set the desired number of threads

    int a[4] = {10, 20, 30, 40}; // Change values in array 'a'
    int b[4] = {5, 15, 25, 35};  // Change values in array 'b'
    int c[4];                   // Declare an array 'c' to store the sum of 'a' and 'b'

    int tid; // Declare a variable to store the thread ID

    #pragma omp parallel num_threads(num_threads)
    {
        tid = omp_get_thread_num(); // Get the thread ID of the current thread
        c[tid] = a[tid] + b[tid];   // Calculate the sum of corresponding elements of 'a' and 'b' and store in 'c'
        printf("c[%d] = %d \n", tid, c[tid]); // Print the result for each thread
    }
}





