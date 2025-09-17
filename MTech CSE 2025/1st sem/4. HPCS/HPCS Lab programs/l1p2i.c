#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int N;
    printf("Enter size: ");
    scanf("%d", &N);

    int *a = malloc(N * sizeof(int));
    int *b = malloc(N * sizeof(int));
    int nthreads = omp_get_num_procs(); // number of CPU cores
    omp_set_num_threads(nthreads);
    // Fill arrays manually
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;         // example initialization
        b[i] = (i + 1) * 10;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        printf("Thread %d: a[%d] + b[%d] = %d\n",
               omp_get_thread_num(), i, i, a[i] + b[i]);
    }

    free(a);
    free(b);
    return 0;
}
