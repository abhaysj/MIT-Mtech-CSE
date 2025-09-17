#include <stdio.h>
#include <omp.h>

int main() {
    int N = 10, sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 1; i <= N; i++) {
        sum += i;
    }

    printf("Sum from 1 to %d = %d\n", N, sum);
    return 0;
}
