#include <stdio.h>

#define K 8  // number of bits

int A[K]; // binary counter, A[0] is LSB

int incrementCounter() {
    int i = 0;
    int cost = 0; // number of bit flips in this operation

    // Flip bits until we find a 0
    while (i < K && A[i] == 1) {
        A[i] = 0;
        i++;
        cost++;
    }

    // If still within range, set the bit to 1
    if (i < K) {
        A[i] = 1;
        cost++;
    }

    return cost;
}

void printCounter() {
    for (int i = K - 1; i >= 0; i--) {
        printf("%d", A[i]);
    }
}

int main() {
    int n = 16; // number of increments
    int totalCost = 0;

    // Initialize counter to 0
    for (int i = 0; i < K; i++) {
        A[i] = 0;
    }

    printf("Initial value: ");
    printCounter();
    printf("\n");

    for (int step = 1; step <= n; step++) {
        int cost = incrementCounter();
        totalCost += cost;

        printf("After %2d increments: ", step);
        printCounter();
        printf(" | Cost this step: %d | Total cost: %d\n", cost, totalCost);
    }

    printf("\nAmortized cost = %.2f\n", (double)totalCost / n);

    return 0;
}
