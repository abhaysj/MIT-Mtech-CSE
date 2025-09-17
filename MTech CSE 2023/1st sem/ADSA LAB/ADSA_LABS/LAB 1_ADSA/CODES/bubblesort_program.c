#include <stdio.h>

void BubbleSort(int arr[], int n, int *countercomp, int *counterswap) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            (*countercomp)++;
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                (*counterswap)++;
            }
        }
    }
}

int main() {
    int arr[100], n, countercomp = 0, counterswap = 0;
    printf("Enter size of array\n");
    scanf("%d", &n);
    printf("Enter array elements:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }
    printf("Original array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t", arr[i]);
    }
    BubbleSort(arr, n, &countercomp, &counterswap);
    printf("\nSorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t", arr[i]);
    }
    printf("\nNumber of comparisons: %d\n", countercomp);
    printf("Number of swaps: %d\n", counterswap);
    
    return 0;
}

