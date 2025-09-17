#include <stdio.h>
#include <stdlib.h>
int compare(const void *num1, const void *num2) {
    int a = *(int *)num1;
    int b = *(int *)num2;
    return a - b;  // Simple comparison for ascending order
}

int main() {
    int n = 20;
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 20;
        printf("%d ", arr[i]);
    }

    printf("\n");

    qsort(arr, n, sizeof(int), compare);

    // Print the sorted array
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    printf("\n");

    // Get user input for the element to search
    int element;
    printf("Enter the element to search: ");
    scanf("%d", &element);

    // Binary search
    int start = 0, end = n - 1;  // Adjusted end index
    while (start <= end) {  // Use <= instead of <
        int mid = (start + end) / 2;
        if (arr[mid] == element) {
            printf("Element found at index: %d\n", mid);
            break;  // Exit the loop if the element is found
        } else if (arr[mid] < element) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }

    return 0;
}
