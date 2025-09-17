#include <stdio.h>
int main() {
    int n, i, j, position, swap, swapc = 0, cmpc = 0;

    printf("Enter the number of elements:");
    scanf("%d", &n);

    int arr[n];

    printf("Enter the elements:\n");
    for (i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }
    for (i = 0; i < (n - 1); i++) {
        position = i;
        for (j = i + 1; j < n; j++) {
            cmpc++;
            if (arr[position] > arr[j]) {
                position = j;
            }
        }
        if (position != i) {
            swap = arr[i];
            arr[i] = arr[position];
            arr[position] = swap;
            swapc++;
        }
    }

    printf("Comparison count : %d\n", cmpc);
    printf("Swap count : %d\n", swapc);

    for (i = 0; i < n; i++) {
        printf("%d\t", arr[i]);
    }

    return 0;
}







