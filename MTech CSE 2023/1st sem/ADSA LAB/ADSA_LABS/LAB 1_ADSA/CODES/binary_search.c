#include <stdio.h>

int binarySearch(int a[], int beg, int end, int val, int *counter) {
    int mid;
    while (beg <= end) {
        (*counter)++;
        mid = beg + (end - beg) / 2;
        if (a[mid] == val) {
            return mid + 1;
        } else if (a[mid] < val) {
            beg = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    return -1;
}

int main() {
    int val, nu;
    printf("Enter the number of elements: ");
    scanf("%d", &nu);
    int a[nu];
    printf("Enter the elements:\n");
    for (int i = 0; i < nu; i++) {
        scanf("%d", &a[i]);
    }
    printf("Enter the element to be searched: ");
    scanf("%d", &val);
    int counter = 0;
    int res = binarySearch(a, 0, nu - 1, val, &counter);
    if (res == -1)
        printf("\nElement is not present in the array");
    else
        printf("\nElement is present at position %d in the array", res);
    printf("\nNumber of comparisons: %d", counter);
    return 0;
}


