
#include <stdio.h>

#define MAX_SIZE 1000

int fenwickTree[MAX_SIZE];

// Function to update the value at index i by adding delta
void update(int i, int delta, int n) {
    while (i <= n) {
        fenwickTree[i] += delta;
        i += (i & -i); // Move to the next node in the tree
    }
}

// Function to query the prefix sum up to index i
int query(int i) {
    int sum = 0;
    while (i > 0) {
        sum += fenwickTree[i];
        i -= (i & -i); // Move to the parent node in the tree
    }
    return sum;
}

// Function to find the range sum between indices l and r
int rangeSum(int l, int r) {
    if (l == 1) {
        return query(r);
    } else {
        return query(r) - query(l - 1);
    }
}

// Function to update the range of values between indices l and r
void rangeUpdate(int l, int r, int delta, int n) {
    update(l, delta, n);
    update(r + 1, -delta, n);
}

// Function to build the Fenwick Tree from an input array
void buildFenwickTree(int arr[], int n) {
    for (int i = 1; i <= n; i++) {
        update(i, arr[i - 1], n);
    }
}

int main() {
    int n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    printf("Enter the elements of the array:\n");
    int arr[MAX_SIZE];
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    // Building the Fenwick Tree
    buildFenwickTree(arr, n);

    // Example: Update value at index 2 to 6
    int indexToUpdate = 2;
    int newValue = 6;
    update(indexToUpdate, newValue - arr[indexToUpdate - 1], n);

    // Example: Query the prefix sum up to index 5
    int queryIndex = 5;
    int result = query(queryIndex);
    printf("Prefix sum up to index %d is: %d\n", queryIndex, result);

    // Example: Find the range sum between indices 2 and 4
    int l = 2, r = 4;
    int rangeSumResult = rangeSum(l, r);
    printf("Range sum between indices %d and %d is: %d\n", l, r, rangeSumResult);

    // Example: Update the range of values between indices 1 and 3 by adding 2
    rangeUpdate(1, 3, 2, n);

    // Display the updated array
    printf("Updated array after range update:\n");
    for (int i = 1; i <= n; i++) {
        printf("%d ", query(i));
    }

    return 0;
}


