#include <stdio.h>

// Define the size of the BIT
#define MAXN 1000

int BIT[MAXN] = {0}; // Binary Indexed Tree
int arr[MAXN];      // Original array

// Function to update the BIT
void update(int index, int val) {
    while (index <= MAXN) {
        BIT[index] += val;
        index += index & -index; // Move to the next position with a different low bit
    }
}

// Function to get the prefix sum up to index
int query(int index) {
    int sum = 0;
    while (index > 0) {
        sum += BIT[index];
        index -= index & -index; // Move to the parent with a different low bit
    }
    return sum;
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    printf("Enter the elements: ");
    for (int i = 1; i <= n; i++) {
        scanf("%d", &arr[i]);
        update(i, arr[i]); // Update the BIT
    }

    int q; // Number of queries
    printf("Enter the number of queries: ");
    scanf("%d", &q);

    double totalAmortizedCost = 0;

    for (int i = 1; i <= q; i++) {
        int type, index, value;
        printf("Query %d: ", i);
        scanf("%d", &type);

        if (type == 1) {
            // Update operation
            scanf("%d %d", &index, &value);
            totalAmortizedCost += 1.0; // Update cost is 1
            int diff = value - arr[index];
            arr[index] = value;
            update(index, diff); // Update the BIT
        } else if (type == 2) {
            // Query operation
            scanf("%d", &index);
            totalAmortizedCost += 1.0; // Query cost is 1
            int prefixSum = query(index);
            printf("Prefix Sum: %d\n", prefixSum);
        }
    }

    double amortizedCostPerOperation = totalAmortizedCost / q;
    printf("Amortized cost per operation: %lf\n", amortizedCostPerOperation);

    return 0;
}

