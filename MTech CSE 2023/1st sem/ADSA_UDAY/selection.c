
#include <stdio.h>
#include <stdlib.h>

int main()
{

    int arr[20];

    for (int i = 0; i < 20; i++)
    {
        arr[i] = rand() % 40;
    }

    for (int i = 0; i < 20; i++)
    {
        printf("%d ", arr[i]);
    }

    printf("\n");

    for (int i = 0; i < 20 - 1; i++)
    {
        // Find the minimum element in unsorted array
        int min_idx = i;
        for (int j = i + 1; j < 20; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;

        // Swap the found minimum element with the first
        // element
        int temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }

    for (int i = 0; i < 20; i++)
    {
        printf("%d ", arr[i]);
    }

    printf("\n");
}