
#include <stdio.h>
#include <stdlib.h>

int compare(const void *num1, const void *num2)
{
    int a = *(int *)num1;
    int b = *(int *)num2;
    return a > b ? 1 : -1;
}

int main()
{
    int n = 20;
    int arr[n];

    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 20;
        printf("%d ", arr[i]);
    }

    printf("\n");

    qsort(arr, n, sizeof(int), compare);

    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }

    printf("\n");

    int element;
    scanf("%d", &element);

    int start = 0, end = n;

    while (start < end)
    {
        int mid = (start + end - 1) / 2;
        if (arr[mid] == element)
        {
            printf("element fount at : %d\n", mid);
        }

        if (arr[mid] < element)
        {
            start = mid + 1;
        }
        else
        {
            end = mid - 1;
        }
    }


}