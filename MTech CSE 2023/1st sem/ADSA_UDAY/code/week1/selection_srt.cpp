#include <iostream>

using namespace std;

int main()
{

    int arr[20];

    for (int i = 0; i < 20; i++)
    {
        arr[i] = rand() % 20;
    }

    for (int i = 0; i < 20; i++)
    {
        cout << arr[i] << " ";
    }

    cout << endl;

    for (int i = 0; i < 20; i++)
    {
        int min_index = i;
        for (int j = i; j < 20; j++)
        {
            if (arr[j] < arr[min_index])
            {
                min_index = j;
            }
        }

        int temp = arr[i];
        arr[i] = arr[min_index];
        arr[min_index] = temp;
    }

    for (int i = 0; i < 20; i++)
    {
        cout << arr[i] << " ";
    }
}