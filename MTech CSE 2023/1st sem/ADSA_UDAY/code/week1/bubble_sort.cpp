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

    int n=20;

    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {

            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = temp;
            }
        }
    }

    for (int i = 0; i < 20; i++)
    {
        cout << arr[i] << " ";
    }

    cout << endl;
}