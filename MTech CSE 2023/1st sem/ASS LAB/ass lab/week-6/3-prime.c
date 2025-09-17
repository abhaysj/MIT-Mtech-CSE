#include <pthread.h>
#include <stdio.h>

void *child_thread(void *param)
{
    // initially, flag is set to true or 1
    int flag = 1;

    int N = (int)param;

    // loop to iterate through 2 to N/2
    for (int i = 2; i <= N / 2; i++)
    {

        // if N is perfectly divisible by i
        // flag is set to 0 i.e false
        if (N % i == 0)
        {
            flag = 0;
            break;
        }
    }

    if (flag)
    {
        printf("The number %d is a Prime Number\n", N);
    }
}

int main()
{
    int start, end;
    printf("enter start and end number : \n");
    scanf("%d %d", &start, &end);
    pthread_t thread[end - start];
    int return_value[10];
    for (int i = start; i < end; i++)
    {
        pthread_create(&thread[i], 0, &child_thread, (void *)i);
    }

    for (int i = start; i < end; i++)
    {
        pthread_join(thread[i], NULL);
    }
}
