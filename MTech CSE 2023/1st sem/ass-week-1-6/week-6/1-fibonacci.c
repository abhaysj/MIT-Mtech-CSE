#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int size, first = 0, second = 1, third, n, a[1000];
int status;

void *cal(void *arg)
{
    for (int i = 2; i <= n; i++)
    {
        first = a[i - 2];
        second = a[i - 1];
        a[i] = first + second;
    }
}

int main()
{
    printf("enter number of elements\n");
    scanf("%d", &n);

    a[0] = 0;
    a[1] = 1;

    pthread_t fd;
    pthread_create(&fd, 0, cal, 0);
    pthread_join(fd, 0);
    wait(&status);

    for (int i = 0; i <= n; i++)
    {
        printf("%d ", a[i]);
    }
}
