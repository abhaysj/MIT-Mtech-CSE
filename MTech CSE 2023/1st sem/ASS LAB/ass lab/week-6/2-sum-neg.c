#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int a[10] = {1, -1, 1, -1, -1, 1, -1, -1, 1, -1};
int status;

void *cal(void *arg)
{
	int sum;
	for (int i = 0; i < 10; i++)
	{
		if (a[i] < 0)
		{
			sum += a[i];
		}
	}
	return (void *)sum;
}

int main()
{
	int sum;
	pthread_t fd;
	pthread_create(&fd, 0, cal, 0);
	pthread_join(fd, (void **)&sum);
	wait(&status);

	printf("sum : %d\n", sum);
}
