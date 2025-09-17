#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

sem_t mutex, wrt;
int readcount = 0;
int data = 0;

void *reader(void *arg)
{
    int n=2;
    while(n)
    {
        sem_wait(&mutex);
        readcount++;
        if (readcount == 1)
        {
            sem_wait(&wrt);
        }
        sem_post(&mutex);

        printf("Reader %d reads: %d\n", *((int *)arg), data);

        sem_wait(&mutex);
        readcount--;
        if (readcount == 0)
        {
            sem_post(&wrt);
        }
        sem_post(&mutex);
        usleep(1100000);
        n--;
    }
}

void *writer(void *arg)
{
    int n=2;
    while(n)
    {
        sem_wait(&wrt);
        data++;
        printf("Writer %d writes: %d\n", *((int *)arg), data);
        sem_post(&wrt);
        usleep(1000000);
        n--;
    }
}

int main()
{
    int i;
    pthread_t read_thread[5], write_thread[5];
    sem_init(&mutex, 0, 1);
    sem_init(&wrt, 0, 1);

    int rval[5], wval[5];
    for (i = 0; i < 5; i++)
    {
        rval[i] = i + 1;
        wval[i] = i + 1;
        pthread_create(&read_thread[i], NULL, reader, &rval[i]);
        pthread_create(&write_thread[i], NULL, writer, &wval[i]);
    }

    for (i = 0; i < 5; i++)
    {
        pthread_join(read_thread[i], NULL);
        pthread_join(write_thread[i], NULL);
    }

    sem_destroy(&mutex);
    sem_destroy(&wrt);

    return 0;
}
