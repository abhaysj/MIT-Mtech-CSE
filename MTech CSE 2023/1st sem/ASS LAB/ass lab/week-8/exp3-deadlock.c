#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

sem_t sem1, sem2;

void *thread1(void *arg)
{
    
    sem_wait(&sem1);
    printf("thread 1 locking sem 1\n");
    sleep(1);
    
    sem_wait(&sem2);
    printf("thread 1 locking sem 2\n");

    sem_post(&sem2);
    sem_post(&sem1);
}

void *thread2(void *arg)
{
    
    sem_wait(&sem2);
    printf("thread 2 locking sem 2\n");
    sleep(1);

    sem_wait(&sem1);
    printf("thread 2 locking sem 1\n");

    sem_post(&sem1);
    sem_post(&sem2);
}

int main()
{
    pthread_t t1, t2;

    sem_init(&sem1, 0, 1);
    sem_init(&sem2, 0, 1);

    pthread_create(&t1, NULL, thread1, NULL);
    pthread_create(&t2, NULL, thread2, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    sem_destroy(&sem1);
    sem_destroy(&sem2);

    return 0;
}
