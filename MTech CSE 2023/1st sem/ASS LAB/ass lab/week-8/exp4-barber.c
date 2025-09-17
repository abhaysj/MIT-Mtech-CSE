#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_CUSTOMERS 10
#define MAX_WAITING 5

sem_t barber, customer, accessSeats;
int waitingCustomers = 0;

void *barber_thread(void *arg)
{
    while (1)
    {
        sem_wait(&customer);
        sem_wait(&accessSeats);
        waitingCustomers--;
        sem_post(&barber);
        sem_post(&accessSeats);

        printf("Barber is cutting hair.\n");
        sleep(2); // Simulate haircut
        printf("Haircut finished.\n");
    }
}

void *customer_thread(void *arg)
{
    while (1)
    {
        sem_wait(&accessSeats);

        if (waitingCustomers < MAX_WAITING)
        {
            waitingCustomers++;
            sem_post(&customer);
            sem_post(&accessSeats);

            sem_wait(&barber);
            printf("Customer is getting a haircut.\n");
        }
        else
        {
            sem_post(&accessSeats);
            printf("Customer leaves as all seats are occupied.\n");
            sleep(1); // Simulate waiting for some time before returning
        }

        sleep(1); // Simulate time between customer arrivals
    }
}

int main()
{
    pthread_t barberThread, customerThreads[MAX_CUSTOMERS];

    sem_init(&barber, 0, 0);
    sem_init(&customer, 0, 0);
    sem_init(&accessSeats, 0, 1);

    pthread_create(&barberThread, NULL, barber_thread, NULL);

    for (int i = 0; i < MAX_CUSTOMERS; i++)
    {
        pthread_create(&customerThreads[i], NULL, customer_thread, NULL);
    }

    for (int i = 0; i < MAX_CUSTOMERS; i++)
    {
        pthread_join(customerThreads[i], NULL);
    }

    return 0;
}
