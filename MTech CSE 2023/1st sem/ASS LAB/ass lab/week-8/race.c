// compile with : gcc -g race.c -lpthread
// detect race  : valgrind -tool=helrind ./a.out

#include <pthread.h>
int counter = 0;

void *func(void *params)
{
    counter++;
}

int main()
{
    pthread_t thread1, thread2;
    pthread_create(&thread1, 0, func, 0);
    pthread_create(&thread2, 0, func, 0);
    pthread_join(thread1, 0);
    pthread_join(thread2, 0);
}
