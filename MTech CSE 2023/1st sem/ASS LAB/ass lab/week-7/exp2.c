#include <stdio.h>
#include <stdlib.h>

#define QUEUE_SIZE 100

// Process structure
typedef struct
{
    int id;
    int priority;
} Process;

typedef struct
{
    Process *processes[QUEUE_SIZE];
    int front;
    int rear;
} Queue;

void initializeQueue(Queue *q)
{
    q->front = -1;
    q->rear = -1;
}

void enqueue(Queue *q, Process *process)
{
    if (q->rear == QUEUE_SIZE - 1)
    {
        printf("Queue is full.\n");
        return;
    }
    if (q->front == -1)
    {
        q->front = 0;
    }
    q->rear++;
    q->processes[q->rear] = process;
}

Process *dequeue(Queue *q)
{
    if (q->front == -1)
    {
        printf("Queue is empty.\n");
        return NULL;
    }
    Process *process = q->processes[q->front];
    q->front++;
    if (q->front > q->rear)
    {
        q->front = q->rear = -1;
    }
    return process;
}

int main()
{
    // Initialize three queues
    Queue queue1, queue2, queue3;
    initializeQueue(&queue1);
    initializeQueue(&queue2);
    initializeQueue(&queue3);

    // Example: Allocate memory for a process
    Process *process1 = (Process *)malloc(sizeof(Process));
    process1->id = 1;
    process1->priority = 1;

    // Enqueue the process based on priority
    if (process1->priority == 1)
    {
        enqueue(&queue1, process1);
    }
    else if (process1->priority == 2)
    {
        enqueue(&queue2, process1);
    }
    else
    {
        enqueue(&queue3, process1);
    }

    // Example: Dequeue and process the next process based on priority
    Process *nextProcess = NULL;
    if (queue1.front != -1)
    {
        nextProcess = dequeue(&queue1);
    }
    else if (queue2.front != -1)
    {
        nextProcess = dequeue(&queue2);
    }
    else if (queue3.front != -1)
    {
        nextProcess = dequeue(&queue3);
    }

    // Process the next process
    if (nextProcess != NULL)
    {
        printf("Processing process with id %d and priority %d.\n", nextProcess->id, nextProcess->priority);
        free(nextProcess); // Release allocated memory for the process
    }
    else
    {
        printf("No process to process.\n");
    }

    return 0;
}
