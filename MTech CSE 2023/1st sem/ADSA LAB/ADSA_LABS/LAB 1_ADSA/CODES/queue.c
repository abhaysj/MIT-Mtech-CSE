#include <stdio.h>
#include <stdbool.h>
#define MAX_SIZE 100

struct Queue {
    int data[MAX_SIZE];
    int front, rear;
};

void initialize(struct Queue *q) {
    q->front = q->rear = -1;
}

bool isEmpty(struct Queue *q) {
    return (q->front == -1);
}

bool isFull(struct Queue *q) {
    return ((q->rear + 1) % MAX_SIZE == q->front);
}

void enqueue(struct Queue *q, int x, int *countin) {
    if (isFull(q)) {
        printf("Queue is full, cannot enqueue %d\n", x);
        return;
    }
    if (isEmpty(q)) {
        q->front = q->rear = 0;
    } else {
        q->rear = (q->rear + 1) % MAX_SIZE;
    }
    q->data[q->rear] = x;
    (*countin)++;
    printf("Enqueued element %d\n", x);
}

int dequeue(struct Queue *q, int *countout) {
    if (isEmpty(q)) {
        printf("Queue is empty, cannot dequeue\n");
        return -1;
    }
    int x = q->data[q->front];
    if (q->front == q->rear) {
        initialize(q);
    } else {
        q->front = (q->front + 1) % MAX_SIZE;
    }
    (*countout)++;
    printf("Dequeued element: %d\n", x);
    return x;
}

int main() {
    struct Queue q;
    int countin = 0, countout = 0;
    initialize(&q);

    int choice, element;
    while (1) {
        printf("Choose an operation:\n");
        printf("1. Enqueue\n");
        printf("2. Dequeue\n");
        printf("3. Quit\n");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("Enter element to enqueue: ");
                scanf("%d", &element);
                enqueue(&q, element, &countin);
                printf("Number of insertions for enqueues: %d\n", countin);
                break;
            case 2:
                dequeue(&q, &countout);
                printf("Number of deletions for dequeues: %d\n", countout);
                break;
            case 3:
                printf("Quitting the program.\n");
                return 0;
            default:
                printf("Invalid choice. Try again.\n");
        }
    }

    return 0;
}
