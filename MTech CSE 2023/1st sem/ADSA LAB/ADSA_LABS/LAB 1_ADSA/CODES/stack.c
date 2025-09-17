#include <stdio.h>
#include <stdbool.h>
#define MAX_SIZE 100

struct Stack {
    int data[MAX_SIZE];
    int top;
};

void initialize(struct Stack *stack) {
    stack->top = -1;
}

bool isEmpty(struct Stack *stack) {
    return (stack->top == -1);
}

bool isFull(struct Stack *stack) {
    return (stack->top == MAX_SIZE - 1);
}

void push(struct Stack *stack, int x, int *countpush) {
    if (isFull(stack)) {
        printf("Stack is full, cannot push %d\n", x);
        return;
    }
    stack->data[++stack->top] = x;
    (*countpush)++;
    printf("Element %d pushed into stack\n", x);
}

int pop(struct Stack *stack, int *countpop) {
    if (isEmpty(stack)) {
        printf("Stack is empty, cannot pop\n");
        return -1;
    }
    int popped = stack->data[stack->top--];
    (*countpop)++;
    printf("Popped element: %d\n", popped);
    return popped;
}

int main() {
    struct Stack stack;
    initialize(&stack);
    int countpush = 0, countpop = 0;

    int choice, element;

    do {
        printf("\nStack Operations:\n");
        printf("1. Check if Stack is Empty\n");
        printf("2. Push Element\n");
        printf("3. Pop Element\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("Stack is %s\n", isEmpty(&stack) ? "empty" : "not empty");
                break;
            case 2:
                printf("Enter an element to push: ");
                scanf("%d", &element);
                push(&stack, element, &countpush);
                break;
            case 3:
                pop(&stack, &countpop);
                break;
            case 4:
                printf("Exiting...\n");
                break;
            default:
                printf("Invalid choice\n");
        }
    } while (choice != 4);

    printf("Total number of pushes: %d\n", countpush);
    printf("Total number of pops: %d\n", countpop);
    return 0;
}






