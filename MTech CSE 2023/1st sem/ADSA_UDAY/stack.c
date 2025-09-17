
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

struct StackNode
{
    int data;
    struct StackNode *next;
};

struct StackNode *newNode(int data)
{
    struct StackNode *new = (struct StackNode *)malloc(sizeof(struct StackNode));
    new->data = data;
    new->next = NULL;
    return new;
}

int isEmpty(struct StackNode **root)
{
    return !root;
}

void push(struct StackNode **root, int data)
{
    struct StackNode *node = newNode(data);
    node->next = *root;
    *root = node;
    printf("%d pushed to stack\n", node->data);
}

void pop(struct StackNode **root)
{
    if (isEmpty(root))
    {
        printf("Stack is empty\n");
        return;
    }

    struct StackNode *temp = *root;
    *root = (*root)->next;

    int data = temp->data;
    printf("%d is popped\n", data);
    free(temp);
}

void peek(struct StackNode **root)
{
    if (isEmpty(root))
    {
        printf("Stack is empty\n");
        return;
    }
    printf("top is %d\n", (*root)->data);
}

int main()
{
    struct StackNode *root = NULL;
    push(&root, 10);
    push(&root, 20);
    push(&root, 30);
    push(&root, 40);
    push(&root, 50);

    pop(&root);
    peek(&root);
    pop(&root);
    peek(&root);
    pop(&root);
}