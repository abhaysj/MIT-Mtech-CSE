#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 100

typedef struct {
    int arr[MAX];
    int topIndex;
    long long totalCost;
} Stack;

void initStack(Stack *s) {
    s->topIndex = -1;
    s->totalCost = 0;
}

int StackEmpty(Stack *s) {
    return (s->topIndex == -1);
}

int StackFull(Stack *s) {
    return (s->topIndex == MAX - 1);
}

void Push(Stack *s, int x) {
    if (StackFull(s)) return; // avoid overflow
    s->arr[++s->topIndex] = x;
    s->totalCost++;
}

int Pop(Stack *s) {
    if (StackEmpty(s)) return -1; // avoid underflow
    s->totalCost++;
    return s->arr[s->topIndex--];
}

void MultiPop(Stack *s, int k) {
    while (!StackEmpty(s) && k-- > 0) {
        Pop(s);
    }
}

int main() {
    Stack s;
    initStack(&s);
    srand(time(NULL));

    int n = 20; // number of random operations
    for (int i = 0; i < n; i++) {
        int op;

        // ensure no invalid operations
        if (StackEmpty(&s)) {
            op = 1; // force Push
        } else if (StackFull(&s)) {
            op = 2; // force Pop
        } else {
            op = rand() % 3 + 1; // 1 = Push, 2 = Pop, 3 = MultiPop
        }

        if (op == 1) {
            int val = rand() % 100;
            Push(&s, val);
            printf("Push %d\n", val);
        } else if (op == 2) {
            int popped = Pop(&s);
            printf("Pop %d\n", popped);
        } else {
            int k = (rand() % (s.topIndex + 1)) + 1; // up to current size
            printf("MultiPop %d\n", k);
            MultiPop(&s, k);
        }
    }

    printf("Total cost: %lld\n", s.totalCost);
    return 0;
}
