#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
// Define a Fibonacci Node structure
struct FibonacciNode {
    int key;
    struct FibonacciNode* next;
    struct FibonacciNode* child;
    struct FibonacciNode* parent;
    bool marked;
    int degree;
};
// Define a Fibonacci Heap structure
struct FibonacciHeap {
    struct FibonacciNode* min;
    int size;
};
// Create an empty Fibonacci Heap
struct FibonacciHeap* createFibonacciHeap() {
    struct FibonacciHeap* heap = (struct FibonacciHeap*)malloc(sizeof(struct FibonacciHeap));
    heap->min = NULL;
    heap->size = 0;
    return heap;
}
// Create a Fibonacci Node with a given key
struct FibonacciNode* createFibonacciNode(int key) {
    struct FibonacciNode* node = (struct FibonacciNode*)malloc(sizeof(struct FibonacciNode));
    node->key = key;
    node->next = node;
    node->child = NULL;
    node->parent = NULL;
    node->marked = false;
    node->degree = 0;
    return node;
}
// Insert a node into the Fibonacci Heap
void insert(struct FibonacciHeap* heap, int key) {
    struct FibonacciNode* node = createFibonacciNode(key);
    if (heap->min == NULL) {  printf("Minimum: %d\n", getMin(heap));
        heap->min = node;
    } else {
        node->next = heap->min->next;
        heap->min->next = node;
        if (node->key < heap->min->key) {
            heap->min = node;
        }
    }
    heap->size++;
}
// Get the minimum key in the Fibonacci Heap
int getMin(struct FibonacciHeap* heap) {
    if (heap->min == NULL) {
        return -1; // Heap is empty
    }
    return heap->min->key;
}
int main() {
    struct FibonacciHeap* heap = createFibonacciHeap();
    insert(heap, 3);
    insert(heap, 100);
    insert(heap, 5);
    insert(heap, 100);
    insert(heap, 20);
    insert(heap, 110);
    insert(heap, 5);
    insert(heap, 6);
    insert(heap, 90);

    printf("Minimum: %d\n", getMin(heap));

    return 0;
}

