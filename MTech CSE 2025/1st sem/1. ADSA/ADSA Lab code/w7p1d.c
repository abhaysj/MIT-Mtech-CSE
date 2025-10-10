#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>

typedef struct FHNode {
    int key;
    int degree;
    bool mark;
    struct FHNode *parent;
    struct FHNode *child;
    struct FHNode *left;
    struct FHNode *right;
} FHNode;

typedef struct FibonacciHeap {
    FHNode *min;
    int n;
} FibonacciHeap;

/* Utility: create nodes and heaps with error checking */
FibonacciHeap *MAKE_FIB_HEAP() {
    FibonacciHeap *H = (FibonacciHeap *)malloc(sizeof(FibonacciHeap));
    if (!H) { fprintf(stderr, "Out of memory\n"); exit(1); }
    H->min = NULL;
    H->n = 0;
    return H;
}

FHNode *MAKE_NODE(int key) {
    FHNode *x = (FHNode *)malloc(sizeof(FHNode));
    if (!x) { fprintf(stderr, "Out of memory\n"); exit(1); }
    x->key = key;
    x->degree = 0;
    x->mark = false;
    x->parent = NULL;
    x->child = NULL;
    x->left = x;
    x->right = x;
    return x;
}

/* Circular doubly-linked list helpers */
static FHNode *insert_into_list(FHNode *head, FHNode *x) {
    if (!head) {
        x->left = x->right = x;
        return x;
    }
    x->right = head->right;
    x->left = head;
    head->right->left = x;
    head->right = x;
    return head;
}

static FHNode *remove_from_list(FHNode *head, FHNode *x) {
    if (x->right == x) return NULL;
    x->left->right = x->right;
    x->right->left = x->left;
    return (head == x) ? x->right : head;
}

/* Concatenate two circular lists */
static FHNode *concatenate_lists(FHNode *a, FHNode *b) {
    if (!a) return b;
    if (!b) return a;
    FHNode *a_right = a->right;
    FHNode *b_left = b->left;
    a->right = b;
    b->left = a;
    a_right->left = b_left;
    b_left->right = a_right;
    return a;
}

/* Memory free helpers (recursive) */
void free_node_recursive(FHNode *x) {
    if (!x) return;
    if (x->child) {
        FHNode *start = x->child, *cur = start;
        do {
            FHNode *next = cur->right;
            free_node_recursive(cur);
            cur = next;
        } while (cur != start);
    }
    free(x);
}

void free_heap(FibonacciHeap *H) {
    if (!H) return;
    FHNode *r = H->min;
    if (r) {
        FHNode *start = r, *cur = start;
        do {
            FHNode *next = cur->right;
            free_node_recursive(cur);
            cur = next;
        } while (cur != start);
    }
    free(H);
}

/* Core operations */
void FIB_HEAP_INSERT(FibonacciHeap *H, FHNode *x) {
    x->degree = 0;
    x->parent = NULL;
    x->child = NULL;
    x->mark = false;
    if (!H->min)
        H->min = insert_into_list(NULL, x);
    else {
        H->min = insert_into_list(H->min, x);
        if (x->key < H->min->key)
            H->min = x;
    }
    H->n += 1;
}

FibonacciHeap *FIB_HEAP_UNION(FibonacciHeap *H1, FibonacciHeap *H2) {
    FibonacciHeap *H = MAKE_FIB_HEAP();
    H->min = concatenate_lists(H1->min, H2->min);
    if (!H1->min || (H2->min && H2->min->key < H1->min->key))
        H->min = H2->min;
    else
        H->min = H1->min;
    H->n = H1->n + H2->n;
    free(H1); free(H2);
    return H;
}

static void FIB_HEAP_LINK(FibonacciHeap *H, FHNode *y, FHNode *x) {
    H->min = remove_from_list(H->min, y);
    y->left = y->right = y;
    y->parent = x;
    x->child = insert_into_list(x->child, y);
    x->degree += 1;
    y->mark = false;
}

static int upper_bound_degree(int n) {
    if (n <= 0) return 0;
    return (int)(log(n) / log(2)) + 2;
}

/* Improved CUT: Keep assignment and only set child head once */
static void CUT(FibonacciHeap *H, FHNode *x, FHNode *y) {
    if (y->child)
        y->child = remove_from_list(y->child, x);
    y->degree -= 1;
    x->parent = NULL;
    x->left = x->right = x;
    x->mark = false;
    H->min = insert_into_list(H->min, x);
}

/* Improved EXTRACT_MIN: free children if heap is destroyed */
FHNode *FIB_HEAP_EXTRACT_MIN(FibonacciHeap *H) {
    FHNode *z = H->min;
    if (z) {
        if (z->child) {
            FHNode *childHead = z->child, *cur = childHead;
            do {
                cur->parent = NULL;
                cur = cur->right;
            } while (cur != childHead);
            H->min = concatenate_lists(H->min, z->child);
            z->child = NULL;
        }
        H->min = remove_from_list(H->min, z);
        if (!H->min) {
            H->min = NULL;
        } else {
            H->min = z->right;
            int D = upper_bound_degree(H->n);
            FHNode **A = (FHNode **)calloc(D + 1, sizeof(FHNode *));
            if (!A) { fprintf(stderr, "Out of memory\n"); exit(1); }
            FHNode *w = H->min, *start = w;
            if (w) {
                do {
                    FHNode *x = w;
                    w = w->right;
                    int d = x->degree;
                    while (d <= D && A[d]) {
                        FHNode *y = A[d];
                        if (x->key > y->key) { FHNode *tmp = x; x = y; y = tmp; }
                        FIB_HEAP_LINK(H, y, x);
                        A[d] = NULL;
                        d++;
                    }
                    if (d > D) {
                        int newD = d + 2;
                        A = (FHNode **)realloc(A, sizeof(FHNode *) * (newD + 1));
                        if (!A) { fprintf(stderr, "Out of memory\n"); exit(1); }
                        for (int i = D+1; i <= newD; ++i) A[i] = NULL;
                        D = newD;
                    }
                    A[d] = x;
                } while (w != start);
            }
            H->min = NULL;
            for (int i = 0; i <= D; ++i)
                if (A[i]) {
                    A[i]->left = A[i]->right = A[i];
                    H->min = insert_into_list(H->min, A[i]);
                    if (!H->min || A[i]->key < H->min->key) H->min = A[i];
                }
            free(A);
        }
        H->n -= 1;
    }
    return z;
}

static void CASCADING_CUT(FibonacciHeap *H, FHNode *y) {
    FHNode *z = y->parent;
    if (z) {
        if (!y->mark) {
            y->mark = true;
        } else {
            CUT(H, y, z);
            CASCADING_CUT(H, z);
        }
    }
}

void FIB_HEAP_DECREASE_KEY(FibonacciHeap *H, FHNode *x, int k) {
    if (k > x->key) {
        fprintf(stderr, "new key is greater than current key\n");
        return;
    }
    x->key = k;
    FHNode *y = x->parent;
    if (y && x->key < y->key) {
        CUT(H, x, y);
        CASCADING_CUT(H, y);
    }
    if (x->key < H->min->key) H->min = x;
}

void FIB_HEAP_DELETE(FibonacciHeap *H, FHNode *x) {
    FIB_HEAP_DECREASE_KEY(H, x, INT_MIN);
    FHNode *removed = FIB_HEAP_EXTRACT_MIN(H);
    if (removed) free(removed);
}

/* Debug helpers */
void print_root_list(FibonacciHeap *H) {
    printf("Heap nodes (root list):\n");
    FHNode *r = H->min;
    if (!r) { printf("  [empty]\n"); return; }
    FHNode *start = r;
    do {
        printf(" key=%d deg=%d mark=%d |", r->key, r->degree, r->mark);
        r = r->right;
    } while (r != start);
    printf("\n");
}

/* Example usage */
int main() {
    FibonacciHeap *H = MAKE_FIB_HEAP();
    FHNode *a = MAKE_NODE(10), *b = MAKE_NODE(3), *c = MAKE_NODE(15), *d = MAKE_NODE(20), *e = MAKE_NODE(25);

    FIB_HEAP_INSERT(H, a);
    FIB_HEAP_INSERT(H, b);
    FIB_HEAP_INSERT(H, c);
    FIB_HEAP_INSERT(H, d);
    FIB_HEAP_INSERT(H, e);

    printf("Initial root list:\n");
    print_root_list(H);

    // Extract min
    FHNode *m = FIB_HEAP_EXTRACT_MIN(H);
    printf("Extracted min: %d\n", m->key);
    free(m);
    print_root_list(H);

    // Decrease key example
    printf("\nDecreasing key of node with value 25 to 2...\n");
    FIB_HEAP_DECREASE_KEY(H, e, 2);
    print_root_list(H);

    // Delete node example
    printf("\nDeleting node with value 20...\n");
    FIB_HEAP_DELETE(H, d);
    print_root_list(H);

    free_heap(H);

    return 0;
}
