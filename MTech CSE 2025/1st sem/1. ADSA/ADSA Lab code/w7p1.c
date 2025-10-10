/* fibonacci_heap.c
   CLRS-style Fibonacci Heap (min-heap) implementation in plain C.
   Note: error checking / memory failure checks are minimal for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>

/* ---------- Node and heap structures ---------- */

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
    int n;          /* number of nodes in heap */
} FibonacciHeap;

/* ---------- Utility: create nodes and heaps ---------- */

FibonacciHeap *MAKE_FIB_HEAP() {
    FibonacciHeap *H = (FibonacciHeap *)malloc(sizeof(FibonacciHeap));
    H->min = NULL;
    H->n = 0;
    return H;
}

FHNode *MAKE_NODE(int key) {
    FHNode *x = (FHNode *)malloc(sizeof(FHNode));
    x->key = key;
    x->degree = 0;
    x->mark = false;
    x->parent = NULL;
    x->child = NULL;
    x->left = x;
    x->right = x;
    return x;
}

/* ---------- Circular doubly-linked list helpers ---------- */

/* insert node x into root list (or any circular list whose head pointer is *head)
   returns new head pointer (usually head if not NULL) */
static FHNode *insert_into_list(FHNode *head, FHNode *x) {
    if (head == NULL) {
        x->left = x->right = x;
        return x;
    } else {
        /* insert x just right of head */
        x->right = head->right;
        x->left = head;
        head->right->left = x;
        head->right = x;
        return head;
    }
}

/* remove node x from its circular list. If x is the only node, return NULL.
   If head == x and list has more nodes, return new head (x->right).
   Caller must ensure pointers of x remain valid or reset as needed. */
static FHNode *remove_from_list(FHNode *head, FHNode *x) {
    if (x->right == x) {
        /* single node */
        return NULL;
    } else {
        x->left->right = x->right;
        x->right->left = x->left;
        if (head == x) return x->right;
        else return head;
    }
}

/* concatenate two circular lists: a and b. Returns head of combined list (a if exists else b) */
static FHNode *concatenate_lists(FHNode *a, FHNode *b) {
    if (!a) return b;
    if (!b) return a;
    /* splice a.right.. and b.right.. */
    FHNode *a_right = a->right;
    FHNode *b_left = b->left;

    a->right = b;
    b->left = a;
    a_right->left = b_left;
    b_left->right = a_right;

    return a; /* keep a as representative */
}

/* ---------- Core operations ---------- */

void FIB_HEAP_INSERT(FibonacciHeap *H, FHNode *x) {
    x->degree = 0;
    x->parent = NULL;
    x->child = NULL;
    x->mark = false;
    x->left = x->right = x;
    H->min = insert_into_list(H->min, x);
    if (H->min == NULL || x->key < H->min->key) H->min = x;
    H->n += 1;
}

FibonacciHeap *FIB_HEAP_UNION(FibonacciHeap *H1, FibonacciHeap *H2) {
    FibonacciHeap *H = MAKE_FIB_HEAP();
    H->min = concatenate_lists(H1->min, H2->min);
    if (H1->min == NULL || (H2->min != NULL && H2->min->key < H1->min->key))
        H->min = H2->min;
    else
        H->min = H1->min;
    H->n = H1->n + H2->n;
    /* free H1 and H2 structures if desired (nodes remain) */
    free(H1);
    free(H2);
    return H;
}

static void FIB_HEAP_LINK(FibonacciHeap *H, FHNode *y, FHNode *x) {
    /* remove y from root list */
    H->min = remove_from_list(H->min, y);
    /* make y a child of x */
    y->left = y->right = y;
    y->parent = x;
    x->child = insert_into_list(x->child, y);
    x->degree += 1;
    y->mark = false;
}

/* Consolidate helper: compute an upper bound on degree -> use floor(log_phi(n)) ~ log2(n) + 1 safe */
static int upper_bound_degree(int n) {
    /* safe upper bound: log2(n) rounded up + 1 */
    if (n <= 0) return 0;
    return (int)(log(n) / log(2)) + 2;
}

// FHNode *FIB_HEAP_EXTRACT_MIN(FibonacciHeap *H) {
//     FHNode *z = H->min;
//     if (z != NULL) {
//         /* 1) for each child x of z, add x to root list */
//         if (z->child != NULL) {
//             FHNode *x = z->child;
//             /* iterate over children (careful since circular) */
//             FHNode *start = x;
//             do {
//                 FHNode *next = x->right;
//                 x->parent = NULL;
//                 /* move x to root list */
//                 z->child = remove_from_list(z->child, x); /* updates z->child if needed */
//                 H->min = insert_into_list(H->min, x);
//                 x = next;
//             } while (start != z->child && z->child != NULL && x != start);
//             /* after loop: all children moved; ensure z->child is NULL */
//             z->child = NULL;
//         }

//         /* 2) remove z from root list */
//         H->min = remove_from_list(H->min, z);

//         if (H->min == NULL) {
//             H->min = NULL;
//         } else {
//             H->min = z->right; /* arbitrary root */
//             /* Consolidate */
//             int D = upper_bound_degree(H->n);
//             FHNode **A = (FHNode **)calloc(D + 1, sizeof(FHNode *));
//             for (int i = 0; i <= D; ++i) A[i] = NULL;

//             /* build list of roots to process (copy because linking changes root list) */
//             FHNode *w = H->min;
//             FHNode *start = w;
//             if (w != NULL) {
//                 do {
//                     FHNode *x = w;
//                     w = w->right;
//                     int d = x->degree;
//                     while (d <= D && A[d] != NULL) {
//                         FHNode *y = A[d];
//                         if (x->key > y->key) {
//                             FHNode *tmp = x; x = y; y = tmp;
//                         }
//                         FIB_HEAP_LINK(H, y, x);
//                         A[d] = NULL;
//                         d = d + 1;
//                     }
//                     if (d > D) {
//                         /* reallocate A larger (rare) */
//                         int newD = d + 2;
//                         A = (FHNode **)realloc(A, sizeof(FHNode *) * (newD + 1));
//                         for (int i = D+1; i <= newD; ++i) A[i] = NULL;
//                         D = newD;
//                     }
//                     A[d] = x;
//                 } while (w != start);
//             }

//             /* reconstruct root list from A[] */
//             H->min = NULL;
//             for (int i = 0; i <= D; ++i) {
//                 if (A[i] != NULL) {
//                     A[i]->left = A[i]->right = A[i];
//                     H->min = insert_into_list(H->min, A[i]);
//                     if (H->min == NULL || A[i]->key < H->min->key) H->min = A[i];
//                 }
//             }
//             free(A);
//         }
//         H->n -= 1;
//     }
//     return z;
// }

/* Find min in O(1) */
FHNode *FIB_HEAP_MIN(FibonacciHeap *H) {
    return H->min;
}

// /* CUT and CASCADING-CUT for decrease-key */
// static void CUT(FibonacciHeap *H, FHNode *x, FHNode *y) {
//     /* remove x from child list of y */
//     if (y->child == x) {
//         y->child = remove_from_list(y->child, x);
//     } else {
//         /* x not equal y->child head; still remove */
//         remove_from_list(y->child, x);
//     }
//     y->degree -= 1;
//     x->parent = NULL;
//     x->left = x->right = x;
//     x->mark = false;
//     H->min = insert_into_list(H->min, x);
// }

/* ---------- Fixed CUT (assign remove_from_list return) ---------- */
static void CUT(FibonacciHeap *H, FHNode *x, FHNode *y) {
    /* remove x from child list of y and update y->child properly */
    if (y->child == NULL) {
        // nothing to do (shouldn't happen for a valid cut)
    } else if (y->child == x) {
        y->child = remove_from_list(y->child, x);
    } else {
        /* remove and keep head as whatever remove_from_list returns (usually same head) */
        y->child = remove_from_list(y->child, x);
    }
    y->degree -= 1;
    x->parent = NULL;
    x->left = x->right = x;
    x->mark = false;
    H->min = insert_into_list(H->min, x);
}

/* ---------- Fixed EXTRACT_MIN using splice + single pass to clear parents ---------- */
FHNode *FIB_HEAP_EXTRACT_MIN(FibonacciHeap *H) {
    FHNode *z = H->min;
    if (z != NULL) {
        /* 1) splice z's child list into the root list in O(1) if children exist */
        if (z->child != NULL) {
            /* set each child's parent = NULL (walk child list once) */
            FHNode *childHead = z->child;
            FHNode *cur = childHead;
            do {
                cur->parent = NULL;
                cur = cur->right;
            } while (cur != childHead);

            /* concatenate the whole child circular list with root list */
            H->min = concatenate_lists(H->min, z->child);

            /* after splicing, clear z->child (children are now in root list) */
            z->child = NULL;
        }

        /* 2) remove z from root list */
        H->min = remove_from_list(H->min, z);

        if (H->min == NULL) {
            /* heap is now empty */
            H->min = NULL;
        } else {
            /* pick an arbitrary root as start for consolidation */
            H->min = z->right;

            /* Consolidate */
            int D = upper_bound_degree(H->n);
            FHNode **A = (FHNode **)calloc(D + 1, sizeof(FHNode *));
            for (int i = 0; i <= D; ++i) A[i] = NULL;

            /* We need to iterate over the current root list once.
               Since linking modifies the root list, we advance w before handling x. */
            FHNode *w = H->min;
            FHNode *start = w;
            if (w != NULL) {
                do {
                    FHNode *x = w;
                    w = w->right; /* move w forward early to keep iteration safe */
                    int d = x->degree;
                    while (d <= D && A[d] != NULL) {
                        FHNode *y = A[d];
                        if (x->key > y->key) {
                            FHNode *tmp = x; x = y; y = tmp;
                        }
                        FIB_HEAP_LINK(H, y, x);
                        A[d] = NULL;
                        d = d + 1;
                    }
                    if (d > D) {
                        int newD = d + 2;
                        A = (FHNode **)realloc(A, sizeof(FHNode *) * (newD + 1));
                        for (int i = D+1; i <= newD; ++i) A[i] = NULL;
                        D = newD;
                    }
                    A[d] = x;
                } while (w != start);
            }

            /* Rebuild root list from A[] */
            H->min = NULL;
            for (int i = 0; i <= D; ++i) {
                if (A[i] != NULL) {
                    A[i]->left = A[i]->right = A[i];
                    H->min = insert_into_list(H->min, A[i]);
                    if (H->min == NULL || A[i]->key < H->min->key) H->min = A[i];
                }
            }
            free(A);
        }
        H->n -= 1;
    }
    return z;
}

static void CASCADING_CUT(FibonacciHeap *H, FHNode *y) {
    FHNode *z = y->parent;
    if (z != NULL) {
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
    if (y != NULL && x->key < y->key) {
        CUT(H, x, y);
        CASCADING_CUT(H, y);
    }
    if (x->key < H->min->key) H->min = x;
}

/* Delete node x: decrease to -infty and extract-min */
void FIB_HEAP_DELETE(FibonacciHeap *H, FHNode *x) {
    FIB_HEAP_DECREASE_KEY(H, x, INT_MIN);
    FHNode *removed = FIB_HEAP_EXTRACT_MIN(H);
    if (removed) {
        free(removed);
    }
}

/* ---------- Debug / traversal helpers (optional) ---------- */

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

/* ---------- Example usage ---------- */

//#ifdef DEMO_MAIN
int main() {
    FibonacciHeap *H = MAKE_FIB_HEAP();
    FHNode *a = MAKE_NODE(10);
    FHNode *b = MAKE_NODE(3);
    FHNode *c = MAKE_NODE(15);
    FHNode *d = MAKE_NODE(20);
    FHNode *e = MAKE_NODE(25);

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

    return 0;
}
//#endif

