/* binomial_heap.c
   CLRS-style binomial heap implementation in C.
   Node representation: key, degree, parent, child, sibling.
   Implements: make-node, merge, union, insert, minimum, extract-min, decrease-key, delete.
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct BinNode {
    int key;
    int degree;
    struct BinNode *parent;
    struct BinNode *child;   // leftmost child
    struct BinNode *sibling; // next sibling (or next root when at root list)
} BinNode;

/* ---------- utilities ---------- */

BinNode* make_node(int key) {
    BinNode *n = (BinNode*)malloc(sizeof(BinNode));
    if (!n) { perror("malloc"); exit(EXIT_FAILURE); }
    n->key = key;
    n->degree = 0;
    n->parent = n->child = n->sibling = NULL;
    return n;
}

/* Print a binomial tree (recursive). indent controls visual depth. */
void print_tree(BinNode *root, int indent) {
    while (root) {
        for (int i = 0; i < indent; ++i) printf("  ");
        printf("%d (deg=%d)\n", root->key, root->degree);
        if (root->child) print_tree(root->child, indent + 1);
        root = root->sibling;
    }
}

/* Print the heap's root list and subtrees */
void print_heap(BinNode *H) {
    printf("Heap root-list:\n");
    BinNode *x = H;
    while (x) {
        printf("B%d:\n", x->degree);
        print_tree(x, 1);
        x = x->sibling;
    }
    printf("---- end heap ----\n");
}

/* ---------- CLRS core operations ---------- */

/* BINOMIAL-LINK: make y a child of z (assumes y->key >= z->key). */
void binomial_link(BinNode *y, BinNode *z) {
    y->parent = z;
    y->sibling = z->child;
    z->child = y;
    z->degree += 1;
}

/* BINOMIAL-HEAP-MERGE: merge two root lists sorted by degree (ascending). */
BinNode* binomial_heap_merge(BinNode *h1, BinNode *h2) {
    if (!h1) return h2;
    if (!h2) return h1;
    BinNode *head = NULL, *tail = NULL;//head - Pointer to the first node in the merged list., tail - Pointer to the last node so far in the merged list.
    BinNode *a = h1, *b = h2;
    while (a && b) {
        BinNode *take;//A temporary pointer to the node we decide to "take" next from either a or b.
        if (a->degree <= b->degree) { take = a; a = a->sibling; }
        else { take = b; b = b->sibling; }
        if (!head) { head = tail = take; }
        else { tail->sibling = take; tail = take; }
    }
    if (a) tail->sibling = a;
    else if (b) tail->sibling = b;
    return head;
}

/* BINOMIAL-HEAP-UNION: merge two heaps' root lists and then fix equal-degree trees. */
BinNode* binomial_heap_union(BinNode *h1, BinNode *h2) {
    BinNode *h = binomial_heap_merge(h1, h2);
    if (!h) return NULL;

    BinNode *prev_x = NULL;
    BinNode *x = h;
    BinNode *next_x = x->sibling;

    while (next_x) {
        if (x->degree != next_x->degree ||
            (next_x->sibling && next_x->sibling->degree == x->degree)) {
            /* move forward */
            prev_x = x;
            x = next_x;
        } else {
            if (x->key <= next_x->key) {
                /* link next_x under x */
                x->sibling = next_x->sibling;
                binomial_link(next_x, x);
            } else {
                /* link x under next_x */
                if (prev_x == NULL) h = next_x;
                else prev_x->sibling = next_x;
                binomial_link(x, next_x);
                x = next_x;
            }
        }
        next_x = x->sibling;
    }
    return h;
}

/* BINOMIAL-HEAP-INSERT (returns new heap head) */
BinNode* binomial_heap_insert(BinNode *H, int key) {
    BinNode *node = make_node(key);
    return binomial_heap_union(H, node);
}

/* BINOMIAL-HEAP-MINIMUM (scan root list) */
BinNode* binomial_heap_minimum(BinNode *H) {
    if (!H) return NULL;
    BinNode *x = H;
    BinNode *y = H;
    int min = x->key;
    while (x) {
        if (x->key < min) { min = x->key; y = x; }
        x = x->sibling;
    }
    return y;
}

/* BINOMIAL-HEAP-EXTRACT-MIN:
   Remove root with minimum key, reverse its child list to create new heap,
   then UNION with remaining heap.
   Returns the extracted node (caller should free it when done).
*/
BinNode* binomial_heap_extract_min(BinNode **Hptr) {//By passing &H (BinNode **), the function can modify the original pointer to point to the new root after extraction.
    BinNode *H = *Hptr;
    if (!H) return NULL;

    /* find min root and its previous in root list */
    BinNode *minNode = H;
    BinNode *minPrev = NULL;
    BinNode *prev = NULL;
    BinNode *curr = H;
    int minKey = curr->key;
    while (curr) {
        if (curr->key < minKey) {
            minKey = curr->key;
            minNode = curr;
            minPrev = prev;
        }
        prev = curr;
        curr = curr->sibling;
    }

    /* remove minNode from root list */
    if (!minPrev) *Hptr = minNode->sibling;
    else minPrev->sibling = minNode->sibling;

    /* reverse minNode->child to form new heap H' */
    BinNode *child = minNode->child;
    BinNode *rev = NULL;
    while (child) {
        BinNode *next = child->sibling;
        child->sibling = rev;
        child->parent = NULL;
        rev = child;
        child = next;
    }

    /* union the two heaps (remaining H and rev) */
    *Hptr = binomial_heap_union(*Hptr, rev);

    /* detach minNode */
    minNode->child = minNode->sibling = minNode->parent = NULL;
    minNode->degree = 0;
    return minNode;
}

/* Helper: find node by key (first match) -- returns pointer to node or NULL.
   Recurses into child lists and siblings. Note: if keys duplicate, returns first found.
*/
BinNode* binomial_heap_find(BinNode *root, int key) {
    BinNode *x = root;
    while (x) {
        if (x->key == key) return x;
        BinNode *res = binomial_heap_find(x->child, key);
        if (res) return res;
        x = x->sibling;
    }
    return NULL;
}

/* BINOMIAL-HEAP-DECREASE-KEY: set node->key to k (k must be <= current key),
   then bubble up by swapping keys with parent while necessary.
*/
void binomial_heap_decrease_key(BinNode *H, BinNode *x, int k) {
    if (!x) return;
    if (k > x->key) {
        fprintf(stderr, "Error: new key is greater than current key\n");
        return;
    }
    x->key = k;
    BinNode *y = x;
    BinNode *z = y->parent;
    while (z && y->key < z->key) {
        int tmp = y->key; y->key = z->key; z->key = tmp;
        y = z; z = y->parent;
    }
}

/* BINOMIAL-HEAP-DELETE: decrease key to -inf then extract-min.
   Returns pointer to deleted node (caller may free).
*/
BinNode* binomial_heap_delete(BinNode **Hptr, BinNode *x) {
    if (!x) return NULL;
    binomial_heap_decrease_key(*Hptr, x, INT_MIN);
    return binomial_heap_extract_min(Hptr);
}

/* Optional: free entire heap (post-order) */
void free_subtree(BinNode *root) {
    while (root) {
        if (root->child) free_subtree(root->child);
        BinNode *next = root->sibling;
        free(root);
        root = next;
    }
}

void free_heap(BinNode *H) {
    free_subtree(H);
}

/* ---------- demo usage in main ---------- */
int main(void) {
    BinNode *H = NULL;

    /* Insert some items */
    H = binomial_heap_insert(H, 10);
    H = binomial_heap_insert(H, 20);
    H = binomial_heap_insert(H, 5);
    H = binomial_heap_insert(H, 30);
    H = binomial_heap_insert(H, 2);
    H = binomial_heap_insert(H, 3);
    H = binomial_heap_insert(H, 34);
    H = binomial_heap_insert(H, 54);
    //H = binomial_heap_insert(H, 2);
    printf("After inserts:\n");
    print_heap(H);

    /* find-min */
    BinNode *mn = binomial_heap_minimum(H);
    if (mn) printf("Minimum key = %d\n\n", mn->key);

    /* extract-min */
    BinNode *ex = binomial_heap_extract_min(&H);
    if (ex) {
        printf("Extracted min = %d\n", ex->key);
        free(ex);
    }
    printf("After extract-min:\n");
    print_heap(H);

    /* decrease-key of node with key 20 to 3 */
    BinNode *node20 = binomial_heap_find(H, 20);
    if (node20) {
        printf("Decreasing key 20 -> 3\n");
        binomial_heap_decrease_key(H, node20, 3);
    } else printf("Node 20 not found\n");
    print_heap(H);

    /* delete node with key 30 */
    BinNode *node30 = binomial_heap_find(H, 30);
    if (node30) {
        printf("Deleting node with key 30\n");
        BinNode *del = binomial_heap_delete(&H, node30);
        if (del) free(del);
    } else printf("Node 30 not found\n");
    print_heap(H);

    /* cleanup */
    free_heap(H);
    return 0;
}



