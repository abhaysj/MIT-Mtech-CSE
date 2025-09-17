// B-Tree (CLRS) — Search, Create, Split-Child, Insert, Insert-NonFull
// Works for any minimum degree t >= 2. Demo provided for t=2 and t=3.
// Compile:  gcc -O2 -std=c11 btree.c -o btree
// Run:      ./btree

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct BTreeNode {
    int *key;                 // Array of keys (size up to 2*t - 1)
    struct BTreeNode **child; // Array of child pointers (size up to 2*t)
    int n;                    // Current number of keys stored
    bool leaf;                // True if this node is a leaf (no children)
    int t;                    // Minimum degree (branching factor)
} BTreeNode;

typedef struct {
    BTreeNode *root;
    int t;
} BTree;

// Allocate a new node with capacity derived from t
static BTreeNode* allocate_node(int t, bool leaf) {
    BTreeNode *x = (BTreeNode*)malloc(sizeof(BTreeNode));
    x->t = t;
    x->leaf = leaf;
    x->n = 0;
    // Max keys = 2t-1; Max children = 2t
    x->key   = (int*)malloc(sizeof(int) * (2*t - 1));
    x->child = (BTreeNode**)malloc(sizeof(BTreeNode*) * (2*t));
    return x;
}

// B-TREE-SEARCH(x, k) -> returns pointer to node containing k, or NULL
BTreeNode* btree_search(BTreeNode *x, int k, int *idx_out) {
    int i = 0;
    while (i < x->n && k > x->key[i]) i++;
    if (i < x->n && k == x->key[i]) { if (idx_out) *idx_out = i; return x; }
    if (x->leaf) return NULL;
    return btree_search(x->child[i], k, idx_out);
}

// B-TREE-CREATE(T)
void btree_create(BTree *T, int t) {
    T->t = t;
    BTreeNode *x = allocate_node(t, true);
    x->n = 0;
    T->root = x;
}

// B-TREE-SPLIT-CHILD(x, i)
// Precondition: child x->child[i] is full (has 2t-1 keys)
void btree_split_child(BTreeNode *x, int i) {
    int t = x->t;
    BTreeNode *y = x->child[i];             // full child
    BTreeNode *z = allocate_node(t, y->leaf);

    z->n = t - 1;                            // z gets the right t-1 keys
    for (int j = 0; j < t - 1; j++)
        z->key[j] = y->key[j + t];

    if (!y->leaf) {
        for (int j = 0; j < t; j++)
            z->child[j] = y->child[j + t];
    }

    y->n = t - 1;                            // y keeps the left t-1 keys

    // Make room for new child in x
    for (int j = x->n; j >= i + 1; j--)
        x->child[j + 1] = x->child[j];
    x->child[i + 1] = z;

    // Move a key from y up into x
    for (int j = x->n - 1; j >= i; j--)
        x->key[j + 1] = x->key[j];
    x->key[i] = y->key[t - 1];
    x->n += 1;
}

// B-TREE-INSERT-NONFULL(x, k) — assumes x is not full
void btree_insert_nonfull(BTreeNode *x, int k) {
    int i = x->n - 1;

    if (x->leaf) {
        // Move larger keys one step to the right to make place for k
        while (i >= 0 && k < x->key[i]) {
            x->key[i + 1] = x->key[i];
            i--;
        }
        x->key[i + 1] = k;
        x->n += 1;
    } else {
        // Find child to descend into
        while (i >= 0 && k < x->key[i]) i--;
        i += 1;

        // If the child is full, split first
        if (x->child[i]->n == 2 * x->t - 1) {
            btree_split_child(x, i);
            // After split, decide which child we should go into
            if (k > x->key[i]) i++;
        }
        btree_insert_nonfull(x->child[i], k);
    }
}

// B-TREE-INSERT(T, k)
void btree_insert(BTree *T, int k) {
    BTreeNode *r = T->root;
    int t = T->t;

    if (r->n == 2*t - 1) {
        // Root is full: make new root s and split child
        BTreeNode *s = allocate_node(t, false);
        T->root = s;
        s->n = 0;
        s->child[0] = r;
        btree_split_child(s, 0);
        // Now insert into the correct child
        int i = (k > s->key[0]) ? 1 : 0;
        btree_insert_nonfull(s->child[i], k);
    } else {
        btree_insert_nonfull(r, k);
    }
}

// Utility: inorder-like traversal (prints keys in sorted order)
void btree_traverse(BTreeNode *x) {
    if (!x) return;
    int i;
    for (i = 0; i < x->n; i++) {
        if (!x->leaf) btree_traverse(x->child[i]);
        printf("%d ", x->key[i]);
    }
    if (!x->leaf) btree_traverse(x->child[i]);
}

// Utility: find predecessor (max key in subtree)
int btree_get_pred(BTreeNode *x) {
    while (!x->leaf)
        x = x->child[x->n];
    return x->key[x->n - 1];
}

// Utility: find successor (min key in subtree)
int btree_get_succ(BTreeNode *x) {
    while (!x->leaf)
        x = x->child[0];
    return x->key[0];
}

// Merge child[i] and child[i+1] into one node
void btree_merge(BTreeNode *x, int i) {
    int t = x->t;
    BTreeNode *y = x->child[i];
    BTreeNode *z = x->child[i+1];

    // Bring down x->key[i] into y
    y->key[t-1] = x->key[i];

    // Copy keys from z into y
    for (int j = 0; j < z->n; j++)
        y->key[j + t] = z->key[j];

    // Copy children if not leaf
    if (!y->leaf) {
        for (int j = 0; j <= z->n; j++)
            y->child[j + t] = z->child[j];
    }

    y->n += z->n + 1;

    // Shift keys in x
    for (int j = i; j < x->n - 1; j++)
        x->key[j] = x->key[j+1];
    for (int j = i+1; j < x->n; j++)
        x->child[j] = x->child[j+1];
    x->n--;

    free(z->key);
    free(z->child);
    free(z);
}

// Fill child[i] if it has only t-1 keys
void btree_fill(BTreeNode *x, int i) {
    int t = x->t;
    if (i > 0 && x->child[i-1]->n >= t) {
        // Borrow from left sibling
        BTreeNode *c = x->child[i];
        BTreeNode *sibling = x->child[i-1];
        for (int j = c->n-1; j >= 0; j--)
            c->key[j+1] = c->key[j];
        if (!c->leaf) {
            for (int j = c->n; j >= 0; j--)
                c->child[j+1] = c->child[j];
        }
        c->key[0] = x->key[i-1];
        if (!c->leaf)
            c->child[0] = sibling->child[sibling->n];
        x->key[i-1] = sibling->key[sibling->n-1];
        c->n += 1;
        sibling->n -= 1;
    }
    else if (i < x->n && x->child[i+1]->n >= t) {
        // Borrow from right sibling
        BTreeNode *c = x->child[i];
        BTreeNode *sibling = x->child[i+1];
        c->key[c->n] = x->key[i];
        if (!c->leaf)
            c->child[c->n+1] = sibling->child[0];
        x->key[i] = sibling->key[0];
        for (int j = 1; j < sibling->n; j++)
            sibling->key[j-1] = sibling->key[j];
        if (!sibling->leaf) {
            for (int j = 1; j <= sibling->n; j++)
                sibling->child[j-1] = sibling->child[j];
        }
        c->n += 1;
        sibling->n -= 1;
    }
    else {
        // Merge with a sibling
        if (i < x->n)
            btree_merge(x, i);
        else
            btree_merge(x, i-1);
    }
}

// Recursive delete
void btree_delete_node(BTreeNode *x, int k) {
    int idx = 0;
    while (idx < x->n && x->key[idx] < k)
        idx++;

    if (idx < x->n && x->key[idx] == k) {
        // Case 1: key in leaf
        if (x->leaf) {
            for (int i = idx+1; i < x->n; i++)
                x->key[i-1] = x->key[i];
            x->n--;
        }
        // Case 2: key in internal node
        else {
            int t = x->t;
            if (x->child[idx]->n >= t) {
                int pred = btree_get_pred(x->child[idx]);
                x->key[idx] = pred;
                btree_delete_node(x->child[idx], pred);
            }
            else if (x->child[idx+1]->n >= t) {
                int succ = btree_get_succ(x->child[idx+1]);
                x->key[idx] = succ;
                btree_delete_node(x->child[idx+1], succ);
            }
            else {
                btree_merge(x, idx);
                btree_delete_node(x->child[idx], k);
            }
        }
    }
    else {
        // Case 3: key not in this node
        if (x->leaf) return; // key not found

        bool flag = (idx == x->n);
        if (x->child[idx]->n < x->t)
            btree_fill(x, idx);

        if (flag && idx > x->n)
            btree_delete_node(x->child[idx-1], k);
        else
            btree_delete_node(x->child[idx], k);
    }
}

// Wrapper: handle shrinking root
void btree_delete(BTree *T, int k) {
    if (!T->root) return;
    btree_delete_node(T->root, k);

    if (T->root->n == 0) {
        BTreeNode *old = T->root;
        if (T->root->leaf)
            T->root = NULL;
        else
            T->root = T->root->child[0];
        free(old->key);
        free(old->child);
        free(old);
    }
}


// Demo: insert a sequence of keys and show search
static void demo_with_t(int t) {
    printf("\n=== Demo with t = %d ===\n", t);
    BTree T; btree_create(&T, t);

    // Example keys
    int keys[] = {20, 10, 30, 50, 40, 60, 70, 80, 90, 0, 5, 15, 25, 35, 45};
    int m = (int)(sizeof(keys)/sizeof(keys[0]));

    for (int i = 0; i < m; i++)
        btree_insert(&T, keys[i]);

    printf("In-order traversal: ");
    btree_traverse(T.root);
    printf("\n");

    // Search demo
    int queries[] = {35, 42, 0, 90};
    for (int i = 0; i < 4; i++) {
        int idx = -1;
        BTreeNode *res = btree_search(T.root, queries[i], &idx);
        if (res) printf("Found %d\n", queries[i]);
        else     printf("Not found %d\n", queries[i]);
    }

        // Delete demo
    printf("Deleting 30, 50, 70...\n");
    btree_delete(&T, 30);
    btree_delete(&T, 50);
    btree_delete(&T, 70);
    printf("Traversal after deletes: ");
    btree_traverse(T.root);
    printf("\n");

}

int main(void) {
    demo_with_t(2);  // minimum degree 2
    demo_with_t(3);  // minimum degree 3
    return 0;
}
