#include <stdio.h>
#include <stdlib.h>

#define T 2 // Change T to 3 for t=3 B-tree

// Counter for operations
int insertionCounter = 0;
int splitCounter = 0;

// Cost for each operation
int insertionCost = 1; // Assume each insertion takes 1 unit of time
int splitCost = T;     // Assume each split operation takes T units of time

struct BTreeNode {
    int *keys;
    struct BTreeNode **children;
    int n; // Current number of keys
    int leaf; // 1 if node is leaf, 0 otherwise
};

struct BTreeNode *createNode(int leaf) {
    struct BTreeNode *newNode = (struct BTreeNode *)malloc(sizeof(struct BTreeNode));
    newNode->keys = (int *)malloc((2 * T - 1) * sizeof(int));
    newNode->children = (struct BTreeNode **)malloc(2 * T * sizeof(struct BTreeNode *));
    newNode->n = 0;
    newNode->leaf = leaf;
    return newNode;
}

void insertBTree(struct BTreeNode *root, int key);

void insertNonFull(struct BTreeNode *x, int key) {
    int i = x->n - 1;
    if (x->leaf) {
        while (i >= 0 && key < x->keys[i]) {
            x->keys[i + 1] = x->keys[i];
            i--;
        }
        x->keys[i + 1] = key;
        x->n++;
    } else {
        while (i >= 0 && key < x->keys[i]) {
            i--;
        }
        i++;
        if (x->children[i]->n == 2 * T - 1) {
            splitCounter++;
            struct BTreeNode *y = x->children[i];
            struct BTreeNode *z = createNode(y->leaf);
            x->children[i] = z;
            z->n = T - 1;
            for (int j = 0; j < T - 1; j++) {
                z->keys[j] = y->keys[j + T];
            }
            if (!y->leaf) {
                for (int j = 0; j < T; j++) {
                    z->children[j] = y->children[j + T];
                }
            }
            y->n = T - 1;
            for (int j = x->n; j > i; j--) {
                x->children[j + 1] = x->children[j];
            }
            x->children[i + 1] = z;
            for (int j = x->n - 1; j >= i; j--) {
                x->keys[j + 1] = x->keys[j];
            }
            x->keys[i] = y->keys[T - 1];
            x->n++;
            insertNonFull(z, key);
        } else {
            insertNonFull(x->children[i], key);
        }
    }
}

void insertBTree(struct BTreeNode *root, int key) {
    if (root->n == 2 * T - 1) {
        splitCounter++;
        struct BTreeNode *newRoot = createNode(0);
        newRoot->children[0] = root;
        insertNonFull(newRoot, key);
    } else {
        insertNonFull(root, key);
    }
}

int main() {
    struct BTreeNode *root = createNode(1);

    int numKeys;
    printf("Enter the number of keys: ");
    scanf("%d", &numKeys);

    int keys[numKeys];
    printf("Enter the keys: ");
    for (int i = 0; i < numKeys; i++) {
        scanf("%d", &keys[i]);
    }

    for (int i = 0; i < numKeys; i++) {
        insertionCounter++;
        insertBTree(root, keys[i]);
    }

    // Calculate the total cost
    int totalCost = insertionCounter * insertionCost + splitCounter * splitCost;

    // Calculate the amortized cost
    double amortizedCost = (double)totalCost / insertionCounter;

    printf("Amortized cost per insertion: %.2lf\n", amortizedCost);
    printf("Total cost: %d\n", totalCost);
    printf("Insertion operations: %d\n", insertionCounter);
    printf("Split operations: %d\n", splitCounter);

    return 0;
}
