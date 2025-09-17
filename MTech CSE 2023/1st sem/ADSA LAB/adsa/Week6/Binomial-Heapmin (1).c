#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h> // For INT_MAX

struct node
{
    int n, degree;
    struct node *parent, *child, *sibling;
};

typedef struct node *Node;
Node H = NULL;
Node Hr = NULL;

Node MAKE_HEAP();
Node HEAP_INSERT(Node H, Node x);
Node HEAP_UNION(Node H1, Node H2);
void HEAP_LINK(Node y, Node z);
Node HEAP_MERGE(Node H1, Node H2);
void DISPLAY(Node H, int depth);
void REVERT_LIST(Node y);
int FIND_MINIMUM_KEY(Node H);

Node CREATE_NODE(int k)
{
    Node p;
    p = (Node)malloc(sizeof(struct node));
    p->n = k;
    return p;
}

Node MAKE_HEAP()
{
    Node np;
    np = NULL;
    return np;
}

Node HEAP_INSERT(Node H, Node x)
{
    Node H1 = MAKE_HEAP();
    x->parent = NULL;
    x->child = NULL;
    x->sibling = NULL;
    x->degree = 0;
    H1 = x;
    H = HEAP_UNION(H, H1);
    return H;
}

Node HEAP_UNION(Node H1, Node H2)
{
    Node prev_x;
    Node next_x;
    Node x;
    Node H = MAKE_HEAP();
    H = HEAP_MERGE(H1, H2);
    if (H == NULL)
        return H;
    prev_x = NULL;
    x = H;
    next_x = x->sibling;
    while (next_x != NULL)
    {
        if ((x->degree != next_x->degree) || ((next_x->sibling != NULL) && (next_x->sibling)->degree == x->degree))
        {
            prev_x = x;
            x = next_x;
        }
        else
        {
            if (x->n <= next_x->n)
            {
                x->sibling = next_x->sibling;
                HEAP_LINK(next_x, x);
            }
            else
            {
                if (prev_x == NULL)
                    H = next_x;
                else
                    prev_x->sibling = next_x;
                HEAP_LINK(x, next_x);
                x = next_x;
            }
        }
        next_x = x->sibling;
    }
    return H;
}

Node HEAP_MERGE(Node H1, Node H2)
{
    Node H = MAKE_HEAP();
    Node y, z, a, b;
    y = H1;
    z = H2;
    if (y != NULL)
    {
        if (z != NULL && y->degree <= z->degree)
            H = y;
        else if (z != NULL && y->degree > z->degree)
            H = z;
        else
            H = y;
    }
    else
        H = z;
    while (y != NULL && z != NULL)
    {
        if (y->degree < z->degree)
        {
            y = y->sibling;
        }
        else if (y->degree == z->degree)
        {
            a = y->sibling;
            y->sibling = z;
            y = a;
        }
        else
        {
            b = z->sibling;
            z->sibling = y;
            z = b;
        }
    }
    return H;
}

void HEAP_LINK(Node y, Node z)
{
    y->parent = z;
    y->sibling = z->child;
    z->child = y;
    z->degree = z->degree + 1;
}

void DISPLAY(Node H, int depth)
{
    if (H == NULL)
        return;

    printf("%*sKey: %d, Degree: %d\n", depth * 4, "", H->n, H->degree);

    DISPLAY(H->child, depth + 1);
    DISPLAY(H->sibling, depth);
}

int FIND_MINIMUM_KEY(Node H)
{
    if (H == NULL)
        return INT_MAX; // Return maximum integer value if the heap is empty

    int min_key = H->n; // Initialize min_key with the key of the root node

    Node current_node = H->sibling;
    while (current_node != NULL)
    {
        if (current_node->n < min_key)
            min_key = current_node->n;

        current_node = current_node->sibling;
    }

    return min_key;
}
double calculate_amortized_cost_aggregate(int num_operations)
{
    // Assume each operation takes constant time
    return (double)num_operations;
}

// Calculate amortized cost using the amortized method
double calculate_amortized_cost_amortized(int num_operations)
{
    // Each operation is O(log n) in a binomial heap, so we assume it takes log(n) time
    return (double)num_operations * log2(num_operations);
}

// Calculate amortized cost using the potential method
double calculate_amortized_cost_potential(int num_operations, int num_nodes)
{
    // Potential function: Φ(H) = c * num_nodes, where c is a constant
    // The potential before an operation is c * num_nodes_before
    // The potential after an operation is c * num_nodes_after
    // Therefore, the change in potential is c * (num_nodes_after - num_nodes_before)

    double c = 1.0; // constant for the potential function
    int num_nodes_before = num_nodes - num_operations; // Assume each operation decreases the number of nodes by 1

    // Change in potential
    double delta_potential = c * (num_nodes - num_nodes_before);

    // Amortized cost using potential method
    return (double)num_operations + delta_potential;
}

int main()
{
    Node np;
    int n, v;
    printf("Enter the Number of values : ");
    scanf("%d", &n);
    printf("Enter %d Values : ", n);
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &v);
        np = CREATE_NODE(v);
        H = HEAP_INSERT(H, np);
    }

    DISPLAY(H, 0);

    

    // Find the minimum key in the binomial heap
    int min_key = FIND_MINIMUM_KEY(H);
    printf("Minimum Key in the Binomial Heap: %d\n", min_key);
int num_operations = log2(n); // Assume n operations (insertions in this case)
    int num_nodes = log2(n);       // Assume initial number of nodes is n

    // Calculate amortized cost using different methods
    double amortized_cost_aggregate = calculate_amortized_cost_aggregate(num_operations);
    double amortized_cost_amortized = calculate_amortized_cost_amortized(num_operations);
    double amortized_cost_potential = calculate_amortized_cost_potential(num_operations, num_nodes);

    printf("\nAmortized Cost (Aggregate Method): %.2f\n", amortized_cost_aggregate);
    printf("Amortized Cost (Accounting Method): %.2f\n", amortized_cost_amortized);
    printf("Amortized Cost (Potential Method): %.2f\n", amortized_cost_potential);

    return 0;
    
}

