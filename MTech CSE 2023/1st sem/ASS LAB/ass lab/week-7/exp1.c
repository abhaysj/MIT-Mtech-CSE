#include <stdio.h>
#include <stdlib.h>

// Memory Allocation Block (MAB) structure
typedef struct mab
{
    int offset;
    int size;
    int allocated;
    struct mab *next;
    struct mab *prev;
} Mab;

typedef Mab *MabPtr;

// Check if memory of specified size is available
MabPtr memChk(MabPtr m, int size)
{
    MabPtr current = m;
    while (current != NULL)
    {
        if (!current->allocated && current->size >= size)
        {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// Allocate memory block of specified size
MabPtr memAlloc(MabPtr m, int size)
{
    MabPtr block = memChk(m, size);
    if (block != NULL)
    {
        block->allocated = 1;
        if (block->size > size)
        {
            memSplit(block, size);
        }
    }
    return block;
}

// Free memory block
void memFree(MabPtr m)
{
    if (m != NULL)
    {
        m->allocated = 0;
        memMerge(m);
    }
}

// Merge adjacent free memory blocks
void memMerge(MabPtr m)
{
    if (m->next != NULL && !(m->next->allocated))
    {
        m->size += m->next->size;
        m->next = m->next->next;
        if (m->next != NULL)
        {
            m->next->prev = m;
        }
    }
}

// Split a memory block into allocated and free blocks
void memSplit(MabPtr m, int size)
{
    MabPtr newBlock = (MabPtr)malloc(sizeof(Mab));
    newBlock->offset = m->offset + size;
    newBlock->size = m->size - size;
    newBlock->allocated = 0;
    newBlock->prev = m;
    newBlock->next = m->next;
    if (m->next != NULL)
    {
        m->next->prev = newBlock;
    }
    m->size = size;
    m->next = newBlock;
}

int main()
{
    // Initialize memory blocks
    Mab block1 = {0, 100, 0, NULL, NULL};
    Mab block2 = {100, 200, 0, NULL, NULL};
    block1.next = &block2;

    // Allocate memory
    MabPtr allocatedBlock = memAlloc(&block1, 150);
    if (allocatedBlock != NULL)
    {
        printf("Memory allocated successfully.\n");

        // Free allocated memory
        memFree(allocatedBlock);
        printf("Memory freed successfully.\n");
    }
    else
    {
        printf("Memory allocation failed.\n");
    }

    return 0;
}
