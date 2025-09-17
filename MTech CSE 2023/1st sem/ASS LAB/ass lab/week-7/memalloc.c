#include <stdio.h>
#include <stdlib.h>

struct mab {
    int offset;
    int size;
    int allocated;
    struct mab *prev;
    struct mab *next;
};

typedef struct mab Mab;
typedef Mab *Mabptr;

int main() {

}