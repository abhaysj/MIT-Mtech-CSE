#include <stdio.h>

#define NUM_SEGMENTS 5
#define SEGMENT_SIZE 1024

// Segment Table Entry
typedef struct
{
    int base;
    int limit;
} SegmentEntry;

// Segment Table
SegmentEntry segmentTable[NUM_SEGMENTS];

void initializeSegmentTable()
{
    // Set base and limit for each segment
    segmentTable[0].base = 0;
    segmentTable[0].limit = 512;

    segmentTable[1].base = 512;
    segmentTable[1].limit = 256;

    segmentTable[2].base = 768;
    segmentTable[2].limit = 128;

    segmentTable[3].base = 896;
    segmentTable[3].limit = 256;

    segmentTable[4].base = 1152;
    segmentTable[4].limit = 128;
}

int convertLogicalToPhysical(int segment, int offset)
{
    if (segment < 0 || segment >= NUM_SEGMENTS)
    {
        printf("Invalid segment number.\n");
        return -1;
    }

    if (offset < 0 || offset >= segmentTable[segment].limit)
    {
        printf("Invalid offset for the segment.\n");
        return -1;
    }

    return segmentTable[segment].base + offset;
}

int main()
{
    initializeSegmentTable();

    // Compute physical addresses for specific byte offsets
    int address1 = convertLogicalToPhysical(2, 53);
    int address2 = convertLogicalToPhysical(3, 852);
    int address3 = convertLogicalToPhysical(0, 1222);

    // Print the computed physical addresses
    printf("Physical address for (i) 53 byte of segment 2: %d\n", address1);
    printf("Physical address for (ii) 852 byte of segment 3: %d\n", address2);
    printf("Physical address for (iii) 1222 byte of segment 0: %d\n", address3);

    return 0;
}
