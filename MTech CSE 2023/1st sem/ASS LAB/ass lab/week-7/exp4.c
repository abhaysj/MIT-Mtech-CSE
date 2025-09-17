#include <stdio.h>
#include <stdbool.h>

#define NUM_PAGES 4
#define NUM_REFERENCES 20

int main()
{
    int pages[NUM_PAGES];          // Page frames
    bool reference_bit[NUM_PAGES]; // Reference bit for each page frame
    int references[NUM_REFERENCES] = {0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4, 1, 2, 3, 0, 4, 2, 3, 1};
    int page_faults = 0;
    int hits = 0;

    for (int i = 0; i < NUM_PAGES; i++)
    {
        pages[i] = -1; // Empty page frame
        reference_bit[i] = false;
    }

    for (int i = 0; i < NUM_REFERENCES; i++)
    {
        int page = references[i];
        bool found = false;

        for (int j = 0; j < NUM_PAGES; j++)
        {
            if (pages[j] == page)
            {
                hits++;
                reference_bit[j] = true;
                found = true;
                break;
            }
        }

        if (!found)
        {
            page_faults++;

            while (true)
            {
                int victim = -1;
                for (int j = 0; j < NUM_PAGES; j++)
                {
                    if (!reference_bit[j])
                    {
                        victim = j;
                        break;
                    }
                }

                if (victim != -1)
                {
                    pages[victim] = page;
                    reference_bit[victim] = true;
                    break;
                }

                for (int j = 0; j < NUM_PAGES; j++)
                {
                    reference_bit[j] = false;
                }
            }
        }

        printf("Step %d: ", i + 1);
        for (int j = 0; j < NUM_PAGES; j++)
        {
            printf("%d ", pages[j]);
        }
        printf("\n");
    }

    double hit_ratio = (double)hits / NUM_REFERENCES;

    printf("Total Page Faults: %d\n", page_faults);
    printf("Hit Ratio: %.2lf\n", hit_ratio);

    return 0;
}
