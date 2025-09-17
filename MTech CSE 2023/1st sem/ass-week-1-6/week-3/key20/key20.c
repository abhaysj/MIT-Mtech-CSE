#include <stdio.h>
#include <stdlib.h>

#define LINES_PER_PAGE 20

void more(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Error: File '%s' not found.\n", filename);
        return;
    }

    char line[1024];
    for (int line_number = 1; fgets(line, sizeof(line), file); line_number++)
    {
        printf("%s", line);
        if (line_number % LINES_PER_PAGE == 0)
        {
            printf("\n--- Press Enter to continue, or q to quit ---\n");
            if (getchar() == 'q')
                break;
        }
    }

    fclose(file);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <filename1> [filename2 ...]\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++)
    {
        more(argv[i]);
    }

    return 0;
}
