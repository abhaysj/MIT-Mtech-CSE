#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

int main(int argc, char *argv[])
{
    // Check if the user has provided a directory name
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <directory>\n", argv[0]);
        return 1;
    }

    // Open the directory
    DIR *dir = opendir(argv[1]);
    if (dir == NULL)
    {
        perror("opendir");
        return 1;
    }

    // Iterate over the files in the directory
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        // Get the inode number of the file
        struct stat statbuf;
        stat(entry->d_name, &statbuf);
        int inode = statbuf.st_ino;

        // Print the inode number
        printf("%d\n", inode);
    }

    // Close the directory
    closedir(dir);

    return 0;
}