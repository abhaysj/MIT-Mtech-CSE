#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

int main() {
    char dir_path[100];
    printf("Enter the directory path: ");
    scanf("%s", dir_path);

    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("Error opening directory");
        return 1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) { // Check if it's a regular file
            char file_path[200];
            snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, entry->d_name);
            
            printf("File: %s\n", entry->d_name);
            printf("Grant read permission? (1/0): ");
            
            int answer;
            scanf("%d", &answer);

            if (answer == 1) {
                if (chmod(file_path, S_IRUSR | S_IRGRP | S_IROTH | S_IXUSR | S_IWUSR) == -1) {
                    perror("Error changing file permissions");
                } else {
                    printf("Read permission granted.\n");
                }
            }
        }
    }

    closedir(dir);
    return 0;
}
