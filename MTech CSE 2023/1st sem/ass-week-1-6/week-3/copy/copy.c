#include <stdio.h>

int main() {
    FILE *sourceFile = fopen("source.txt", "r");
    FILE *destinationFile = fopen("destination.txt", "w");

    if (sourceFile == NULL) {
        printf("Error: Source file 'source.txt' not found.\n");
        return 1;
    }

    if (destinationFile == NULL) {
        printf("Error: Unable to create destination file 'destination.txt'.\n");
        fclose(sourceFile);
        return 1;
    }

    int character;
    while ((character = fgetc(sourceFile)) != EOF) {
        fputc(character, destinationFile);
    }

    printf("File copied successfully.\n");

    fclose(sourceFile);
    fclose(destinationFile);

    return 0;
}
