#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

int main()
{
    int fd;
    char buffer[11];
    off_t offset;

    // Open the file for reading and writing.
    fd = open("file.txt", O_RDWR);

    FILE* fp = fopen("file.txt", "r");

    if (fd == -1)
    {
        perror("open");
        exit(1);
    }

    fseek(fp, 0L, SEEK_END);

    // calculating the size of the file
    long int res = ftell(fp);
    rewind(fp);

    for (offset = 0; offset < res; offset += 11) {

        if(offset == 0 || offset == ((res%10)/2)*11 || offset == (res%10-1)*11) {
            lseek(fd, offset, SEEK_SET);
            read(fd, buffer, 11);
            write(1, buffer, 11);
            printf("\n");
        }
    }

    close(fd);

    return 0;
}