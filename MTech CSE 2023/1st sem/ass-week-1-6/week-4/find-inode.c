#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

struct stat statbuf;

int main(int argc, char * argv[]) {

	char *filename = argv[1];
	stat(filename, &statbuf);
	printf("inode : %ld\n", statbuf.st_ino);

}