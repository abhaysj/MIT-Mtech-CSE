#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

struct stat statbuf;

int main(int argc, char * argv[]) {

	char *filename = argv[1];
	stat(filename, &statbuf);
	
	printf("Id of device : %ld\n", statbuf.st_dev );
	printf("Inode number : %ld\n", statbuf.st_ino );
	printf("Permission   : %d\n", statbuf.st_mode );
	printf("N(Hardlinks) : %ld\n", statbuf.st_nlink );
	printf("UserId Owner : %d\n", statbuf.st_uid );
	printf("GroupId Owner: %d\n", statbuf.st_gid );
	printf("DeviceId     : %ld\n", statbuf.st_rdev );
	printf("size (bytes) : %ld\n", statbuf.st_size );
	printf("block size   : %ld\n", statbuf.st_blksize );
	printf("block alloc  : %ld\n", statbuf.st_blocks );
	printf("time         : %ld\n", statbuf.st_atime );
	printf("time         : %ld\n", statbuf.st_ctime );
	printf("time         : %ld\n", statbuf.st_mtime );

}