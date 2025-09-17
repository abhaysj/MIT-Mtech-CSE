#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>


int main(int argc, char * argv[]) {

	char oldLink[100] = argv[1];
	char newLink[100] = argv[2];
	int status = symlink(oldLink,newLink);	
		
	printf("%d\n", status);

	if(status) {
		if (errno == EEXIST) {
			printf("link already exists\n");	
		}		
	}

	unlink(newLink);

}