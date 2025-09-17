#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>


int main(int argc, char * argv[]) {

	char oldLink[100] = argv[1];
	char newLink[100] = argv[2];
	int status = link(oldLink,newLink);	
		
	if(status) {
		if (errno == EEXIST) {
			printf("link already exists\n");	
		} else {
			printf("link created !! \n");	
		}

	}

	unlink(newLink);
	printf("link deleted\n");	

}