#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char * argv[]) {

	pid_t pid = fork();
	int status;

	if (pid < 0) {
		perror("error in forking ....");
		exit(1);
	}

	if (pid == 0) {
		for(int i=0; i<3; i++) {
			printf("child is working \n");
			sleep(1);
		}
	} else {
		printf("parent is waiting\n");
		wait(&status);
		printf("parent wait completed\n");
	}

}