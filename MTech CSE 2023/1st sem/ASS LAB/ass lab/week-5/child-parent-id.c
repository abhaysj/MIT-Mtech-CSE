#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>


int main(int argc, char * argv[]){
	pid_t myId = getpid();
	pid_t parentId = getppid();

	printf("myid : %d | parent id : %d \n", myId, parentId);

	pid_t pid;
	int status;
	int n;
	printf("fork program starting\n");
	
	pid = fork();	
	pid_t child_pid = getpid();
	
	if (pid == 0) {
		printf("hello from child %d\n", getpid());
		printf("id of parent from child : %d\n", getppid());
	} else {
		printf("hello from parent %d \n", getpid());
	}
	
	wait(&status);
}