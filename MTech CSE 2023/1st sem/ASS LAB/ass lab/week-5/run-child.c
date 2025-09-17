#include<sys/types.h>
#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<sys/wait.h>

int main(int argc, char *argv[])
{
	pid_t pid;
	int status; 
	/* fork another process */
    printf("PARENT starting\n");
	pid = fork();
	if (pid < 0) { /* error occurred */
		fprintf(stderr, "Fork Failed");
		exit(-1);
	}
	else if (pid == 0) { /* child process */
		execlp("/mnt/c/Users/yadav/OneDrive/Documents/manipal/ass/lab/week-5/parent", "parent", NULL);
	}
	else { /* parent process */
		/* parent will wait for the child to complete */
		wait (&status);
		printf ("Child Complete\n");
		exit(0);
	}
}