#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/wait.h>

int main()
{
  printf("PARENT : %d\n", getppid());
  int status;
  pid_t child_pid;
  

  child_pid = fork();
  if (child_pid == 0)
  {
    printf("before parent die %d\n", getpid());
    sleep(4);
    exit(0);
  }
  else
  {
    wait(&status);
    printf("child status %d", status);
    exit(0);
  }
  return 0;
}