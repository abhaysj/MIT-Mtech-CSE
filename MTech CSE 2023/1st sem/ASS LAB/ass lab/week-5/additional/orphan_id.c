#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

int main()
{
  printf("PARENT : %d\n", getppid());
  pid_t child_pid;

  child_pid = fork();
  if (child_pid == 0)
  {
    printf("before parent die %d\n", getpid());
    sleep(100);
    printf("after parent die %d\n", getpid());
  }
  else
  {
    sleep(2);    
    exit(0);
  }
  return 0;
}