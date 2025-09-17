#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
    printf("Parent PID: %d\n", getpid());

    pid_t child_pid = fork();

    if (child_pid == 0) {
        // This is the child process
        printf("Child PID: %d, Parent PID: %d\n", getpid(), getppid());
    }

    return 0;
}
