#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
    pid_t child_pid = fork();

    if (child_pid < 0) {
        perror("Fork failed");
        exit(1);
    } else if (child_pid == 0) {
        // This is the child process
        printf("Child PID: %d\n", getpid());
        exit(0); // Child exits without waiting
    } else {
        // This is the parent process
        printf("Parent PID: %d, Child PID: %d\n", getpid(), child_pid);
        // Parent sleeps briefly, allowing the child to become a zombie
        sleep(10);
    }

    return 0;
}
