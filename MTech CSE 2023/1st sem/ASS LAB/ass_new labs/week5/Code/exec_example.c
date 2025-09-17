#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
    pid_t pid;

    printf("Parent process started\n");

    pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(1);
    } else if (pid == 0) {
        // Child process
        printf("Child process started\n");
        execl("./wait_example", "wait_example", NULL);
        perror("Exec failed");
        exit(1);
    } else {
        // Parent process
        wait(NULL);
        printf("Parent process completed\n");
    }

    return 0;
}
