#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

int main(int argc, char * argv[]) {
printf("Enter the path : \n");
char oldLink[100];
scanf("%s", oldLink);
char newLink[100] = "";
strcat(newLink, oldLink);
strcat(newLink,"-new");
int status = link(oldLink,newLink);
if(status) {
if (errno == EEXIST) {
printf("link already exists\n");
}
}
printf("new link : %s\n", newLink);
unlink(newLink);
printf("new link deleted \n");
}


