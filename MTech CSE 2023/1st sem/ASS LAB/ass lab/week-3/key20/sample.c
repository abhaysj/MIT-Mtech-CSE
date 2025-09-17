#include<stdio.h>
#include<stdlib.h>
#include <fcntl.h>

int main(int argc, char *argv[]) {

    FILE *fp = fopen("numbers.txt","w+");

    for(int i=0; i<=100; i++) {
        fprintf(fp,"%d\n",i);
    }

    fclose(fp);

}