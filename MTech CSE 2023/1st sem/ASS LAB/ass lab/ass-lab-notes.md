
## Week 6

```c
// create a thread
pthread_t t1;

pthread_create( pthread_t, 0 /* attribute, keep it 0 */ , func_name, (void *)data );

// thread array
pthread_t thread[end - start];
pthread_create(&thread[i], 0, &child_thread, (void *)i);

// join without data
pthread_join(thread[i], NULL);
// join with data
pthread_join(pthread_t, (void **)&data);

int status; // global
wait(&status);

// thread function
void *func_name(void *param)

// extract data from param
int option = (int)param;

// return data from thread function
return (void *)sum;
```

## Week 5 forking 

```c
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

pid_t pid = fork();
pid_t myId = getpid();
pid_t parentId = getppid();

// to execute
execlp("/mnt/c/Users/yadav/parent", "parent", NULL);

// fork create 2 process
// parent process has child id
// pid from fork() in child is 0
// remember to wait
// orphan process (parent dies before child – adopted by “init” process
// zombie (defunct) child process (a child with exit() call, but no corresponding wait() in the sleeping parent
```

## Week 4 struct stat

```c
struct stat statbuf;

	printf("Id of device : %ld\n", statbuf.st_dev );
	printf("Inode number : %ld\n", statbuf.st_ino );
	printf("Permission   : %d\n", statbuf.st_mode );
	printf("N(Hardlinks) : %ld\n", statbuf.st_nlink );
	printf("UserId Owner : %d\n", statbuf.st_uid );
	printf("GroupId Owner: %d\n", statbuf.st_gid );
	printf("DeviceId     : %ld\n", statbuf.st_rdev );
	printf("size (bytes) : %ld\n", statbuf.st_size );
	printf("block size   : %ld\n", statbuf.st_blksize );
	printf("block alloc  : %ld\n", statbuf.st_blocks );
	printf("time         : %ld\n", statbuf.st_atime );
	printf("time         : %ld\n", statbuf.st_ctime );
	printf("time         : %ld\n", statbuf.st_mtime );

// directory traversal
DIR *dir = opendir(".");
struct dirent *entry;
while ((entry = readdir(dir)) != NULL)
{
    printf("%s\n", entry->d_name);
}



```

## Week 3 : file handling

```c
char *fgets(char *s, int size, FILE *stream);
```

## Week 2

```bash
for((i=2;i<=num;i++))
{
  fact=$((fact*i))
}

# for string calculations
| bc

echo $"(-$b-sqrt($b^2-4*$a*$c))/(2*$a)" | bc 


#!/bin/bash

# touch a b c B

if [ $# -gt 0 ]
then
  for file in "$@"
  do
    if [ -f "$file" ]
    then
      upper_file=$(echo "$file" | tr '[:lower:]' '[:upper:]')
      if [ ! -f "$upper_file" ]
      then
        mv "$file" "$upper_file"
      else
        echo "Error: $upper_file already exists"
      fi
    else
      echo "Error: $file does not exist"
    fi
  done
else
  echo "Error: No file names provided"
fi
```