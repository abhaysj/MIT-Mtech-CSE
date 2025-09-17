#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

// size_t write(int fildes, const void *buf, size_t nbytes);

int main()
{
	if ((write(1, "Here is some data\n", 18)) != 18)
	{
		write(2, "A write error has occurred on file descriptor 1\n", 46);
		// write ( 1:standard-output, data, nbytes of data ) : returns byte of data
	}

	ssize_t len = write(1, "uday", 4);
	printf("\nlength : %ld\n", len);

	exit(0);
}