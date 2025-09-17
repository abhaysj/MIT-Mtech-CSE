#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

// int read(int fildes, void *buf, size_t nbytes);
// return 0 : nothing to read
// return 1 : more to read
// return -1 : error in reading

int main()
{
	char buffer[128];
	int nread;
	nread = read(0, buffer, 128);
	if (nread == -1)
	{
		write(2, "A read error has occurred\n", 26);
		if ((write(1, buffer, nread)) != nread)
		{
			write(2, "A write error has occurred\n", 27);
			exit(0);
		}
	}
	else
	{
		write(2, buffer, 26);
	}
}