#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    char str[] = "0000";
    int length = strlen(str);

    for (int j = 0; j < 15; j++)
    {
        int i = length - 1;
        while (str[i] == '1')
        {
            str[i] = '0';
            i--;
        }
        if (i >= 0)
            str[i] = '1';

        printf("%s", str);
        printf("\n");
    }
}
