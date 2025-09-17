#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void toggle_string(char *s, int size)
{
   int tid = threadIdx.x;
   if (tid < size)
   {
   if (s[tid] >= 'A' && s[tid] <= 'Z')
            s[tid] = s[tid] + 'a' - 'A';
    else if (s[tid] >= 'a' && s[tid] <= 'z')
            s[tid] = s[tid] + 'A' - 'a';

   }
}

int main()
{
    int n;
    char h_s[100],h_r[100];
    char *d_s ;
    printf("\nEnter the string\n");
    scanf("%s",h_s);
    n = strlen(h_s);
    cudaMalloc((void **)&d_s,n*sizeof(char));
    cudaMemcpy(d_s,h_s,n*sizeof(char), cudaMemcpyHostToDevice);
    toggle_string <<<1,n>>> (d_s,n);
    cudaMemcpy(h_r,d_s,n*sizeof(char), cudaMemcpyDeviceToHost);
    printf("\n new toggled string is \n");
    for(int i=0;i<n;i++)
       printf("%c",h_r[i]);
    cudaFree(d_s);


    return 0;


}
