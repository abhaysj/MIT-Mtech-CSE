#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

__global__ void toggle_string(char *s, char *r, int size)
{
   int tid = threadIdx.x;
   if (tid < size)
   {
   char ch =s[tid];
   int a = (int)ch;

   int num = 0;
   while (a > 0) {
           int digit = a % 10;
           num = num * 10 + digit;
           a = a/10;
       }
    r[tid]=(char)num;

   }
}

int main()
{
    int n;
    char h_s[100],h_r[100];
    char *d_s, *d_r ;
    printf("\nEnter the string\n");
    scanf("%s",h_s);
    n = strlen(h_s);
    cudaMalloc((void **)&d_s,n*sizeof(char));
    cudaMalloc((void **)&d_r,n*sizeof(char));
    cudaMemcpy(d_s,h_s,n*sizeof(char), cudaMemcpyHostToDevice);
    toggle_string <<< 1,n>>> (d_s,d_r,n);
    cudaMemcpy(h_r,d_r,n*sizeof(char), cudaMemcpyDeviceToHost);
    printf("\n new reverse ACII string is \n");
      for(int i=0;i<n;i++)
       printf("%c",h_r[i]);
    cudaFree(d_s);
    cudaFree(d_r);
  

    return 0;


}
