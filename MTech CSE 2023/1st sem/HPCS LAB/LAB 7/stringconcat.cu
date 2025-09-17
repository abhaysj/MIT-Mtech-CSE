#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

__global__ void toggle_string(char *s, char *r, int size,int N)
{
   int tid = threadIdx.x;
   if (tid < N)
   {
    for(int i=0;i<size;i++)
    {
      r[tid*size+i]=s[i];
    }

   }
}

int main()
{
    int size;
    char h_s[100],h_r[100];
    char *d_s, *d_r ;
    printf("\nEnter the string\n");
    scanf("%s",h_s);
    size = strlen(h_s);
    int N;
    printf("\nEnter the number of times string needs to be repeated");
    scanf("%d",&N); 
    cudaMalloc((void **)&d_s,size*sizeof(char));
    cudaMalloc((void **)&d_r,N*size*sizeof(char));
    cudaMemcpy(d_s,h_s,size*sizeof(char), cudaMemcpyHostToDevice);
    toggle_string <<< 1,N>>> (d_s,d_r,size,N);
    cudaMemcpy(h_r,d_r,N*size*sizeof(char), cudaMemcpyDeviceToHost);
    printf("\nnew string\n");
    for(int i=0;i<N*size;i++)
       printf("%c",h_r[i]);
    cudaFree(d_s);
    cudaFree(d_r);
  

    return 0;


}
