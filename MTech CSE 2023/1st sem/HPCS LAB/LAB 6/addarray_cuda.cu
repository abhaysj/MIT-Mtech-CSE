#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void addArray(int *a, int *b,int *c, int size)
{
   int tid = threadIdx.x;
   if (tid < size)
   {
    c[tid] = a[tid] + b[tid];
   }
}

int main()
{
    int n;
    printf("\nEnter the size of an array:");
    scanf("%d",&n);
    int h_a[n],h_b[n],h_c[n];
    int *d_a, *d_b, *d_c;
    printf ("\n Enter array A elements");
        for (int i =0; i<n; i++)
            scanf("%d",&h_a[i]);
    printf ("\n Enter array B elements");
        for (int i =0; i<n; i++)
            scanf("%d",&h_b[i]);
    cudaMalloc((void **)&d_a,n*sizeof(int));
    cudaMalloc((void **)&d_b,n*sizeof(int));
    cudaMalloc((void **)&d_c,n*sizeof(int));
    cudaMemcpy(d_a,&h_a,n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&h_b,n*sizeof(int), cudaMemcpyHostToDevice);
    addArray <<< 1,n>>> (d_a,d_b,d_c,n);
    cudaMemcpy(&h_c,d_c,n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n sum of array elements\n");
    for(int i=0;i<n;i++)
        printf("%d ",h_c[i]);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;


}
