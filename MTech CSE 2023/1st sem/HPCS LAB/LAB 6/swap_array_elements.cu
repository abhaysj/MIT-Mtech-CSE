#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void swap_array_elements(int *a,int size)
{
   int tid = threadIdx.x*2;
   if (tid < size-1)
   {
    int temp =a[tid];
    a[tid]=a[tid+1];
    a[tid+1]=temp;
   }
}

int main()
{
    int n;
    printf("\nEnter the size of an array:");
    scanf("%d",&n);
    int h_a[n],h_resultant[n];
    int *d_a;
    printf ("\n Enter array A elements\n");
        for (int i =0; i<n; i++)
            scanf("%d",&h_a[i]);
  
    cudaMalloc((void **)&d_a,n*sizeof(int));


    cudaMemcpy(d_a,&h_a,n*sizeof(int), cudaMemcpyHostToDevice);

    swap_array_elements <<< 1,n>>> (d_a,n);
    cudaMemcpy(h_resultant,d_a,n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n swapped new array elements\n");
    for(int i=0;i<n;i++)
        printf("%d ",h_resultant[i]);
    cudaFree(d_a);

    return 0;


}
