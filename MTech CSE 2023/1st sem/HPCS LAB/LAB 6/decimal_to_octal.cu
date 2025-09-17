#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void decimal_to_octal(int *decimal, int *octal, int size)
{
   int tid = threadIdx.x;
   if (tid < size)
   {
    int decimal_num=decimal[tid];
    int octal_num[100],octalNumber;
    int i=0;
    while (decimal_num >0)
        {
         octal_num[i++]=decimal_num % 8;
         decimal_num = decimal_num /8;  
         
        }
    for (int j=i-1;j>=0;j--)
      	octalNumber=octalNumber*10 +octal_num[j];
    octal[tid]=octalNumber;
      
}
}

int main()
{
    int n;
    printf("\nEnter the size of an array:");
    scanf("%d",&n);
    int h_decimal[n],h_octal[n];
    int *d_decimal, *d_octal;
    printf ("\n Enter array decimal numbers");
     for (int i =0; i<n; i++)
            scanf("%d",&h_decimal[i]);
    cudaMalloc((void **)&d_decimal,n*sizeof(int));
    cudaMalloc((void **)&d_octal,n*sizeof(int));
    cudaMemcpy(d_decimal,&h_decimal,n*sizeof(int), cudaMemcpyHostToDevice);

    decimal_to_octal <<< 1,n>>> (d_decimal,d_octal,n);
    cudaMemcpy(h_octal,d_octal,n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n Octal Numbers are\n");
    for(int i=0;i<n;i++)
        printf("%d ",h_octal[i]);
    cudaFree(d_decimal);
    cudaFree(d_octal);
    return 0;


}
