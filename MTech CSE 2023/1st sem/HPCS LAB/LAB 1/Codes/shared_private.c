#include <stdio.h>
#include <omp.h>

int main() {

	int sharedVar = 10;
	int privateVar = 100;

	omp_set_num_threads(4);

	#pragma omp parallel shared(sharedVar) private(privateVar)
	{
		int privateVar = 0;
		int tid = omp_get_thread_num();
		privateVar += tid;
		sharedVar += tid;
		printf("private : %d | sharedVar : %d | tid : %d \n", privateVar, sharedVar, tid );
	}

	printf("private : %d | sharedVar : %d\n", privateVar, sharedVar );

}
