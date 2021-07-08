#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include "common.h"
#include "support.h"
using namespace std;

int main(int argc, char const *argv[])
{
	//printf("argc : %d \n",argc);
	if (argc == 5)
	{
        //int repeats = 1;
        int num_split = 0;
        int m = 1024;
        int k = 1024;
        int n = 1024;
		long mem_free = (long)2*1024*1024*1024;
	    m = atoi(argv[1]);
        k = atoi(argv[2]);
        n = atoi(argv[3]);
		
		float *A_c_handle, *B_c_handle, *C_c_handle;
		checkCuda(cudaMallocHost((void**)&A_c_handle, sizeof(float)*m*k ));
		checkCuda(cudaMallocHost((void**)&B_c_handle, sizeof(float)*n*k ));
		checkCuda(cudaMallocHost((void**)&C_c_handle, sizeof(float)*m*n ));

		if (argv[4][2]=='p'&&argv[4][6]=='_')
		{
			printf("gepdot_mul\n");
			printf("m: %d k: %d n: %d\n",m,k,n);
			Split_GEPDOT_Mul(m,k,A_c_handle,k,n,B_c_handle,num_split,C_c_handle,mem_free);
			//verify(A_c_handle, B_c_handle, C_c_handle, m,k,n);
		}
		else if (argv[4][2]=='p'&&argv[4][6]=='\0')
		{
			printf("gepdot\n");
			printf("m: %d k: %d n: %d\n",m,k,n);
			Split_GEPDOT(m,k,A_c_handle,k,n,B_c_handle,num_split,C_c_handle,mem_free);
			//verify(A_c_handle, B_c_handle, C_c_handle, m,k,n);
		}
		else if (argv[4][2]=='b')
		{
			printf("gebp_mul\n");
			printf("m: %d k: %d n: %d\n",m,k,n);
			Split_GEBP_Mul(m,k,A_c_handle,k,n,B_c_handle,num_split,C_c_handle,mem_free);
			//verify(A_c_handle, B_c_handle, C_c_handle, m,k,n);
		}
		else 
		{
			 printf("Invalid input parameters\n");
			 exit(0);
		}
		
	}
	else 
	{
		 printf("Invalid input parameters\n");
		 exit(0);
	}
}                                                                                 
