#ifndef _COMMON_v1_H_
#define _COMMON_v1_H_

#include <cuda_runtime.h>
#include "cublas_v2.h"


const char* cublasGetErrorString(cublasStatus_t status);

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result);
cublasStatus_t checkCublas(cublasStatus_t result);

float * Split_GEBP_Mul(
        int a1Rows, int a1Cols,  float * A_c_handle,
        int a2Rows, int a2Cols,  float * B_c_handle,
        int num_split, float* C_c_handle, long mem_free);
float * Split_GEPDOT(
        int a1Rows, int a1Cols,  float * A_c_handle,
        int a2Rows, int a2Cols,  float * B_c_handle,
        int num_split, float* C_c_handle, long mem_free);
float * Split_GEPDOT_Mul(
        int a1Rows, int a1Cols,  float * A_c_handle,
        int a2Rows, int a2Cols,  float * B_c_handle,
        int num_split, float* C_c_handle, long mem_free);
#endif


