#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include "common.h"
#include <iomanip>
using namespace std;

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}


float * Split_GEBP_Mul(
        int a1Rows, int a1Cols,  float * A_c_handle,
        int a2Rows, int a2Cols,  float * B_c_handle,
        int num_split, float* C_c_handle, long mem_free)
{
        int m = a1Rows;
        int k = a1Cols;
        int n = a2Cols;
        if ((long)m*k*n==0)
        {
          cout<<"ERROR: the argument of Rows and Cols can not be 0";
        }

        if (num_split==0)
        {
          if (mem_free*0.8>(long)820*2*1024*1024)
          {
            num_split = (long)a2Rows*a2Cols*sizeof(float)/1024/1024/820;
          }
          else{
            num_split = (long)2*a2Rows*a2Cols*sizeof(float)/(0.8*(mem_free-sizeof(float)*a1Rows*a1Cols));
          }
          num_split = (num_split>1) ? num_split : 1;
        }

        int col = n/num_split;
        int left= n-col*num_split;
        float* x_input = 0;
        float* w_split1 = 0;
        float* w_split2 = 0;
        float* dvs_output1 = 0;
        float* dvs_output2 = 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t  handle;
        cublasStatus_t stat;
        cudaStream_t streams[3];
        cudaEvent_t startEvent, stopEvent;
        cudaEvent_t tb1,tb2,gemm1,gemm2,out1,out2;
        float time;
        float total_time=0;
        //clock_t t1, t2;
        checkCublas(cublasCreate (&handle));
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );
        checkCuda( cudaEventCreate(&tb1) );
        checkCuda( cudaEventCreate(&tb2) );
        checkCuda( cudaEventCreate(&gemm1) );
        checkCuda( cudaEventCreate(&gemm2) );
        checkCuda( cudaEventCreate(&out1) );
        checkCuda( cudaEventCreate(&out2) );
        checkCuda(cudaStreamCreate(&streams[0]));
        checkCuda(cudaStreamCreate(&streams[1]));
        checkCuda(cudaStreamCreate(&streams[2]));

        checkCuda(cudaMalloc((void**) &x_input, sizeof(float)*m*k));
        checkCuda(cudaMalloc((void**) &w_split1, sizeof(float) * col * k));
        checkCuda(cudaMalloc((void**) &w_split2, sizeof(float) * col * k));
        checkCuda(cudaMalloc((void**) &dvs_output1,  sizeof(float) * col * m));
        checkCuda(cudaMalloc((void**) &dvs_output2,  sizeof(float) * col * m));

        checkCuda( cudaEventRecord(startEvent, 0) );
        clock_t t_start = clock();
        checkCuda(cudaMemcpyAsync(x_input, A_c_handle, sizeof(float)*m*k, cudaMemcpyHostToDevice,streams[1]));
        int i = 0;
        for (i = 0; i < num_split/2; i++)
        {
          if (i==0)
          {
            for(int j = 0; j < k; j++){
            checkCuda(cudaMemcpyAsync(w_split1+j*col, B_c_handle+(long)j*n+i*2*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
            }
            checkCublas(cublasSetStream(handle, streams[1]));
            stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
              col, m, k,
              &alpha,
              w_split1, col,
              x_input, k,
              &beta,
              dvs_output1, col );
            checkCuda( cudaEventRecord(gemm1, streams[1]) );
            if(stat != CUBLAS_STATUS_SUCCESS){
                cerr << "cublasSgemmBatched failed" << endl;
                exit(1);
              }
          }
          for(int j = 0; j < k; j++){
              checkCuda(cudaMemcpyAsync(w_split2+j*col, B_c_handle+(long)j*n+(i*2+1)*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[2]));
               }
          if (i>0){ checkCuda( cudaEventSynchronize(out2) ); }
          checkCublas(cublasSetStream(handle, streams[2]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
            col, m, k,
            &alpha,
            w_split2, col,
            x_input, k,
            &beta,
            dvs_output2, col );
          checkCuda( cudaEventRecord(gemm2, streams[2]) );
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          checkCuda( cudaEventSynchronize(gemm1) );
          for(int j = 0; j < m; j++){
              checkCuda(cudaMemcpyAsync(C_c_handle+(long)j*n+2*(i)*col, dvs_output1+j*col, sizeof(float)*col, cudaMemcpyDeviceToHost,streams[0]));
          }
          checkCuda( cudaEventRecord(out1, streams[0]));
          if (i<(num_split/2-1))
              {
                  for(int j = 0; j < k; j++){
                  checkCuda(cudaMemcpyAsync(w_split1+j*col, B_c_handle+(long)j*n+(i+1)*2*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
                  }
                  checkCublas(cublasSetStream(handle, streams[1]));
                  checkCuda( cudaEventSynchronize(out1) );
                  stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                    col, m, k,
                    &alpha,
                    w_split1, col,
                    x_input, k,
                    &beta,
                    dvs_output1, col );
                  checkCuda( cudaEventRecord(gemm1, streams[1]) );
                  if(stat != CUBLAS_STATUS_SUCCESS){
                      cerr << "cublasSgemmBatched failed" << endl;
                      exit(1);
                  }
              }
          checkCuda( cudaEventSynchronize(gemm2) );
          for(int j = 0; j < m; j++){
              checkCuda(cudaMemcpyAsync(C_c_handle+(long)j*n+(2*i+1)*col, dvs_output2+j*col, sizeof(float)*col, cudaMemcpyDeviceToHost,streams[0]));
              }
          checkCuda( cudaEventRecord(out2, streams[0]));
        }
        int l=2*i;
        if (l<num_split)
        {
          for(int j = 0; j < k; j++){
            checkCuda(cudaMemcpyAsync(w_split1+j*col, B_c_handle+(long)j*n+i*2*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          }
          checkCuda( cudaEventSynchronize(out1) );
          checkCublas(cublasSetStream(handle, streams[1]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
              col, m, k,
              &alpha,
              w_split1, col,
              x_input, k,
              &beta,
              dvs_output1, col );
          checkCuda( cudaEventRecord(gemm1, streams[1]) );
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          l++;
        }
        if (left!=0)
        {
          if(col<left) {cout<<"num of split is too large"<<endl; return 0;}
          for(int j = 0; j < k; j++){
              checkCuda(cudaMemcpyAsync(w_split2+j*left, B_c_handle+(long)j*n+l*col, left*sizeof(float), cudaMemcpyHostToDevice,streams[2]));
               }
          checkCuda( cudaEventSynchronize(out2) );
          checkCublas(cublasSetStream(handle, streams[2]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
            left, m, k,
            &alpha,
            w_split2, left,
            x_input, k,
            &beta,
            dvs_output2, left );
          checkCuda( cudaEventRecord(gemm2, streams[2]) );
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
        }
        if (2*i<num_split)
        {
          checkCuda( cudaEventSynchronize(gemm1) );
          for(int j = 0; j < m; j++){
              checkCuda(cudaMemcpyAsync(C_c_handle+(long)j*n+2*i*col, dvs_output1+j*col, sizeof(float)*col, cudaMemcpyDeviceToHost,streams[0]));
          }
        }
        if (left!=0)
        {
          checkCuda( cudaEventSynchronize(gemm2) );
          for(int j = 0; j < m; j++){
              checkCuda(cudaMemcpyAsync(C_c_handle+(long)j*n+l*col, dvs_output2+j*left, sizeof(float)*left, cudaMemcpyDeviceToHost,streams[0]));
          }
        }
        checkCuda( cudaDeviceSynchronize());
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        //clock_t t_end = clock();
        checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
        total_time = time/1000;
        cout << "GPU time: " << total_time << " s "<< endl;
        //total_time = (double)(t_end-t_start) / CLOCKS_PER_SEC;
        //cout << "CPU time: " << total_time << " s "<< endl;
        checkCuda(cudaFree(x_input));
        checkCuda(cudaFree(w_split1));
        checkCuda(cudaFree(w_split2));
        checkCuda(cudaFree(dvs_output1));
        checkCuda(cudaFree(dvs_output2));
        checkCuda(cudaStreamDestroy(streams[0]));
        checkCuda(cudaStreamDestroy(streams[1]));
        checkCuda(cudaStreamDestroy(streams[2]));
        cublasDestroy ( handle ) ;
        return 0;
}



float * Split_GEPDOT(
        int a1Rows, int a1Cols,  float * A_c_handle,
        int a2Rows, int a2Cols,  float * B_c_handle,
        int num_split, float* C_c_handle, long mem_free)
{
        int m = a1Rows;
        int k = a1Cols;
        int n = a2Cols;
        if ((long)m*k*n==0)
        {
          cout<<"ERROR: the argument of Rows and Cols can not be 0";
        }
        if (num_split==0)
        {
          if (mem_free*0.8>(long)1000*2*1024*1024)
          {
            int s1 = (long)a1Rows*a1Cols*sizeof(float)/1024/1024/1000;
            int s2 = (long)a2Rows*a2Cols*sizeof(float)/1024/1024/1000;
            num_split = (s1>s2)?s1:s2;
          }
          else
          {
            int s1 = (long)a1Rows*a1Cols*sizeof(float)/(0.4*mem_free);
            int s2 = (long)a2Rows*a2Cols*sizeof(float)/(0.4*mem_free);
            num_split = (s1>s2)?s1:s2;
          }
          num_split = (num_split>1) ? num_split : 1;
        }
	//cout <<"num_split: "<< num_split << endl;
        int col = k/num_split;
        int left= k-col*num_split;
        float* x_input1 = 0;
        float* w_split1 = 0;
        float* dvs_output1 = 0;
        //float* dvs_output2 = 0;
        float* dvs_output_total = 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t  handle;
        cublasStatus_t stat;
        cudaStream_t streams[3];
        cudaEvent_t startEvent, stopEvent;
        float time;
        //float total_time=0;
        checkCublas(cublasCreate (&handle));
        checkCuda( cudaEventCreate(&startEvent));
        checkCuda( cudaEventCreate(&stopEvent) );

        checkCuda(cudaMalloc((void**) &x_input1, sizeof(float)*m*col));
        checkCuda(cudaMalloc((void**) &w_split1, sizeof(float) * col* n));
        checkCuda(cudaMalloc((void**) &dvs_output1,  sizeof(float) * n * m));
        checkCuda(cudaMalloc((void**) &dvs_output_total,  sizeof(float) * n * m));
        checkCuda( cudaEventRecord(startEvent, 0) );
        stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, &beta,
              dvs_output_total, n,
              &beta,
              dvs_output1, n,
              dvs_output_total, n);
        if(stat != CUBLAS_STATUS_SUCCESS){
          cerr << "cublasSgemmBatched failed" << endl;
          exit(1);
        }
        int i;
        for (i = 0; i < num_split; i++)
        {
          for(int j = 0; j < m; j++){
            checkCuda(cudaMemcpyAsync(x_input1+j*col, A_c_handle+(long)j*k+i*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          }
          checkCuda(cudaMemcpyAsync(w_split1, B_c_handle+(long)n*col*i, n*col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, col,
                &alpha,
                w_split1, n,
                x_input1, col,
                &beta,
                dvs_output1, n );
          if(stat != CUBLAS_STATUS_SUCCESS){
            cerr << "cublasSgemmBatched failed" << endl;
            exit(1);
          }
          stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, &alpha,
              dvs_output_total, n,
              &alpha,
              dvs_output1, n,
              dvs_output_total, n);
          if(stat != CUBLAS_STATUS_SUCCESS){
                  cerr << "cublasSgemmBatched failed" << endl;
                  exit(1);
          }
        }
        if (left!=0)
        {
          for(int j = 0; j < m; j++){
            checkCuda(cudaMemcpyAsync(x_input1+j*left, A_c_handle+(long)j*k+i*col, left*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          }
          checkCuda(cudaMemcpyAsync(w_split1, B_c_handle+(long)n*col*i, n*left*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, left,
                &alpha,
                w_split1, n,
                x_input1, left,
                &beta,
                dvs_output1, n );
          if(stat != CUBLAS_STATUS_SUCCESS){
            cerr << "cublasSgemmBatched failed" << endl;
            exit(1);
          }
          stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, &alpha,
              dvs_output_total, n,
              &alpha,
              dvs_output1, n,
              dvs_output_total, n);
          if(stat != CUBLAS_STATUS_SUCCESS){
                  cerr << "cublasSgemmBatched failed" << endl;
                  exit(1);
          }
      }
      checkCuda( cudaDeviceSynchronize());
      checkCuda( cudaMemcpyAsync(C_c_handle, dvs_output_total, sizeof(float)*m*n, cudaMemcpyDeviceToHost,0));
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      cout  << "GPU time: " << time/1000 << " s " << endl;
      checkCuda(cudaFree(x_input1));
      checkCuda(cudaFree(w_split1));
      checkCuda(cudaFree(dvs_output1));
      checkCuda(cudaFree(dvs_output_total));
      cublasDestroy (handle);
      return 0;
}


float * Split_GEPDOT_Mul(
        int a1Rows, int a1Cols,  float * A_c_handle,
        int a2Rows, int a2Cols,  float * B_c_handle,
        int num_split, float* C_c_handle, long mem_free)
{
        int m = a1Rows;
        int k = a1Cols;
        int n = a2Cols;
        if ((long)m*k*n==0)
        {
          cout<<"ERROR: the argument of Rows and Cols can not be 0";
        }
        if (num_split==0)
        {
          if (mem_free*0.8>(long)800*2*1024*1024)
          {
            int s1 = (long)a1Rows*a1Cols*sizeof(float)/1024/1024/800;
            int s2 = (long)a2Rows*a2Cols*sizeof(float)/1024/1024/800;
            num_split = (s1>s2)?s1:s2;
          }
          else
          {
            int s1 = (long)a1Rows*a1Cols*sizeof(float)/(0.4*mem_free);
            int s2 = (long)a2Rows*a2Cols*sizeof(float)/(0.4*mem_free);
            num_split = (s1>s2)?s1:s2;
          }
          num_split = (num_split>1) ? num_split : 1;
        }
	//cout <<"num_split: "<< num_split << endl;
        int col = k/num_split;
        int left= k-col*num_split;
        float* x_input1 = 0;
        float* x_input2 = 0;
        float* w_split1 = 0;
        float* w_split2 = 0;
        float* dvs_output1 = 0;
        float* dvs_output2 = 0;
        float* dvs_output_total = 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t  handle;
        cublasStatus_t stat;
        cudaStream_t streams[3];
        cudaEvent_t startEvent, stopEvent;
        float time;
        //float total_time=0;
        checkCublas(cublasCreate (&handle));
        checkCuda(cudaEventCreate(&startEvent));
        checkCuda(cudaEventCreate(&stopEvent) );
        checkCuda(cudaStreamCreate(&streams[0]));
        checkCuda(cudaStreamCreate(&streams[1]));
        checkCuda(cudaStreamCreate(&streams[2]));
        checkCuda(cudaMalloc((void**) &x_input1, sizeof(float)*m*col));
        checkCuda(cudaMalloc((void**) &x_input2, sizeof(float)*col*m));
        checkCuda(cudaMalloc((void**) &w_split1, sizeof(float) * col * n));
        checkCuda(cudaMalloc((void**) &w_split2, sizeof(float) * col * n));
        checkCuda(cudaMalloc((void**) &dvs_output1,  sizeof(float) * n * m));
        checkCuda(cudaMalloc((void**) &dvs_output2,  sizeof(float) * n * m));
        checkCuda(cudaMalloc((void**) &dvs_output_total,  sizeof(float) * n * m));
        checkCuda( cudaEventRecord(startEvent, 0) );
        stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, &beta,
              dvs_output_total, n,
              &beta,
              dvs_output1, n,
              dvs_output_total, n);
        if(stat != CUBLAS_STATUS_SUCCESS){
                cerr << "cublasSgemmBatched failed" << endl;
                exit(1);
        }
        int i;
        for (i = 0; i < num_split/2; i++)
        {
          for(int j = 0; j < m; j++){
            checkCuda(cudaMemcpyAsync(x_input1+j*col, A_c_handle+(long)j*k+i*2*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          }
          checkCuda(cudaMemcpyAsync(w_split1, B_c_handle+(long)n*col*i*2, n*col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          checkCublas(cublasSetStream(handle, streams[1]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, col,
            &alpha,
            w_split1, n,
            x_input1, col,
            &beta,
            dvs_output1, n );
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, &alpha,
          dvs_output_total, n,
          &alpha,
          dvs_output1, n,
          dvs_output_total, n);
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          checkCublas(cublasSetStream(handle, streams[2]));
          for(int j = 0; j < m; j++){
            checkCuda(cudaMemcpyAsync(x_input2+j*col, A_c_handle+(long)j*k+(i*2+1)*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[2]));
          }
          checkCuda(cudaMemcpyAsync(w_split2, B_c_handle+(long)n*col*(i*2+1), n*col*sizeof(float), cudaMemcpyHostToDevice,streams[2]));
	        stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, col,
                &alpha,
                w_split2, n,
                x_input2, col,
                &beta,
                dvs_output2, n );
          if(stat != CUBLAS_STATUS_SUCCESS){
                  cerr << "cublasSgemmBatched failed" << endl;
                  exit(1);
          }
          stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, &alpha,
                dvs_output_total, n,
                &alpha,
                dvs_output2, n,
                dvs_output_total, n);
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
        }
        int l=2*i;
        if (l<num_split)
        {
          for(int j = 0; j < m; j++){
            checkCuda(cudaMemcpyAsync(x_input1+j*col, A_c_handle+(long)j*k+i*2*col, col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          }
          checkCuda(cudaMemcpyAsync(w_split1, B_c_handle+(long)n*col*i*2, n*col*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          checkCublas(cublasSetStream(handle, streams[1]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, col,
            &alpha,
            w_split1, n,
            x_input1, col,
            &beta,
            dvs_output1, n );
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, &alpha,
          dvs_output_total, n,
          &alpha,
          dvs_output1, n,
          dvs_output_total, n);
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          l++;
        }
        if (left!=0)
        {
          if(col<left) {cout<<"num of split is too large"<<endl; return 0;}
          for(int j = 0; j < m; j++){
            checkCuda(cudaMemcpyAsync(x_input1+j*left, A_c_handle+(long)j*k+l*col, left*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          }
          checkCuda(cudaMemcpyAsync(w_split1, B_c_handle+(long)n*col*l, n*left*sizeof(float), cudaMemcpyHostToDevice,streams[1]));
          checkCublas(cublasSetStream(handle, streams[1]));
          stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, left,
            &alpha,
            w_split1, n,
            x_input1, left,
            &beta,
            dvs_output1, n );
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
          stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, &alpha,
          dvs_output_total, n,
          &alpha,
          dvs_output1, n,
          dvs_output_total, n);
          if(stat != CUBLAS_STATUS_SUCCESS){
              cerr << "cublasSgemmBatched failed" << endl;
              exit(1);
          }
        }
        checkCuda( cudaDeviceSynchronize());
        checkCuda( cudaMemcpyAsync(C_c_handle, dvs_output_total, sizeof(float)*m*n, cudaMemcpyDeviceToHost,0));
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
        cout  << "GPU time: " << time/1000 << " s " << endl;
        checkCuda(cudaFree(x_input1));
	      checkCuda(cudaFree(x_input2));
        checkCuda(cudaFree(w_split1));
        checkCuda(cudaFree(w_split2));
        checkCuda(cudaFree(dvs_output1));
        checkCuda(cudaFree(dvs_output2));
        checkCuda(cudaStreamDestroy(streams[0]));
        checkCuda(cudaStreamDestroy(streams[1]));
        checkCuda(cudaStreamDestroy(streams[2]));
        cublasDestroy ( handle ) ;
        return 0;
}

