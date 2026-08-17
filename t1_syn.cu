#include <cstdio>
#include <iostream>

__device__ int val = 0;
__global__ void barrier(int *data, int n){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  	int sum = 0;
  	int sum2 = 0;

	if (tid<n){
		data[tid] = tid;
	}

  	__syncthreads();
	if (threadIdx.x == 0){
   		atomicAdd(&val, 1);
	}
  	while ( atomicAdd(&val, 0) < gridDim.x){}

  	__syncthreads();

  	if (tid == 0){
   		for (int i=0; i<n; i++){
      			sum+=i;
      			sum2 += data[i];
   		}
 
 		printf("sum = %d\n", sum);
  		printf("sum2 = %d\n", sum2);
  	}

}

int main(){
	int n = 1024;
	int *h_data = new int[n];
	int *d_data;
  
	cudaMalloc(&d_data, n* sizeof(int));

	int threadsPerBlock = 128;
	int block = 8;

	barrier<<<block, threadsPerBlock>>>(d_data, n);
	cudaDeviceSynchronize();

	cudaMemcpy(h_data, d_data, n* sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(d_data);
	delete[] h_data;
	return 0;
}

