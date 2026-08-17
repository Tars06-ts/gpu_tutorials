#include <cstdio>
#include <iostream>

__device__ int val = 0;
__global__ void barrier(int *data, int n){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid<n){
		data[tid] = tid;
	}

  __syncthreads();
	if (threadIdx.x == 0){
    		atomicAdd(&val, 1);
	}

	while ( atomicAdd(&val, 0)< gridDim.x){}
  __syncthreads();

    if (tid<n and tid >=128){
		data[tid] = data[tid-128];
	}

}

int main(){
	int n = 1024;
	int *h_data = new int[n];
	int *d_data;

	cudaMalloc(&d_data, n* sizeof(int));

	int threadsPerBlock = 128;
	int block = (n + threadsPerBlock -1) / threadsPerBlock;

	barrier<<<block, threadsPerBlock>>>(d_data, n);
	cudaDeviceSynchronize();

	cudaMemcpy(h_data, d_data, n* sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<n; i+=128){
		std::cout << "data[" << i << "] = "<< h_data[i] << std::endl;
	}
	std::cout<<"\n"<< std::endl;

	cudaFree(d_data);
	delete[] h_data;
	return 0;
}
