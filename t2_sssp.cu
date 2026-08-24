#include <iostream>
#include <vectors>
#include <cuda_runtime.h>
#include <cmath>

#define INF 1e9

__global__ void sssp(int *row_ptr, int *col_ind, int *weights, int *dist, int *changed, int V){
	int u = blockIdx.x * blockDim.x + threadIdx.x;

	if (u>=V || dist[u] == INF) return;

	int start = row_ptr[u];
	int end = row_ptr[u+1];

	for (int i= start; i<end; i++){
		int v = col_ind[i];
		int w = weights[i];
		int new_dist = dist[u] + w;

		if (new_dist < dist[v]){
			int old_dist = atomicMin(&dist[v], new_dist);
			if (new_dist < old_dist) *changed = 1;
		}
	}
}

int main(){
	int V = 5;
	int source = 0;

	std::vector<std::vector<std::pair<int, int>>>adj(V);
	adj[0] = {{1, 4}, {2, 2}};
	adj[1] = {{2, 1}, {3, 5}};
	adj[2] = {{3, 8}, {4, 10}};
	adj[3] = {{4, 2}};

	std::vector<int> h_row_ptr(V+1, 0);
	std::vector<int> h_col_ind;
	std::vector<int> h_weights;

	for (int i=0; i<V; i++){
		h_row_ptr[i] = h_col_ind.size();
		for (auto &e : adj[i]){
			h_col_ind.push_back(e.first);
			h_weights.push_back(e.second);
		}
	}

	 h_row_ptr[V] = h_col_ind.size();
 	 int E =  h_col_ind.size(); 

	std::vector<int> h_dist(V, INF);
	h_dist[source] = 0;
	
	int *d_row_ptr, *d_col_ind, *d_weights, *d_dist, *d_changed;

  	cudaMalloc((void**)&d_row_ptr, (V+1)*sizeof(int));
	cudaMalloc((void**)&d_col_ind, E*sizeof(int));
	cudaMalloc((void**)&d_weights, E*sizeof(int));
	cudaMalloc((void**)&d_dist, V*sizeof(int));
	cudaMalloc((void**)&d_changed, sizeof(int));

	cudaMemcpy(d_row_ptr, h_row_ptr.data(), (V+1)*sizeof(int), cudaMemcpyHostToDevice);
 	cudaMemcpy(d_col_ind, h_col_ind.data(), E*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, h_weights.data(), E*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, h_dist.data(), V*sizeof(int), cudaMemcpyHostToDevice);


	int threads_per_block = 256;
	int blocks = (V+threads_per_block-1)/threads_per_block;

	int h_changed = 1;

	while (h_changed){
    		h_changed = 0;
    		cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

    		sssp<<<blocks, threads_per_block>>>(d_row_ptr, d_col_ind, d_weights, d_dist, d_changed, V);

    		cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    		cudaDeviceSynchronize();
  	}

  	cudaMemcpy(h_dist.data(), d_dist, V*sizeof(int), cudaMemcpyDeviceToHost);

  	std::cout << "Shortest distances from source " << source << ":\n";
  	for (int i = 0; i < V; i++) {
      		std::cout << "Node " << i << ": " << (h_dist[i] == INF ? -1 : h_dist[i]) << "\n";
  	}

  	cudaFree(d_row_ptr);
  	cudaFree(d_col_ind);
  	cudaFree(d_weights);
  	cudaFree(d_dist);
  	cudaFree(d_changed);

  return 0;
}

