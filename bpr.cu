/*
	A GPU implementation of Bayesian Personalized Ranking
	Created by Ashley Wang
*/


#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include "gputimer.h"


using namespace std;


#define NUM_USERS 2000
#define NUM_ITEMS 5000
#define NUM_SAMPLES 86670
#define RANK 128


struct triple {
	int uid;
	int iid_seen;
	int iid_unseen;
};

struct embedding {
	float vals[RANK];
};


void print_embedding(embedding *embed) {
	printf("{ ");
	for (int i = 0; i < RANK; i++)  { printf("%.3f ", embed->vals[i]); }
		printf("}\n");
}


__inline__ __device__
float dot(const embedding * a, const embedding * b) {
	float val = 0;
	for(int i=0; i<RANK; i++) {
		float a_val = a->vals[i], b_val = b->vals[i];
		val += a_val* b_val;
	}
	return val;
}


__global__ void bpr_update_kernel(triple * user_items, embedding * user_mat, embedding * prod_mat, float alpha, float lambda) {

	__shared__ embedding shared_memory;
	embedding * temp = &shared_memory;

	for (int i = blockIdx.x; i < NUM_SAMPLES; i += gridDim.x) {
		int uid = user_items[i].uid,
			iid_seen = user_items[i].iid_seen, 
			iid_unseen = user_items[i].iid_unseen;

		embedding *user = &user_mat[uid],
			*item_seen = &prod_mat[iid_seen],
			*item_unseen = &prod_mat[iid_unseen];

		float user_val = user->vals[threadIdx.x],
			seen_val = item_seen->vals[threadIdx.x],
			unseen_val = item_unseen->vals[threadIdx.x];

		temp->vals[threadIdx.x] = seen_val - unseen_val;
		__syncthreads();

		float score = dot(user, temp);
		float z = 1.0 / (1.0 + exp(score));

		// if (uid == 29 && iid_seen == 1481 && threadIdx.x == 0) {
		// 	printf("%.3f %.3f\n", score, z);
		// }

		if (z < .5) continue;

		atomicAdd(&user->vals[threadIdx.x], alpha*(z*(seen_val-unseen_val)-lambda*user_val));
		atomicAdd(&item_seen->vals[threadIdx.x], alpha*(z*user_val-lambda*seen_val));
		atomicAdd(&item_unseen->vals[threadIdx.x], alpha*(-z*user_val-lambda*unseen_val));
	}

}

int main(int argc,char **argv) {

	// Setup input data
	ifstream fin("/home/swang3/data/user_item_trim.txt");
	triple user_item_trim[NUM_SAMPLES];
	int i = 0;
	string line;
	while (getline(fin, line)) {
		istringstream iss(line);
		triple temp;
		iss >> temp.uid;
		iss >> temp.iid_seen;
		iss >> temp.iid_unseen;
		user_item_trim[i] = temp;
		i++;
	}
	triple *d_user_item_trim;
	cudaMalloc((void **) &d_user_item_trim, NUM_SAMPLES * sizeof(triple));
	cudaMemcpy(d_user_item_trim, user_item_trim, NUM_SAMPLES * sizeof(triple), cudaMemcpyHostToDevice);

	// Setup user/item embeddings
	embedding user_mat[NUM_USERS], prod_mat[NUM_ITEMS];
	srand(time(NULL));
	for(int i=0; i<NUM_USERS; i++)
		for(int j=0; j<RANK; j++)
			user_mat[i].vals[j] = ((float) rand()/RAND_MAX);

	for(int i=0; i<NUM_ITEMS; i++)
		for(int j=0; j<RANK; j++)
			prod_mat[i].vals[j] = ((float) rand()/RAND_MAX);

	// print_embedding(&user_mat[29]);
	// print_embedding(&prod_mat[1481]);

	embedding *d_user_mat, *d_prod_mat;
	cudaMalloc((void **) &d_user_mat, NUM_USERS * sizeof(embedding));
	cudaMemcpy(d_user_mat, user_mat, NUM_USERS * sizeof(embedding), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_prod_mat, NUM_ITEMS * sizeof(embedding));
	cudaMemcpy(d_prod_mat, prod_mat, NUM_ITEMS * sizeof(embedding), cudaMemcpyHostToDevice);

	// Run Matrix Factorization Kernel
	GpuTimer timer;
	timer.Start();
	bpr_update_kernel<<<256, RANK>>>(d_user_item_trim, d_user_mat, d_prod_mat, 0.1, 0.001);
	timer.Stop();

	cudaMemcpy(user_mat, d_user_mat, NUM_USERS * sizeof(embedding), cudaMemcpyDeviceToHost);
	cudaMemcpy(prod_mat, d_prod_mat, NUM_ITEMS * sizeof(embedding), cudaMemcpyDeviceToHost);

	cudaFree(d_user_item_trim);
	cudaFree(d_user_mat);
	cudaFree(d_prod_mat);
	printf("Time elapsed = %g ms\n", timer.Elapsed());

	// print_embedding(&user_mat[29]);
	// print_embedding(&prod_mat[1481]);

	return 0;
}