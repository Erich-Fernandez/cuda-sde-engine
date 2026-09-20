#include "configs.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <vector>
#include <nvrtc.h>
#include <cuda.h>
#include "ParallelAdd.h"



__global__ void BlockParallelScan(double* Input, double* Output, double* Intermediate, int InputSize) {
	__shared__ double XY[SECTION_SIZE];

	// load the inputs into the shared memory 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < InputSize) { XY[threadIdx.x] = Input[i]; }
	else {
		XY[threadIdx.x] = 0.0f; // Critical: Prevents adding garbage data
	}


	// this is the reduction step
	for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2) {
		__syncthreads();

		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < SECTION_SIZE) XY[index] += XY[index - stride];

	}

	// this is the distribution step
	for (int stride = (SECTION_SIZE) / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) XY[index + stride] += XY[index];
	}

	__syncthreads();
	if (i < InputSize) {
		Output[i] = XY[threadIdx.x];
	}

	if (threadIdx.x == SECTION_SIZE - 1 && i < InputSize) {
		Intermediate[blockIdx.x] = XY[threadIdx.x];
	}
}

__global__ void IntermediateParallelScan(double* Intermediate, int IntermediateSize) {
	__shared__ double XY[SECTION_SIZE];

	// load the inputs into the shared memory 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < IntermediateSize) { XY[threadIdx.x] = Intermediate[i]; }
	else {
		XY[threadIdx.x] = 0.0f; // Critical: Prevents adding garbage data
	}

	// this is the reduction step
	for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2) {
		__syncthreads();

		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < SECTION_SIZE) XY[index] += XY[index - stride];

	}

	// this is the distribution step
	for (int stride = (SECTION_SIZE) / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) XY[index + stride] += XY[index];
	}

	__syncthreads();
	if (i < IntermediateSize) {
		Intermediate[i] = XY[threadIdx.x];
	}

}


__global__ void AddBack(double* Output, double* Intermediate, int InputSize) {


	int i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

	if (i < InputSize) {
		Output[i] += Intermediate[blockIdx.x];
	}
}



// this is the inefficient one where we actually calculate all the intermediate sums.
// remember, we only really need the final total sum anyway

float timed_parallel_add(double* input, double* output, int InputSize) {

	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop, CU_EVENT_DEFAULT);
	cuEventRecord(start, 0);







	int INTERMEDIATE_SIZE = (InputSize + SECTION_SIZE - 1) / SECTION_SIZE;
	CUdeviceptr intermediate_CUptr;
	cuMemAlloc(&intermediate_CUptr, INTERMEDIATE_SIZE * sizeof(double));

	double* intermediate = reinterpret_cast<double*>(intermediate_CUptr);

	//double* intermediate;
	//cudaMalloc((void**)&intermediate, INTERMEDIATE_SIZE * sizeof(double));

	BlockParallelScan << <INTERMEDIATE_SIZE, SECTION_SIZE >> > (input, output, intermediate, InputSize);
	IntermediateParallelScan << <((INTERMEDIATE_SIZE + SECTION_SIZE - 1) / (SECTION_SIZE)), SECTION_SIZE >> > (intermediate, INTERMEDIATE_SIZE);
	AddBack << <INTERMEDIATE_SIZE, SECTION_SIZE >> > (output, intermediate, InputSize);

	cuEventRecord(stop, 0);
	cuEventSynchronize(stop); // This also handles the sync you had before cuMemFree

	float milliseconds = 0;
	cuEventElapsedTime(&milliseconds, start, stop);

	// 4. Cleanup
	cuMemFree(intermediate_CUptr);
	cuEventDestroy(start);
	cuEventDestroy(stop);

	return milliseconds;
}

__global__ void ReductionKernel(const double* input, double* output, int n) {
	extern __shared__ double XY[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;

	if (i < n) XY[tid] = input[i];
	else XY[tid] = 0.0f;

	__syncthreads();


	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2 ) { 
		if (tid < s) {
			XY[tid] += XY[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = XY[0];
	}


}


ParallelSumResult timed_parallel_add_pureReduction(double* input, int n) {
	ParallelSumResult return_value = { 0.0, 0.0f};
	int threads_per_block = 256;
	size_t shared_mem_size = threads_per_block * sizeof(double);
	int n_current = n;

	int intermediate_size = (n + threads_per_block - 1) / threads_per_block;
	CUdeviceptr intermediate_CUptr, final_time_CUptr;
	cuMemAlloc(&intermediate_CUptr, intermediate_size * sizeof(double));
	cuMemAlloc(&final_time_CUptr, sizeof(double));

	double* intermediate = reinterpret_cast<double*>(intermediate_CUptr);
	double* final_time = reinterpret_cast<double*>(final_time_CUptr);

	double* temp_input = input;
	double* temp_output = intermediate;
	
	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop, CU_EVENT_DEFAULT);

	cuEventRecord(start, 0);

	while (n_current > 1) {
	
		int gridSize = (n_current + threads_per_block - 1) / (threads_per_block);
	
		if (gridSize == 1) {
			ReductionKernel << <1, threads_per_block, shared_mem_size >> > (temp_input, final_time, n_current);
			n_current = 1;
		}
		else {
			ReductionKernel << <gridSize, threads_per_block, shared_mem_size >> > (temp_input, temp_output, n_current);

			// swap temp_input and temp_output
			double* prev_input = temp_input;
			temp_input = temp_output;
			temp_output = prev_input;
			n_current = gridSize;
		}
	}

	cuEventRecord(stop, 0);
	cuEventSynchronize(stop);
	cuEventElapsedTime(&return_value.milliseconds, start, stop);

	cuMemcpyDtoH(&return_value.sum, final_time_CUptr, sizeof(double));
	cuMemFree(intermediate_CUptr); cuMemFree(final_time_CUptr);
	cuEventDestroy(start); cuEventDestroy(stop);

	return return_value;
}

