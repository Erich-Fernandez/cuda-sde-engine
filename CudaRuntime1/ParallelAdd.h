#pragma once
struct ParallelSumResult {
	double sum;
	float milliseconds;
};


float timed_parallel_add(double* input, double* output, int InputSize);
ParallelSumResult timed_parallel_add_pureReduction(double* input, int n);