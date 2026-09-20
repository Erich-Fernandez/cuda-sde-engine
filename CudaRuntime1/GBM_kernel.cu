#include "GBM_kernel.h"
#include "ParallelAdd.h"
#include <functional>


// registers per thread usage : 37
// registers * 1536 threads per block : 61,440 [thus we should just maximize occupancy] 
// ideal launch configurations : 256 threads per block
const std::string gbm_basic = R"kernel(
#include <curanddx.hpp>

using RNG = decltype(curanddx::Generator<curanddx::pcg>() + curanddx::SM<RNG_SM>() + curanddx::Thread());

extern "C" __global__ void gbmKernel(
    double* squared_error,
    const double x,
    const double b,
    const double s,
    const int iterations,    // n
    const double T,
    const unsigned long long seed,
    const typename RNG::offset_type offset
)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    double h = T / iterations;
    double B_total = 0;

    RNG rng(seed, ((offset + tid) % 65536), ((offset + tid) / 65536));

    double S = x;
    const double drift = b * h;
    const double std_dev = sqrt(h); // Standard deviation is sqrt(dt)

    // Using the same generator object for efficiency
    curanddx::normal<double, curanddx::icdf> dist(0.0, 1.0);

    for (int iter = 0; iter < iterations; iter++) {
        // Generate standard normal and scale by sqrt(h)
        double dB = dist.generate(rng) * std_dev;
        B_total += dB;

        // S = S + S * (b*dt + s*dW)
        // Using fma for double precision (fmaf is for float)
        double change_factor = fma(s, dB, drift);
        S = fma(S, change_factor, S);
    }
    // generate closed form output
    // (b - 0.5 * s * s) * T + s * B_total
    double exponent = fma(s, B_total, (b - 0.5 * s * s) * T);
    double closed_form = x * exp(exponent);

    squared_error[tid] = (S - closed_form) * (S - closed_form);
}
)kernel";




void GBM(double x, // initial value
    double b,                             // drift
    double s,                             // diffusion  
    int iterations,
    double right_endpoint,
    size_t sample_size,
    GpuContext& ctx,
    unsigned runs = 1) {

    GpuKernel kernel(ctx, gbm_basic, "gbmKernel");
    CUdeviceptr errors_CUptr;
    cuMemAlloc(&errors_CUptr, sample_size * sizeof(double));

    int ideal_block_size = 256;
    int GRIDSIZE = (sample_size + ideal_block_size - 1) / ideal_block_size;


    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    double* errors = reinterpret_cast<double*>(errors_CUptr);

    unsigned long long h_offset = 0ULL; // No more 0ULL literal

    float pure_gbm_time = kernel.launch_timed(GRIDSIZE, 1, 1, ideal_block_size, 1, 1, errors, x, b, s, iterations, right_endpoint, seed, h_offset);

    CUdeviceptr sum_errors_CUptr;
    cuMemAlloc(&sum_errors_CUptr, sample_size * sizeof(double));
    double* sum_errors = reinterpret_cast<double*>(sum_errors_CUptr);

    //double* sum_errors;
    //cudaMalloc((void**) &sum_errors, sample_size * sizeof(double));
    float  parallel_add_time = timed_parallel_add(errors, sum_errors, sample_size);

    double final_sum_host = 0.0;
    CUdeviceptr last_element_ptr = sum_errors_CUptr + ((sample_size - 1) * sizeof(double));
    cuMemcpyDtoH(&final_sum_host, last_element_ptr, sizeof(double));

    //cudaMemcpy(&final_sum_host, &sum_errors[sample_size - 1], sizeof(double), cudaMemcpyDeviceToHost);
    
    
    cuMemFree(errors_CUptr); cuMemFree(sum_errors_CUptr);

    double final_error = final_sum_host / (double)sample_size;

    //printf("Steps: %d | log(Steps): %lf | MSE: %lf | log(MSE) : %lf | pure_GBM_time(ms) : %f | parallel_add_time(ms) : %f\n", 
    //    iterations, 
    //    log((double)iterations), 
    //    final_error, 
    //    log(final_error), 
    //    pure_gbm_time,
    //    parallel_add_time
    //    );

    printf("%f, %f\n", pure_gbm_time, parallel_add_time);
}



const std::string gbm_sup = R"kernel(
#include <curanddx.hpp>

using RNG = decltype(curanddx::Generator<curanddx::pcg>() + curanddx::SM<RNG_SM>() + curanddx::Thread());

extern "C" __global__ void gbmKernel_sup(
    double* squared_error,
    const double x,
    const double b,
    const double s,
    const int iterations,    // n
    const double T,
    const unsigned long long seed,
    const typename RNG::offset_type offset
)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    double h = T / iterations;
    double B_total = 0;

    RNG rng(seed, ((offset + tid) % 65536), ((offset + tid) / 65536));

    double S = x;
    const double drift = b * h;
    const double std_dev = sqrt(h); // Standard deviation is sqrt(dt)
    const double b_minus_pt5s_squared = b - 0.5 * s * s;

    // Using the same generator object for efficiency
    curanddx::normal<double, curanddx::icdf> dist(0.0, 1.0);

    double max_squared_error = 0;
    double elapsed_h = 0;

    for (int iter = 0; iter < iterations; iter++) {
        elapsed_h += h;

        // Generate standard normal and scale by sqrt(h)
        double dB = dist.generate(rng) * std_dev;
        B_total += dB;

        // S = S + S * (b*dt + s*dW)
        // Using fma for double precision (fmaf is for float)
        double change_factor = fma(s, dB, drift);
        S = fma(S, change_factor, S);

        // generate closed form output
        // (b - 0.5 * s * s) * T + s * B_total
        double exponent = fma(s, B_total, (b_minus_pt5s_squared)*elapsed_h);
        double closed_form = x * exp(exponent);

        double temp_squared_error = (S - closed_form) * (S - closed_form);

        max_squared_error = fmax(max_squared_error, temp_squared_error);
    }

    squared_error[tid] = max_squared_error;
}
)kernel";

void GBM_sup(double x, // initial value
    double b,                             // drift
    double s,                             // diffusion  
    int iterations,
    double right_endpoint,
    size_t sample_size,
    GpuContext& ctx,
    unsigned runs = 1) {
    GpuKernel kernel(ctx, gbm_sup, "gbmKernel_sup");
    CUdeviceptr errors_CUptr;
    cuMemAlloc(&errors_CUptr, sample_size * sizeof(double));

    int ideal_block_size = 256;
    int GRIDSIZE = (sample_size + ideal_block_size - 1) / ideal_block_size;


    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    double* errors = reinterpret_cast<double*>(errors_CUptr);

    unsigned long long h_offset = 0ULL; // No more 0ULL literal

    float pure_gbm_time = kernel.launch_timed(GRIDSIZE, 1, 1, ideal_block_size, 1, 1, errors, x, b, s, iterations, right_endpoint, seed, h_offset);

    CUdeviceptr sum_errors_CUptr;
    cuMemAlloc(&sum_errors_CUptr, sample_size * sizeof(double));
    double* sum_errors = reinterpret_cast<double*>(sum_errors_CUptr);

    //double* sum_errors;
    //cudaMalloc((void**) &sum_errors, sample_size * sizeof(double));
    float  parallel_add_time = timed_parallel_add(errors, sum_errors, sample_size);

    double final_sum_host = 0.0;
    CUdeviceptr last_element_ptr = sum_errors_CUptr + ((sample_size - 1) * sizeof(double));
    cuMemcpyDtoH(&final_sum_host, last_element_ptr, sizeof(double));

    //cudaMemcpy(&final_sum_host, &sum_errors[sample_size - 1], sizeof(double), cudaMemcpyDeviceToHost);


    cuMemFree(errors_CUptr); cuMemFree(sum_errors_CUptr);

    double final_error = final_sum_host / (double)sample_size;

    //printf("Steps: %d | log(Steps): %lf | MSE: %lf | log(MSE) : %lf | pure_GBM_time(ms) : %f | parallel_add_time(ms) : %f\n",
    //    iterations,
    //    log((double)iterations),
    //    final_error,
    //    log(final_error),
    //    pure_gbm_time,
    //    parallel_add_time
    //);

    printf("%f, %f\n", pure_gbm_time, parallel_add_time);
}








void GBM_upgradedParallel(double x, // initial value
    double b,                             // drift
    double s,                             // diffusion  
    int iterations,
    double right_endpoint,
    size_t sample_size,
    GpuContext& ctx,
    unsigned runs = 1) {

    GpuKernel kernel(ctx, gbm_basic, "gbmKernel");
    CUdeviceptr errors_CUptr;
    cuMemAlloc(&errors_CUptr, sample_size * sizeof(double));

    int ideal_block_size = 256;
    int GRIDSIZE = (sample_size + ideal_block_size - 1) / ideal_block_size;


    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    double* errors = reinterpret_cast<double*>(errors_CUptr);

    unsigned long long h_offset = 0ULL; // No more 0ULL literal

    float pure_gbm_time = kernel.launch_timed(GRIDSIZE, 1, 1, ideal_block_size, 1, 1, errors, x, b, s, iterations, right_endpoint, seed, h_offset);


    ParallelSumResult sum_and_time = timed_parallel_add_pureReduction(errors, sample_size);

    cuMemFree(errors_CUptr); 

    double final_error = sum_and_time.sum / (double)sample_size;

    //printf("Steps: %d | log(Steps): %lf | MSE: %lf | log(MSE) : %lf | pure_GBM_time(ms) : %f | parallel_add_time(ms) : %f\n",
    //    iterations,
    //    log((double)iterations),
    //    final_error,
    //    log(final_error),
    //    pure_gbm_time,
    //    sum_and_time.milliseconds
    //);

    printf("%f, %f\n", pure_gbm_time, sum_and_time.milliseconds);
}