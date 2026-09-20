#pragma once
#include "CUDArunner.hpp"
#include "Configs.h"
#include <chrono>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>


void GBM(double x, // initial value
    double b,                             // drift
    double s,                             // diffusion  
    int iterations,
    double right_endpoint,
    size_t sample_size,
    GpuContext& ctx,
    unsigned runs);

void GBM_sup(double x, // initial value
    double b,                             // drift
    double s,                             // diffusion  
    int iterations,
    double right_endpoint,
    size_t sample_size,
    GpuContext& ctx,
    unsigned runs);

void GBM_upgradedParallel(double x, // initial value
    double b,                             // drift
    double s,                             // diffusion  
    int iterations,
    double right_endpoint,
    size_t sample_size,
    GpuContext& ctx,
    unsigned runs);