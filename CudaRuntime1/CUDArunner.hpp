#pragma once

#include <vector>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "nvrtc_helper.hpp" 
#include "common.hpp"

constexpr int SAMPLE_SIZE = 1000;
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y;

// Use a formula for the grid
const int GRID_X = (SAMPLE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;


class GpuContext {
public:
    CUdevice   cuDevice;
    CUcontext  context;
    GpuContext() {
        int current_device;
        CU_CHECK_AND_EXIT(cudaGetDevice(&current_device));

        CU_CHECK_AND_EXIT(cuInit(0));
        CU_CHECK_AND_EXIT(cuDeviceGet(&cuDevice, current_device));
#if CUDA_VERSION >= 13000
        CU_CHECK_AND_EXIT(cuCtxCreate(&context, (CUctxCreateParams*)0, 0, cuDevice));
#else
        CU_CHECK_AND_EXIT(cuCtxCreate(&context, 0, cuDevice));
#endif
    }
    ~GpuContext() { cuCtxDestroy(context); }
};

class GpuKernel {
private:
    CUmodule   module;
    CUfunction kernel;

public:
    GpuKernel(GpuContext& ctx, const std::string& source, const std::string& name) {

        int current_device;
        CU_CHECK_AND_EXIT(cudaGetDevice(&current_device));

        // Create NVRTC program
        std::string kernel_name_cu = name + ".cu";
        nvrtcProgram program;
        NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,           // program
            source.c_str(),      // buffer
            kernel_name_cu.c_str(), // name
            0,                  // numHeaders
            NULL,               // headers
            NULL));             // includeNames

        // Prepare compilation options
        std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "-I" CUDA_INCLUDE_DIR, // Add path to CUDA include directory
        "-I" CUDA_CCCL_INCLUDE_DIR
        };

        // This function should theoretically be able to handle all the directories without having to 
        // manually type the directories as I've done above, but sadly get_curanddx_include_dirs actually cannot find CCCL.
        std::vector<std::string> curanddx_include_dirs = nvrtc::get_curanddx_include_dirs();
        for (auto& d : curanddx_include_dirs) {
            opts.push_back(d.c_str());
        }

        // Add GPU_ARCHITECTURE definition to opts
        std::string gpu_architecture_definition =
            "-DRNG_SM=" + std::to_string(nvrtc::get_device_architecture(current_device) * 10);
        opts.push_back(gpu_architecture_definition.c_str());

        // Add gpu-architecture to opts
        std::string gpu_architecture_option = nvrtc::get_device_architecture_option(current_device);
        opts.push_back(gpu_architecture_option.c_str());

        // Compile the kernel
        nvrtcResult compileResult = nvrtcCompileProgram(program,                       // program
            static_cast<int>(opts.size()), // numOptions
            opts.data());                  // options

        // Obtain compilation log from the program
        if (compileResult != NVRTC_SUCCESS) {
            for (auto o : opts) {
                std::cout << o << std::endl;
            }
            nvrtc::print_program_log(program);
            std::exit(1);
        }

        // Obtain cubin from the program.
        size_t cubin_size;
        NVRTC_SAFE_CALL(nvrtcGetCUBINSize(program, &cubin_size));
        auto cubin = std::make_unique<char[]>(cubin_size);
        NVRTC_SAFE_CALL(nvrtcGetCUBIN(program, cubin.get()));

        // Destroy the program.
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

        // Set the passed context as "current" for this thread
        cuCtxPushCurrent(ctx.context);

        CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.get(), 0, 0, 0));
        CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, name.c_str()));

        // get register usage of kernel function [uncomment to check the kernel register usage]
        //int regs = 0;
        //cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);

        //std::cout << "Kernel: " << name << " | Registers: " << regs << std::endl;

        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
    }

    ~GpuKernel() { cuModuleUnload(module); }

    template <typename... Args>
    void launch(unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, Args... args) {
        void* params[] = { &args... };
        cuLaunchKernel(kernel, gx, gy, gz, bx, by, bz, 0, 0, params, 0);
        cuCtxSynchronize();
    }

    template <typename... Args>
    float launch_timed(const std::string& label, 
                       unsigned gx, unsigned gy, unsigned gz,
                       unsigned bx, unsigned by, unsigned bz, Args... args) {
        
        CUevent start, stop;
        cuEventCreate(&start, CU_EVENT_DEFAULT);
        cuEventCreate(&stop, CU_EVENT_DEFAULT);

        // Record the start event
        cuEventRecord(start, 0);

        // Perform the actual launch
        launch(gx, gy, gz, bx, by, bz, args...);

        // Record the stop event
        cuEventRecord(stop, 0);
        cuEventSynchronize(stop);

        float milliseconds = 0;
        cuEventElapsedTime(&milliseconds, start, stop);

        std::cout << "[GPU Kernel: " << label << "] Time: " << milliseconds << " ms" << std::endl;

        cuEventDestroy(start);
        cuEventDestroy(stop);
        
        return milliseconds;
    }

    template <typename... Args>
    float launch_timed(unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, Args... args) {

        CUevent start, stop;
        cuEventCreate(&start, CU_EVENT_DEFAULT);
        cuEventCreate(&stop, CU_EVENT_DEFAULT);

        // Record the start event
        cuEventRecord(start, 0);

        // Perform the actual launch
        launch(gx, gy, gz, bx, by, bz, args...);

        // Record the stop event
        cuEventRecord(stop, 0);
        cuEventSynchronize(stop);

        float milliseconds = 0;
        cuEventElapsedTime(&milliseconds, start, stop);
        cuEventDestroy(start);
        cuEventDestroy(stop);

        return milliseconds;
    }
};