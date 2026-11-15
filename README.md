# Euler-Maruyama Convergence Tested On GBM Using Monte Carlo Simulation Running on CUDA C++

This is a CUDA C++ project made as a visual studio solution (custom configurations must be made to adapt it to your PC - meaning running the code as is likely to just produce errors - more details given below). Specifically, this is a Euler-Maruyama scheme convergence benchmark project, applied to the Geometric Brownian Motion. This was made as a final project for my GPU Computing class.

Built using curanddx and parallel reduction kernels.

> 📄 **Full Technical Report & Plots:**  
> For the actual equation used, GPU register occupancy analysis, runtime profiling, and empirical convergence log-log plots, **[Read the Full Technical Report (PDF)](doc/report.pdf)**.

> ## Technical & Architectural Highlights
* **NVRTC Integration:** Due to bugs in the current curanddx version, it is necessary to build your own custom Driver API to be able to use curanddx. My usage is illustrative, as I have yet to see a detailed guide on how to navigate the current bugs.
* **RNG Directly Inside the Register:** Curanddx is better than the more ancient cuRand specifically because it caches the states directly in the register rather than on global memory, maximizing throughput. Though of course, this is a tradeoff and one should weigh the cons of more register usage.
* **FMA:** Fused Multiply-Add was used to perform float operations in a single clock cycle. 
* **Parallel Reduction:** Two-pass shared-memory parallel reduction kernel was used.
* **Final Plotting:** Log-log linear regression is given in the full technical report to empirically verify the theoretical order of convergence, for $p = 2$ at least.

---

## Performance Benchmarks and NVRTC & CURANDDX Concerns

Hardware Environment: **NVIDIA GeForce RTX 3070**  
Sample Size ($N$): **100,000,000 paths** | Step Count ($n$): **160 iterations**

| Metric / Kernel | Measured Execution | Efficiency / Bandwidth |
| :--- | :--- | :--- |
| **Point-wise GBM Kernel** | $6,363.42\text{ ms}$ | $15.24\text{ GFLOPs}$ |
| **Supremum GBM Kernel** | $—$ | **$28.20\text{ GFLOPs}$** ($10\%$ Theoretical Peak FP64) |
| **Parallel Sum Reduction** | **$13.94\text{ ms}$** | **$57.82\text{ GB/s}$** Effective Bandwidth |

*Register Usage:* Capped at $37$ registers/thread, sustaining peak SM warp occupancy ($256$ threads/block $\times$ $6$ blocks/SM).

My VS solution only works for my specific GPU. If you are not using an RTX 3070 and would like to run this code, change the configs before you run it. Furthermore, this code only runs for the version 0.2.3 as I had to do patchwork because some of the libraries' file locations were jumbled around. For context, curanddx, or at least ver 0.2.3, had certain header and dependent files containing mismatched include paths, causing "file-not-found" errors even if you correctly installed the library.

## Repository Structure

```text
.
├── docs/
│   └── report.pdf           # Full technical paper with regression plots & analysis
├──CudaRuntime1
│   ├── CUDArunner.hpp       # The custom class to run kernels that use curanddx and some other technicalities with CUDA
│   ├── Configs.h            
│   ├── CudaRuntime1.vcxproj # Visual Studio project file
│   ├── GBM_kernel.cu        # the main code for the actual GBM and others
│   ├── GBM_kernel.h         
│   ├── ParallelAdd.cu       # kernel for the parallel reduced addition
│   ├── ParallelAdd.h        
│   ├── rcommon.hpp          # helper files for CUDArunner.hpp, mostly gotten from official nvidia sample code
│   ├── main.cu              # just the main loop that produces the data in the infographics in report.pdf
│   ├── nvrtc_helper.hpp     # helper files for CUDArunner.hpp, mostly gotten from official nvidia sample code
└── README.md
