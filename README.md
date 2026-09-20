# CUDA-Accelerated Monte Carlo SDE Engine

A high-throughput CUDA C++ simulation engine engineered to benchmark the numerical convergence of the **Euler-Maruyama scheme** for Geometric Brownian Motion (GBM) across $10^8$ stochastic sample paths. 

Built using low-level CUDA Driver APIs (NVRTC), register-level RNG caching via `curanddx`, hardware FMA instructions, and custom shared-memory parallel reduction kernels.

> 📄 **Full Technical Report & Plots:**  
> For the complete stochastic calculus derivations, GPU register occupancy analysis, runtime profiling, and empirical convergence log-log plots, **[Read the Full Technical Report (PDF)](docs/report.pdf)**.
>
> ## Technical & Architectural Highlights

* **Low-Level CUDA Driver API & NVRTC Integration:** Circumvented CUDA Runtime API limitations to compile runtime kernels using NVRTC and manage GPU execution contexts via the Driver API.
* **In-Register RNG State Caching:** Integrated `curanddx` to cache random number generator states directly in SM registers rather than global memory, maximizing throughput during path generation.
* **Hardware-Accelerated Double Precision:** Utilized Fused Multiply-Add (`fma`) hardware instructions to perform float operations in a single clock cycle while preserving critical floating-point precision for small error terms.
* **Parallel Reduction Architecture:** Designed custom two-pass shared-memory parallel reduction kernels to aggregate sample path expectations across 100,000,000 values in $13.94\text{ ms}$.
* **Empirical Scheme Verification:** Executed log-log linear regressions across varying step sizes ($n = 16 \dots 160$) to empirically verify the theoretical order of convergence ($p = 2$).

---

## Headline Performance Benchmarks

Hardware Environment: **NVIDIA GeForce RTX 3070**  
Sample Size ($N$): **100,000,000 paths** | Step Count ($n$): **160 iterations**

| Metric / Kernel | Measured Execution | Efficiency / Bandwidth |
| :--- | :--- | :--- |
| **Point-wise GBM Kernel** | $6,363.42\text{ ms}$ | $15.24\text{ GFLOPs}$ |
| **Supremum GBM Kernel** | $—$ | **$28.20\text{ GFLOPs}$** ($10\%$ Theoretical Peak FP64) |
| **Parallel Sum Reduction** | **$13.94\text{ ms}$** | **$57.82\text{ GB/s}$** Effective Bandwidth |

*Register Usage:* Capped at $37$ registers/thread, sustaining peak SM warp occupancy ($256$ threads/block $\times$ $6$ blocks/SM).

---

## Repository Structure

```text
.
├── docs/
│   └── report.pdf           # Full technical paper with regression plots & analysis
├── src/                     # C++ / CUDA C header and source files (.cu, .hpp, .h, .c)
├── cuda-sde-engine.sln      # Visual Studio solution file
├── cuda-sde-engine.vcxproj  # Visual Studio project file
└── README.md
