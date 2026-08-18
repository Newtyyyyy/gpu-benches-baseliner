// Force-included (via -include) only for the release-hip-nvidia build, which compiles
// the hand-tuned gpu-benches/*/hip/*.hip sources through HIP's NVIDIA platform (nvcc).
// Those sources are off-limits to edit, and call hipMallocAsync(&typed_ptr, ...) the way
// real HIP's templated convenience overload allows. But nvidia_detail/nvidia_hip_runtime_api.h
// also declares a concrete non-template hipMallocAsync(void**, ...) that C++ overload
// resolution prefers over the template, so the typed-pointer call fails to compile. Rewrite
// the call at the call site instead of touching the forbidden hip/ sources.
#ifndef HIP_NVIDIA_MALLOC_ASYNC_SHIM_H
#define HIP_NVIDIA_MALLOC_ASYNC_SHIM_H

#include <cuda.h>
#include <cuda_runtime.h>

// ROCm 7.2.4's nvidia_detail/nvidia_hip_runtime_api.h gates its hipMemcpyBatchAsync,
// hipMemcpy3DBatchAsync and hipKernelGetLibrary wrappers on CUDA_VERSION >= 12020,
// assuming CUDA's batch-memcpy/library API surface exists from CUDA 12.2 onward. It
// doesn't: CUDA 12.4.131 (the toolkit pinned for this build) has neither the
// cudaMemcpyAttributes/cudaMemcpy3DBatchOp types nor these three functions - they
// were only added in a later CUDA release. None of those hip* wrappers are called by
// this project; declare (never define/call) just enough for their bodies to
// type-check without touching the forbidden hip/ sources.
#if CUDA_VERSION < 13000
enum cudaMemcpyFlags {};
enum cudaMemcpySrcAccessOrder {};
enum cudaMemcpy3DOperandType {};
struct cudaMemcpyAttributes;
struct cudaMemcpy3DBatchOp;
extern cudaError_t cudaMemcpyBatchAsync(void *const *dsts, const void *const *srcs, const size_t *sizes,
                                        size_t count, struct cudaMemcpyAttributes *attrs, size_t *attrsIdxs,
                                        size_t numAttrs, size_t *failIdx, cudaStream_t stream);
extern cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, struct cudaMemcpy3DBatchOp *opList, size_t *failIdx,
                                          unsigned long long flags, cudaStream_t stream);
extern CUresult cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel);
#endif

#include <hip/hip_runtime.h>

#define hipMallocAsync(ptr, size, stream) hipMallocAsync(reinterpret_cast<void **>(ptr), size, stream)

#endif // HIP_NVIDIA_MALLOC_ASYNC_SHIM_H
