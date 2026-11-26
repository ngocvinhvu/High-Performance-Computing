/*
 * Dynamic Win32 binding for NVIDIA CUDA Driver API (nvcuda.dll) and NVRTC (nvrtc64_*.dll).
 */
#ifndef GPU_DRIVER_H
#define GPU_DRIVER_H

#include "core/sys_timer.h"

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

typedef int nvrtcResult;
typedef void *nvrtcProgram;

#define CUDA_OK  0
#define NVRTC_OK 0

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuDeviceComputeCapability)(int *, int *, CUdevice);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxDestroy)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
static CUresult (*p_cuMemGetInfo)(size_t *, size_t *);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, CUstream, void **, void **);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);

static nvrtcResult (*p_nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *);
static nvrtcResult (*p_nvrtcCompileProgram)(nvrtcProgram, int, const char *const *);
static nvrtcResult (*p_nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetProgramLog)(nvrtcProgram, char *);
static nvrtcResult (*p_nvrtcGetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetPTX)(nvrtcProgram, char *);
static nvrtcResult (*p_nvrtcDestroyProgram)(nvrtcProgram *);

static void *resolve_dll_fn(HMODULE handle, const char *primary, const char *fallback)
{
    void *fn = (void *)(uintptr_t)GetProcAddress(handle, primary);
    if (!fn && fallback) fn = (void *)(uintptr_t)GetProcAddress(handle, fallback);
    if (!fn) fatal_err("Cannot resolve symbol from CUDA/NVRTC DLL:", primary);
    return fn;
}

static void bind_cuda_driver_and_nvrtc(void)
{
    HMODULE h_cuda = LoadLibraryA("nvcuda.dll");
    if (!h_cuda) fatal_err("Failed to load nvcuda.dll", NULL);

    static const char *nvrtc_candidates[] = {
        "nvrtc64_120_0.dll",
        "bin/nvrtc64_120_0.dll",
        "nvrtc64_112_0.dll",
        NULL
    };
    HMODULE h_nvrtc = NULL;
    for (int i = 0; nvrtc_candidates[i] && !h_nvrtc; i++)
        h_nvrtc = LoadLibraryA(nvrtc_candidates[i]);
    if (!h_nvrtc) fatal_err("Failed to load nvrtc64_120_0.dll (run python build_engine.py)", NULL);

    p_cuInit = resolve_dll_fn(h_cuda, "cuInit", NULL);
    p_cuDeviceGet = resolve_dll_fn(h_cuda, "cuDeviceGet", NULL);
    p_cuDeviceGetName = resolve_dll_fn(h_cuda, "cuDeviceGetName", NULL);
    p_cuDeviceComputeCapability = resolve_dll_fn(h_cuda, "cuDeviceComputeCapability", NULL);
    p_cuCtxCreate = resolve_dll_fn(h_cuda, "cuCtxCreate_v2", "cuCtxCreate");
    p_cuCtxDestroy = resolve_dll_fn(h_cuda, "cuCtxDestroy_v2", "cuCtxDestroy");
    p_cuCtxSynchronize = resolve_dll_fn(h_cuda, "cuCtxSynchronize", NULL);
    p_cuMemAlloc = resolve_dll_fn(h_cuda, "cuMemAlloc_v2", "cuMemAlloc");
    p_cuMemFree = resolve_dll_fn(h_cuda, "cuMemFree_v2", "cuMemFree");
    p_cuMemcpyHtoD = resolve_dll_fn(h_cuda, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    p_cuMemcpyDtoH = resolve_dll_fn(h_cuda, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    p_cuMemsetD32 = resolve_dll_fn(h_cuda, "cuMemsetD32_v2", "cuMemsetD32");
    p_cuMemGetInfo = resolve_dll_fn(h_cuda, "cuMemGetInfo_v2", "cuMemGetInfo");
    p_cuModuleLoadData = resolve_dll_fn(h_cuda, "cuModuleLoadData", NULL);
    p_cuModuleUnload = resolve_dll_fn(h_cuda, "cuModuleUnload", NULL);
    p_cuModuleGetFunction = resolve_dll_fn(h_cuda, "cuModuleGetFunction", NULL);
    p_cuLaunchKernel = resolve_dll_fn(h_cuda, "cuLaunchKernel", NULL);
    p_cuGetErrorString = resolve_dll_fn(h_cuda, "cuGetErrorString", NULL);

    p_nvrtcCreateProgram = resolve_dll_fn(h_nvrtc, "nvrtcCreateProgram", NULL);
    p_nvrtcCompileProgram = resolve_dll_fn(h_nvrtc, "nvrtcCompileProgram", NULL);
    p_nvrtcGetProgramLogSize = resolve_dll_fn(h_nvrtc, "nvrtcGetProgramLogSize", NULL);
    p_nvrtcGetProgramLog = resolve_dll_fn(h_nvrtc, "nvrtcGetProgramLog", NULL);
    p_nvrtcGetPTXSize = resolve_dll_fn(h_nvrtc, "nvrtcGetPTXSize", NULL);
    p_nvrtcGetPTX = resolve_dll_fn(h_nvrtc, "nvrtcGetPTX", NULL);
    p_nvrtcDestroyProgram = resolve_dll_fn(h_nvrtc, "nvrtcDestroyProgram", NULL);
}

#endif
