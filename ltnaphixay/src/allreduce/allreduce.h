/*
 * allreduce.h - the interface shared by every allreduce method.
 *
 *   void allreduce_<name>(float** grads, size_t n, int k, cudaStream_t* streams);
 *
 * Contract (every method follows it, so the comparison is fair):
 *  1. grads[i] is a device pointer on GPU i holding n floats (i = 0..k-1).
 *  2. When all streams[i] have finished, EVERY grads[i] must hold the element-wise
 *     AVERAGE (sum / k) of the k input buffers. The result is written in place.
 *  3. Enqueue work on streams[i] (the stream that belongs to GPU i) and return.
 *     No cudaDeviceSynchronize() or other blocking host waits (they would break
 *     overlap with the backward pass).
 *     You may use cudaEvents to make stream i wait for stream j.
 *  4. Must work for any n (n need not be divisible by k) and k = 1..8.
 *  5. Allowed: cudaMemcpyPeerAsync, cudaMemcpyAsync, own kernels, peer
 *     access from kernels, the helpers below. NOT allowed: NCCL, cuBLAS, MPI
 *     collectives, any other communication/math library.
 *  6. Get temporary buffers from ar_scratch() so allocation is not timed.
 *  7. Use ar_add()/ar_scale() for reductions so all methods use the same
 *     arithmetic kernel.
 *  8. Each method orders its own cross-GPU work: on entry, streams[i] is only ordered after
 *     the work that produced grads[i]. Before stream i reads or writes grads[j]
 *     (or GPU j's scratch), make it wait for an event recorded on streams[j].
 *     ar_barrier() does this for all pairs (simple but coarse; finer events are
 *     faster; tree, ring and rd use them, see ar_sync.h).
 *
 * Adding a method: implement it in src/allreduce/<name>.cu and add one line to
 * the table in src/allreduce/registry.cu.
 */
#ifndef ALLREDUCE_H
#define ALLREDUCE_H
#include <cuda_runtime.h>
#include <stddef.h>

typedef void (*allreduce_fn)(float **grads, size_t n, int k, cudaStream_t *streams);

/* ---- Methods ------------------------------------------------------------- */
void allreduce_host_ref(float **grads, size_t n, int k, cudaStream_t *streams); /* reference: via host */
void allreduce_tree    (float **grads, size_t n, int k, cudaStream_t *streams); /* flat reduce to GPU 0 + broadcast */
void allreduce_ring    (float **grads, size_t n, int k, cudaStream_t *streams); /* ring */
void allreduce_rd      (float **grads, size_t n, int k, cudaStream_t *streams); /* recursive doubling */
/* Overlap is the training loop's --overlap mode, see src/cuda/train_dp.cu */

/* ---- Registry / dispatch --------------------------------------------------- */
/* Select a method by name ("host_ref", "tree", "ring", "rd"). Returns 0 if ok. */
int  allreduce_select(const char *name);
const char *allreduce_selected_name(void);
void allreduce_list(void);
/* Calls the selected method (no-op when k == 1). */
void allreduce(float **grads, size_t n, int k, cudaStream_t *streams);

/* ---- Shared helpers (src/allreduce/common.cu) --------------------------------- */
void   ar_setup(int k);                 /* enable peer access where possible  */
int    ar_can_p2p(int i, int j);        /* 1 if GPU i can access GPU j memory */
float *ar_scratch(int dev, int slot, size_t bytes); /* cached temp buffer, slot 0..3 */
void   ar_add(int dev, float *dst, const float *src, size_t n, cudaStream_t s); /* dst += src */
void   ar_scale(int dev, float *x, float alpha, size_t n, cudaStream_t s);      /* x *= alpha */
void   ar_barrier(int k, cudaStream_t *streams); /* every stream waits for all others */
void   ar_teardown(void);

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,       \
                __LINE__, cudaGetErrorString(e_));                             \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#endif
