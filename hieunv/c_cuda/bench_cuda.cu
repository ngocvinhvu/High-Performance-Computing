// bench_cuda.cu - CUDA version of the benchmarks, every kernel written by hand (no cuBLAS, no cuDNN,
// no Thrust/CUB): the program links only the CUDA runtime.
//
// Same experiments and CSV columns as bench_seq.c and the PyTorch code in ../code:
//   conv  caffe    per image: im2col kernel, then GEMM kernel W(96x121) x col(121x2916) + bias  [main series]
//         batched  one im2col for the whole batch, one batched GEMM launch (grid.z = image)
//         direct   one thread per output value, weights in constant memory, no im2col
//   fc    matvec_naive  one thread per row (reads of a warp are n floats apart: not coalesced)
//         matvec        one warp per row, float4 coalesced reads, warp-shuffle reduction        [main series]
//         gemm_naive    one thread per element of C
//         gemm_tiled    16x16 tiles of A and B in shared memory
//         gemm          64-column tiles, each thread keeps a TM x 4 block of C in registers   [main series]
//   bw    sum, dot (two-pass reductions), copy, triad (float4 grid-stride loops), matvec
//   fcb   (n x n)(n x B): mm1 = matvec, mm16..mm128 = gemm
//   step  caffe conv split into im2col only / GEMM only / both
//
// Timing as in the PyTorch version: synchronize, start the host clock, launch, note the time when the
// launches return (issue_ms), synchronize, stop the clock. Data is already on the GPU; host<->device
// copies are not timed. Before each implementation the GPU is kept busy for --warm-seconds (GEMM).
//
// Build:  nvcc -O3 -std=c++17 -arch=sm_75 -o bench_cuda bench_cuda.cu      (sm_75 = Tesla T4)
// Usage:  ./bench_cuda --check-only
//         ./bench_cuda --machine kaggle --outdir results/kaggle/c_cuda1 --order-seed 1 --repeat 30
#include "common.h"

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        cudaError_t e_ = (call);                                                                           \
        if (e_ != cudaSuccess)                                                                             \
            die("CUDA %s at %s:%d: %s", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); \
    } while (0)
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

static int g_sms = 40;          // multiprocessors
static int g_full_blocks = 160; // blocks of 256 threads that fill every SM once

static int grid_for(long n, int block) {
    long g = (n + block - 1) / block, cap = (long)g_sms * 32;
    return (int)(g < 1 ? 1 : g < cap ? g : cap);
}

// ===================================================================== kernels

__global__ void fill_k(float *__restrict__ p, long n, unsigned long long seed) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        unsigned long long z = seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)i * 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 31;
        z *= 0x94D049BB133111EBULL;
        z ^= z >> 29;
        p[i] = (float)(z >> 40) * (2.0f / 16777216.0f) - 1.0f;
    }
}

// ---- im2col for nb images: col[b][r][p], r = (c*11 + kh)*11 + kw, p = oh*54 + ow (Caffe layout).
// Consecutive threads write consecutive p, so the writes are coalesced.
__global__ void im2col_k(const float *__restrict__ x, float *__restrict__ col, int nb) {
    const long total = (long)nb * COL_ELEMS;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        const int p = (int)(i % PATCHES);
        const long t = i / PATCHES;
        const int r = (int)(t % PATCH_LEN), b = (int)(t / PATCH_LEN);
        const int oh = p / OUT_HW, ow = p % OUT_HW;
        const int c = r / (KS * KS), kh = (r / KS) % KS, kw = r % KS;
        col[i] = x[b * IMG_ELEMS + (long)c * IMG * IMG + (long)(oh * STRIDE + kh) * IMG + ow * STRIDE + kw];
    }
}

// ---- GEMM, row-major: C (M x N) = A (M x K) * B (K x N) + bias[row]; blockIdx.z selects a matrix pair
// B + z*sB, C + z*sC (A is shared), which is how the batched conv runs every image in one launch.

__global__ void gemm_naive_k(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                             const float *__restrict__ bias, float *__restrict__ C, long sB, long sC) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
    B += blockIdx.z * sB;
    C += blockIdx.z * sC;
    if (row >= M || col >= N) return;
    float s = 0.f;
    for (int k = 0; k < K; k++) s = fmaf(A[(long)row * K + k], B[(long)k * N + col], s);
    C[(long)row * N + col] = s + (bias ? bias[row] : 0.f);
}

template <int T>
__global__ void gemm_tiled_k(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                             const float *__restrict__ bias, float *__restrict__ C, long sB, long sC) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float s = 0.f;
    B += blockIdx.z * sB;
    C += blockIdx.z * sC;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(long)row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(long)(k0 + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; k++) s = fmaf(As[ty][k], Bs[k][tx], s);
        __syncthreads();
    }
    if (row < M && col < N) C[(long)row * N + col] = s + (bias ? bias[row] : 0.f);
}

// Register blocking: a block of 16 x 16 threads computes a (16*TM) x 64 tile of C; thread (tx, ty) owns
// rows ty + 16i (i < TM) and columns tx + 16j (j < 4), so its 4*TM sums stay in registers. Tiles of
// A (stored transposed, padded) and B, 16 values of k deep, are staged in shared memory.
template <int TM>
__global__ void __launch_bounds__(256) gemm_reg_k(int M, int N, int K, const float *__restrict__ A,
                                                  const float *__restrict__ B, const float *__restrict__ bias,
                                                  float *__restrict__ C, long sB, long sC) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16, TN = 4;
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[TM][TN];
    B += blockIdx.z * sB;
    C += blockIdx.z * sC;
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int e = tid; e < BM * BK; e += 256) { // consecutive threads: consecutive k of one row of A
            const int m = e / BK, k = e % BK, gm = m0 + m, gk = k0 + k;
            As[k][m] = (gm < M && gk < K) ? A[(long)gm * K + gk] : 0.f;
        }
        for (int e = tid; e < BK * BN; e += 256) { // consecutive threads: consecutive columns of B
            const int k = e / BN, n = e % BN, gk = k0 + k, gn = n0 + n;
            Bs[k][n] = (gk < K && gn < N) ? B[(long)gk * N + gn] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int gm = m0 + ty + 16 * i;
        if (gm < M) {
            const float bv = bias ? bias[gm] : 0.f;
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const int gn = n0 + tx + 16 * j;
                if (gn < N) C[(long)gm * N + gn] = acc[i][j] + bv;
            }
        }
    }
}

// ---- direct convolution: one thread per output value out[b][oc][p]. A warp shares one filter (p runs
// fastest), so the weights are read from constant memory as a broadcast.
__constant__ float c_w[C_OUT * PATCH_LEN];
__constant__ float c_bias[C_OUT];

__global__ void conv_direct_k(const float *__restrict__ x, float *__restrict__ out, int nb) {
    const long total = (long)nb * OUT_ELEMS;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        const int p = (int)(i % PATCHES);
        const long t = i / PATCHES;
        const int oc = (int)(t % C_OUT), b = (int)(t / C_OUT);
        const int oh = p / OUT_HW, ow = p % OUT_HW;
        const float *src = x + b * IMG_ELEMS + (long)(oh * STRIDE) * IMG + ow * STRIDE;
        float s = c_bias[oc];
        for (int c = 0; c < C_IN; c++)
#pragma unroll
            for (int kh = 0; kh < KS; kh++)
#pragma unroll
                for (int kw = 0; kw < KS; kw++)
                    s = fmaf(c_w[oc * PATCH_LEN + (c * KS + kh) * KS + kw], src[(long)c * IMG * IMG + kh * IMG + kw], s);
        out[i] = s;
    }
}

// ---- matrix-vector
__global__ void matvec_naive_k(int n, const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *row = A + (long)i * n;
    float s = 0.f;
    for (int k = 0; k < n; k++) s = fmaf(row[k], x[k], s);
    y[i] = s;
}

__global__ void matvec_warp_k(int n, const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y) {
    const int row = (int)((blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    if (row >= n) return; // the whole warp leaves together
    const float *a = A + (long)row * n;
    float s = 0.f;
    if ((n & 3) == 0) { // rows start on a 16-byte boundary: read float4
        const float4 *a4 = reinterpret_cast<const float4 *>(a), *x4 = reinterpret_cast<const float4 *>(x);
        for (int k = lane; k < n / 4; k += 32) {
            const float4 u = a4[k], v = x4[k];
            s = fmaf(u.x, v.x, s);
            s = fmaf(u.y, v.y, s);
            s = fmaf(u.z, v.z, s);
            s = fmaf(u.w, v.w, s);
        }
    } else {
        for (int k = lane; k < n; k += 32) s = fmaf(a[k], x[k], s);
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) y[row] = s;
}

// ---- STREAM-like kernels
__device__ float block_reduce_sum(float v) { // result valid in thread 0
    __shared__ float warp_part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarps = (blockDim.x + 31) >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) warp_part[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lane < nwarps ? warp_part[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__global__ void sum_partial_k(const float *__restrict__ x, long n, float *__restrict__ part) {
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x, stride = (long)gridDim.x * blockDim.x;
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    const long n4 = n / 4;
    float s = 0.f;
    for (long i = tid; i < n4; i += stride) {
        const float4 v = x4[i];
        s += (v.x + v.y) + (v.z + v.w);
    }
    for (long i = 4 * n4 + tid; i < n; i += stride) s += x[i];
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) part[blockIdx.x] = s;
}

__global__ void dot_partial_k(const float *__restrict__ x, const float *__restrict__ y, long n, float *__restrict__ part) {
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x, stride = (long)gridDim.x * blockDim.x;
    const float4 *x4 = reinterpret_cast<const float4 *>(x), *y4 = reinterpret_cast<const float4 *>(y);
    const long n4 = n / 4;
    float s = 0.f;
    for (long i = tid; i < n4; i += stride) {
        const float4 u = x4[i], v = y4[i];
        s = fmaf(u.x, v.x, s);
        s = fmaf(u.y, v.y, s);
        s = fmaf(u.z, v.z, s);
        s = fmaf(u.w, v.w, s);
    }
    for (long i = 4 * n4 + tid; i < n; i += stride) s = fmaf(x[i], y[i], s);
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) part[blockIdx.x] = s;
}

__global__ void final_sum_k(const float *__restrict__ part, int m, float *__restrict__ out) {
    float s = 0.f;
    for (int i = threadIdx.x; i < m; i += blockDim.x) s += part[i];
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) out[0] = s;
}

__global__ void copy_k(const float *__restrict__ x, float *__restrict__ y, long n) {
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x, stride = (long)gridDim.x * blockDim.x;
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    float4 *y4 = reinterpret_cast<float4 *>(y);
    for (long i = tid; i < n / 4; i += stride) y4[i] = x4[i];
    for (long i = 4 * (n / 4) + tid; i < n; i += stride) y[i] = x[i];
}

#define TRIAD_S 3.0f
__global__ void triad_k(const float *__restrict__ b, const float *__restrict__ c, float *__restrict__ o, long n) {
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x, stride = (long)gridDim.x * blockDim.x;
    const float4 *b4 = reinterpret_cast<const float4 *>(b), *c4 = reinterpret_cast<const float4 *>(c);
    float4 *o4 = reinterpret_cast<float4 *>(o);
    for (long i = tid; i < n / 4; i += stride) {
        const float4 u = b4[i], v = c4[i];
        o4[i] = make_float4(u.x + TRIAD_S * v.x, u.y + TRIAD_S * v.y, u.z + TRIAD_S * v.z, u.w + TRIAD_S * v.w);
    }
    for (long i = 4 * (n / 4) + tid; i < n; i += stride) o[i] = b[i] + TRIAD_S * c[i];
}

// ===================================================================== launchers

enum GemmKind { G_NAIVE, G_TILED, G_REG };

static void gemm_gpu(GemmKind kind, int M, int N, int K, const float *A, const float *B, const float *bias, float *C,
                     int batch = 1, long sB = 0, long sC = 0) {
    if (kind == G_NAIVE) {
        dim3 blk(32, 8), grd((N + 31) / 32, (M + 7) / 8, batch);
        gemm_naive_k<<<grd, blk>>>(M, N, K, A, B, bias, C, sB, sC);
    } else if (kind == G_TILED) {
        dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16, batch);
        gemm_tiled_k<16><<<grd, blk>>>(M, N, K, A, B, bias, C, sB, sC);
    } else if (M % 64 != 0 && M % 48 == 0) { // M = 96 (the conv): two 48-row tiles, nothing wasted
        dim3 blk(16, 16), grd((N + 63) / 64, M / 48, batch);
        gemm_reg_k<3><<<grd, blk>>>(M, N, K, A, B, bias, C, sB, sC);
    } else {
        dim3 blk(16, 16), grd((N + 63) / 64, (M + 63) / 64, batch);
        gemm_reg_k<4><<<grd, blk>>>(M, N, K, A, B, bias, C, sB, sC);
    }
}

static void im2col_gpu(const float *x, float *col, int nb) { im2col_k<<<grid_for(nb * COL_ELEMS, 256), 256>>>(x, col, nb); }

static void conv_caffe_gpu(int nb, const float *x, const float *w, const float *bias, float *col, float *out) {
    for (int b = 0; b < nb; b++) { // two launches per image, as in Caffe
        im2col_gpu(x + b * IMG_ELEMS, col, 1);
        gemm_gpu(G_REG, C_OUT, PATCHES, PATCH_LEN, w, col, bias, out + b * OUT_ELEMS);
    }
}

static void conv_batched_gpu(int nb, const float *x, const float *w, const float *bias, float *cols, float *out) {
    im2col_gpu(x, cols, nb);
    gemm_gpu(G_REG, C_OUT, PATCHES, PATCH_LEN, w, cols, bias, out, nb, COL_ELEMS, OUT_ELEMS);
}

static void conv_direct_gpu(int nb, const float *x, float *out) {
    conv_direct_k<<<grid_for(nb * OUT_ELEMS, 256), 256>>>(x, out, nb);
}

static void set_direct_weights(const float *w_host, const float *bias_host) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, w_host, sizeof(float) * C_OUT * PATCH_LEN));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias, bias_host, sizeof(float) * C_OUT));
}

static void matvec_gpu(int naive, int n, const float *A, const float *x, float *y) {
    if (naive)
        matvec_naive_k<<<(n + 255) / 256, 256>>>(n, A, x, y);
    else
        matvec_warp_k<<<(n + 7) / 8, 256>>>(n, A, x, y);
}

static void sum_gpu(const float *x, long n, float *part, float *out) {
    sum_partial_k<<<g_full_blocks, 256>>>(x, n, part);
    final_sum_k<<<1, 256>>>(part, g_full_blocks, out);
}

static void dot_gpu(const float *x, const float *y, long n, float *part, float *out) {
    dot_partial_k<<<g_full_blocks, 256>>>(x, y, n, part);
    final_sum_k<<<1, 256>>>(part, g_full_blocks, out);
}

static void fc_batch_gpu(int n, int B, const float *A, const float *X, float *Y) {
    if (B == 1)
        matvec_gpu(0, n, A, X, Y);
    else
        gemm_gpu(G_REG, n, B, n, A, X, NULL, Y);
}

// ===================================================================== memory helpers

static float *dmalloc(long n) {
    void *p = NULL;
    CUDA_CHECK(cudaMalloc(&p, (size_t)(n > 0 ? n : 1) * sizeof(float)));
    return (float *)p;
}
static float *drandom(long n, unsigned long long seed) {
    float *p = dmalloc(n);
    fill_k<<<g_full_blocks, 256>>>(p, n, seed);
    KERNEL_CHECK();
    return p;
}
static float *dzero(long n) {
    float *p = dmalloc(n);
    CUDA_CHECK(cudaMemset(p, 0, (size_t)n * sizeof(float)));
    return p;
}
static float *to_dev(const float *h, long n) {
    float *p = dmalloc(n);
    CUDA_CHECK(cudaMemcpy(p, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}
static void to_host(float *h, const float *d, long n) {
    CUDA_CHECK(cudaMemcpy(h, d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
}
static void dfree(const float *p) { CUDA_CHECK(cudaFree((void *)p)); }
static float *host_random(long n, unsigned long long seed) {
    float *p = falloc(n);
    rng_seed(seed);
    fill_random(p, n);
    return p;
}

// ===================================================================== timing

template <class F>
static Stats bench_gpu(F fn, int warmup, int repeat) {
    double *ms = dalloc(repeat), *iss = dalloc(repeat), t0, t1;
    Stats s;
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = epoch_s();
    for (int i = 0; i < repeat; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const double start = now_s();
        fn();
        iss[i] = (now_s() - start) * 1e3;
        CUDA_CHECK(cudaDeviceSynchronize());
        ms[i] = (now_s() - start) * 1e3;
    }
    t1 = epoch_s();
    KERNEL_CHECK();
    s = make_stats(ms, iss, repeat, t0, t1);
    free(ms), free(iss);
    return s;
}

// Keep the GPU busy so it reaches its sustained clocks before anything is timed (as gpu_warmup in ../code).
static void gpu_warmup(double seconds) {
    if (seconds <= 0) return;
    const int n = 2048;
    float *a = drandom((long)n * n, 1), *c = dzero((long)n * n);
    const double end = now_s() + seconds;
    while (now_s() < end) {
        gemm_gpu(G_REG, n, n, n, a, a, NULL, c);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    dfree(a), dfree(c);
}

static void row(FILE *csv, const char *exp, const char *impl, long size, const Stats &s, const Args *a, double flops,
                double bytes) {
    csv_row(csv, exp, impl, "cuda", 1, size, &s, a->warmup, a->repeat, flops, bytes, a->machine);
}

// ===================================================================== experiments

static void exp_conv(const Args *a) {
    static const char *impls[] = {"caffe", "batched", "direct"};
    FILE *csv = csv_open(a->outdir, "conv.csv");
    float *hw = host_random((long)C_OUT * PATCH_LEN, 1), *hb = host_random(C_OUT, 2);
    float *w = to_dev(hw, (long)C_OUT * PATCH_LEN), *bias = to_dev(hb, C_OUT);
    set_direct_weights(hw, hb);
    for (int ii = 0; ii < 3; ii++) {
        long list[MAX_LIST];
        int cnt = parse_list(ii == 2 ? a->direct_batches : a->batches, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        gpu_warmup(a->warm_seconds);
        for (int t = 0; t < cnt; t++) {
            const int nb = (int)list[t];
            float *x = drandom(nb * IMG_ELEMS, 100 + nb), *out = dzero(nb * OUT_ELEMS);
            float *col = dzero((ii == 1 ? nb : 1) * COL_ELEMS);
            Stats s;
            if (ii == 0)
                s = bench_gpu([&] { conv_caffe_gpu(nb, x, w, bias, col, out); }, a->warmup, a->repeat);
            else if (ii == 1)
                s = bench_gpu([&] { conv_batched_gpu(nb, x, w, bias, col, out); }, a->warmup, a->repeat);
            else
                s = bench_gpu([&] { conv_direct_gpu(nb, x, out); }, a->warmup, a->repeat);
            row(csv, "conv", impls[ii], nb, s, a, FLOPS_PER_IMAGE * nb, 4.0 * nb * (IMG_ELEMS + OUT_ELEMS));
            say("conv  cuda %-13s batch=%-5d %10.3f ms +- %.3f  (%.1f us/img, %.0f GFLOP/s)  issue %.3f ms", impls[ii],
                nb, s.mean, s.std, s.mean / nb * 1e3, FLOPS_PER_IMAGE * nb / (s.mean * 1e6), s.issue);
            dfree(x), dfree(out), dfree(col);
        }
    }
    dfree(w), dfree(bias);
    ffree(hw), ffree(hb);
    fclose(csv);
}

static void exp_fc(const Args *a) {
    static const char *impls[] = {"matvec_naive", "matvec"};
    FILE *csv = csv_open(a->outdir, "fc.csv");
    for (int ii = 0; ii < 2; ii++) {
        long list[MAX_LIST];
        int cnt = parse_list(a->mv_sizes, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        gpu_warmup(a->warm_seconds);
        for (int t = 0; t < cnt; t++) {
            const int n = (int)list[t];
            float *A = drandom((long)n * n, 200 + n), *x = drandom(n, 300 + n), *y = dzero(n);
            const int naive = ii == 0;
            Stats s = bench_gpu([&] { matvec_gpu(naive, n, A, x, y); }, a->warmup, a->repeat);
            row(csv, "fc", impls[ii], n, s, a, 2.0 * n * n, 4.0 * ((double)n * n + 2.0 * n));
            say("fc    cuda %-13s n=%-6d %10.3f ms +- %.3f  (%.1f GFLOP/s, %.1f GB/s)", impls[ii], n, s.mean, s.std,
                2.0 * n * n / (s.mean * 1e6), 4.0 * n * n / (s.mean * 1e6));
            dfree(A), dfree(x), dfree(y);
        }
    }
    fclose(csv);
}

static void exp_gemm(const Args *a) {
    static const char *impls[] = {"gemm_naive", "gemm_tiled", "gemm"};
    static const GemmKind kinds[] = {G_NAIVE, G_TILED, G_REG};
    FILE *csv = csv_open(a->outdir, "fc.csv");
    for (int ii = 0; ii < 3; ii++) {
        long list[MAX_LIST];
        int cnt = parse_list(ii == 0 ? a->gemm_naive_sizes : a->gemm_sizes, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        gpu_warmup(a->warm_seconds);
        for (int t = 0; t < cnt; t++) {
            const int n = (int)list[t];
            const double flops = 2.0 * n * (double)n * n;
            float *A = drandom((long)n * n, 400 + n), *B = drandom((long)n * n, 500 + n), *C = dzero((long)n * n);
            const GemmKind kind = kinds[ii];
            Stats s = bench_gpu([&] { gemm_gpu(kind, n, n, n, A, B, NULL, C); }, a->warmup, a->repeat);
            row(csv, "fc", impls[ii], n, s, a, flops, 12.0 * n * n);
            say("gemm  cuda %-13s n=%-6d %10.3f ms +- %.3f  (%.0f GFLOP/s)", impls[ii], n, s.mean, s.std,
                flops / (s.mean * 1e6));
            dfree(A), dfree(B), dfree(C);
        }
    }
    fclose(csv);
}

static void exp_bw(const Args *a) {
    static const char *kernels[] = {"sum", "dot", "copy", "triad", "matvec"};
    static const double bytes_per[] = {4, 8, 8, 12, 4}, flops_per[] = {1, 2, 0, 2, 2};
    FILE *csv = csv_open(a->outdir, "bw.csv");
    float *part = dzero(g_full_blocks), *res = dzero(1);
    for (int kk = 0; kk < 5; kk++) {
        long list[MAX_LIST];
        int cnt = parse_list(a->bw_sizes, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        gpu_warmup(a->warm_seconds);
        for (int t = 0; t < cnt; t++) {
            const int n = (int)list[t];
            const long N = (long)n * n;
            float *x = drandom(N, 600 + n);
            float *y = (kk == 1 || kk == 3) ? drandom(N, 700 + n) : kk == 4 ? drandom(n, 700 + n) : NULL;
            float *o = (kk == 2 || kk == 3) ? dzero(N) : kk == 4 ? dzero(n) : NULL;
            Stats s;
            if (kk == 0)
                s = bench_gpu([&] { sum_gpu(x, N, part, res); }, a->warmup, a->repeat);
            else if (kk == 1)
                s = bench_gpu([&] { dot_gpu(x, y, N, part, res); }, a->warmup, a->repeat);
            else if (kk == 2)
                s = bench_gpu([&] { copy_k<<<g_full_blocks, 256>>>(x, o, N); }, a->warmup, a->repeat);
            else if (kk == 3)
                s = bench_gpu([&] { triad_k<<<g_full_blocks, 256>>>(x, y, o, N); }, a->warmup, a->repeat);
            else
                s = bench_gpu([&] { matvec_gpu(0, n, x, y, o); }, a->warmup, a->repeat);
            row(csv, "bw", kernels[kk], n, s, a, flops_per[kk] * N, bytes_per[kk] * N);
            say("bw    cuda %-13s n=%-6d %10.3f ms +- %.3f  (%.1f GB/s)", kernels[kk], n, s.mean, s.std,
                bytes_per[kk] * N / (s.mean * 1e6));
            dfree(x);
            if (y) dfree(y);
            if (o) dfree(o);
        }
    }
    dfree(part), dfree(res);
    fclose(csv);
}

static void exp_fcb(const Args *a) {
    long sizes[MAX_LIST], bs[MAX_LIST], order[MAX_LIST];
    const int ns = parse_list(a->fc_sizes, sizes, MAX_LIST), nB = parse_list(a->fc_b, bs, MAX_LIST);
    int cnt = 0;
    FILE *csv = csv_open(a->outdir, "fcb.csv");
    if (ns * nB > MAX_LIST) die("too many (n, B) pairs");
    for (int i = 0; i < ns * nB; i++) order[cnt++] = i;
    shuffle(order, cnt, a->order_seed);
    gpu_warmup(a->warm_seconds);
    for (int t = 0; t < cnt; t++) {
        const int n = (int)sizes[order[t] / nB], B = (int)bs[order[t] % nB];
        const double flops = 2.0 * n * (double)n * B;
        float *A = drandom((long)n * n, 800 + n), *X = drandom((long)n * B, 900 + B), *Y = dzero((long)n * B);
        char impl[32];
        Stats s = bench_gpu([&] { fc_batch_gpu(n, B, A, X, Y); }, a->warmup, a->repeat);
        snprintf(impl, sizeof impl, "mm%d", B);
        row(csv, "fcb", impl, n, s, a, flops, 4.0 * n * n);
        say("fcb   cuda B=%-4d n=%-6d %10.3f ms +- %.3f  (%.0f GFLOP/s)", B, n, s.mean, s.std, flops / (s.mean * 1e6));
        dfree(A), dfree(X), dfree(Y);
    }
    fclose(csv);
}

static void exp_step(const Args *a) {
    static const char *steps[] = {"im2col", "gemm", "caffe"};
    FILE *csv = csv_open(a->outdir, "step.csv");
    float *w = drandom((long)C_OUT * PATCH_LEN, 1), *bias = drandom(C_OUT, 2), *col = dzero(COL_ELEMS);
    for (int ss = 0; ss < 3; ss++) {
        long list[MAX_LIST];
        int cnt = parse_list(a->step_batches, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        gpu_warmup(a->warm_seconds);
        for (int t = 0; t < cnt; t++) {
            const int nb = (int)list[t];
            float *x = drandom(nb * IMG_ELEMS, 100 + nb), *out = dzero(nb * OUT_ELEMS);
            Stats s;
            im2col_gpu(x, col, 1); // GEMM-only needs valid columns
            KERNEL_CHECK();
            if (ss == 0)
                s = bench_gpu([&] { for (int b = 0; b < nb; b++) im2col_gpu(x + b * IMG_ELEMS, col, 1); },
                              a->warmup, a->repeat);
            else if (ss == 1)
                s = bench_gpu([&] {
                        for (int b = 0; b < nb; b++)
                            gemm_gpu(G_REG, C_OUT, PATCHES, PATCH_LEN, w, col, bias, out + b * OUT_ELEMS);
                    }, a->warmup, a->repeat);
            else
                s = bench_gpu([&] { conv_caffe_gpu(nb, x, w, bias, col, out); }, a->warmup, a->repeat);
            row(csv, "step", steps[ss], nb, s, a, ss ? FLOPS_PER_IMAGE * nb : 0, 0);
            say("step  cuda %-13s batch=%-5d %10.3f ms +- %.3f  (%.1f us/img)  issue %.3f ms", steps[ss], nb, s.mean,
                s.std, s.mean / nb * 1e3, s.issue);
            dfree(x), dfree(out);
        }
    }
    dfree(w), dfree(bias), dfree(col);
    fclose(csv);
}

// ===================================================================== correctness (against float64 on the CPU)

static double gpu_vs_ref(const float *d, const double *ref, long n) {
    float *h = falloc(n);
    double e;
    CUDA_CHECK(cudaDeviceSynchronize());
    to_host(h, d, n);
    e = rel_err(h, ref, n);
    ffree(h);
    return e;
}

static void check_gemm_shape(int M, int N, int K, int with_bias, int batch) {
    static const char *names[] = {"gemm_naive", "gemm_tiled", "gemm"};
    const long sB = (long)K * N, sC = (long)M * N;
    float *hA = host_random((long)M * K, 11), *hB = host_random(sB * batch, 12), *hbias = host_random(M, 13);
    double *ref = dalloc(sC * batch);
    float *A = to_dev(hA, (long)M * K), *B = to_dev(hB, sB * batch), *bias = to_dev(hbias, M), *C = dzero(sC * batch);
    for (int z = 0; z < batch; z++) ref_gemm(M, N, K, hA, hB + z * sB, with_bias ? hbias : NULL, ref + z * sC);
    for (int kind = 0; kind < 3; kind++) {
        char what[96];
        CUDA_CHECK(cudaMemset(C, 0, (size_t)(sC * batch) * sizeof(float)));
        gemm_gpu((GemmKind)kind, M, N, K, A, B, with_bias ? bias : NULL, C, batch, sB, sC);
        KERNEL_CHECK();
        snprintf(what, sizeof what, "%-11s %dx%dx%d%s%s", names[kind], M, N, K, with_bias ? " +bias" : "",
                 batch > 1 ? " batch 3" : "");
        check(what, gpu_vs_ref(C, ref, sC * batch), 1e-4);
    }
    dfree(A), dfree(B), dfree(bias), dfree(C);
    ffree(hA), ffree(hB), ffree(hbias);
    free(ref);
}

static int run_checks(void) {
    g_fail = 0;
    // GEMM: odd shapes (tile edges), K > tile depth, both register-tile shapes (M = 96 uses 48-row tiles), batch
    check_gemm_shape(37, 53, 29, 1, 1);
    check_gemm_shape(5, 17, 3, 0, 1);
    check_gemm_shape(64, 80, 300, 0, 1);
    check_gemm_shape(48, 100, 20, 1, 1);
    check_gemm_shape(200, 8, 200, 0, 1);
    check_gemm_shape(C_OUT, PATCHES, PATCH_LEN, 1, 3);
    { // im2col and the three convolutions, 3 images
        const int nb = 3;
        float *hx = host_random(nb * IMG_ELEMS, 21), *hw = host_random((long)C_OUT * PATCH_LEN, 22);
        float *hb = host_random(C_OUT, 23), *hcol = falloc(COL_ELEMS), *gcol = falloc(nb * COL_ELEMS);
        double *ref = dalloc(nb * OUT_ELEMS);
        float *x = to_dev(hx, nb * IMG_ELEMS), *w = to_dev(hw, (long)C_OUT * PATCH_LEN), *bias = to_dev(hb, C_OUT);
        float *cols = dzero(nb * COL_ELEMS), *out = dzero(nb * OUT_ELEMS);
        int same = 1;
        im2col_gpu(x, cols, nb);
        KERNEL_CHECK();
        to_host(gcol, cols, nb * COL_ELEMS);
        for (int b = 0; b < nb; b++) {
            im2col_cpu(hx + b * IMG_ELEMS, hcol);
            same &= !memcmp(hcol, gcol + b * COL_ELEMS, sizeof(float) * COL_ELEMS);
        }
        check("im2col kernel = im2col on the CPU (exact)", same ? 0.0 : 1.0, 0.0);
        ref_conv(nb, hx, hw, hb, ref);
        set_direct_weights(hw, hb);
        conv_caffe_gpu(nb, x, w, bias, cols, out);
        KERNEL_CHECK();
        check("conv caffe, 3 images", gpu_vs_ref(out, ref, nb * OUT_ELEMS), 1e-4);
        CUDA_CHECK(cudaMemset(out, 0, sizeof(float) * nb * OUT_ELEMS));
        conv_batched_gpu(nb, x, w, bias, cols, out);
        KERNEL_CHECK();
        check("conv batched, 3 images", gpu_vs_ref(out, ref, nb * OUT_ELEMS), 1e-4);
        CUDA_CHECK(cudaMemset(out, 0, sizeof(float) * nb * OUT_ELEMS));
        conv_direct_gpu(nb, x, out);
        KERNEL_CHECK();
        check("conv direct, 3 images", gpu_vs_ref(out, ref, nb * OUT_ELEMS), 1e-4);
        dfree(x), dfree(w), dfree(bias), dfree(cols), dfree(out);
        ffree(hx), ffree(hw), ffree(hb), ffree(hcol), ffree(gcol);
        free(ref);
    }
    { // matvec: n = 37 and 1 take the scalar path, the others the float4 path
        static const int sizes[] = {1, 37, 300, 1000, 4096};
        for (int t = 0; t < 5; t++) {
            const int n = sizes[t];
            float *hA = host_random((long)n * n, 31), *hx = host_random(n, 32);
            double *ref = dalloc(n);
            float *A = to_dev(hA, (long)n * n), *x = to_dev(hx, n), *y = dzero(n);
            ref_matvec(n, hA, hx, ref);
            for (int naive = 1; naive >= 0; naive--) {
                char what[64];
                CUDA_CHECK(cudaMemset(y, 0, sizeof(float) * n));
                matvec_gpu(naive, n, A, x, y);
                KERNEL_CHECK();
                snprintf(what, sizeof what, "%-12s n=%d", naive ? "matvec_naive" : "matvec", n);
                check(what, gpu_vs_ref(y, ref, n), 1e-4);
            }
            dfree(A), dfree(x), dfree(y);
            ffree(hA), ffree(hx);
            free(ref);
        }
    }
    { // STREAM-like kernels: one length with a tail, one multiple of 4
        static const long lens[] = {1000003, 1L << 22};
        for (int t = 0; t < 2; t++) {
            const long N = lens[t];
            float *hx = host_random(N, 41), *hy = host_random(N, 42), *ho = falloc(N), got;
            double s = 0, d = 0, sabs = 0, dabs = 0;
            double *ref = dalloc(N);
            float *x = to_dev(hx, N), *y = to_dev(hy, N), *o = dzero(N), *part = dzero(g_full_blocks), *res = dzero(1);
            char what[64];
            for (long i = 0; i < N; i++) {
                s += hx[i], d += (double)hx[i] * hy[i];
                sabs += fabs(hx[i]), dabs += fabs((double)hx[i] * hy[i]);
            }
            sum_gpu(x, N, part, res);
            KERNEL_CHECK();
            to_host(&got, res, 1);
            snprintf(what, sizeof what, "sum N=%ld", N);
            check(what, fabs(got - s) / sabs, 1e-6);
            dot_gpu(x, y, N, part, res);
            KERNEL_CHECK();
            to_host(&got, res, 1);
            snprintf(what, sizeof what, "dot N=%ld", N);
            check(what, fabs(got - d) / dabs, 1e-6);
            copy_k<<<g_full_blocks, 256>>>(x, o, N);
            KERNEL_CHECK();
            to_host(ho, o, N);
            snprintf(what, sizeof what, "copy N=%ld (exact)", N);
            check(what, memcmp(hx, ho, sizeof(float) * N) ? 1.0 : 0.0, 0.0);
            triad_k<<<g_full_blocks, 256>>>(x, y, o, N);
            KERNEL_CHECK();
            for (long i = 0; i < N; i++) ref[i] = (double)hx[i] + 3.0 * hy[i];
            snprintf(what, sizeof what, "triad N=%ld", N);
            check(what, gpu_vs_ref(o, ref, N), 1e-6);
            dfree(x), dfree(y), dfree(o), dfree(part), dfree(res);
            ffree(hx), ffree(hy), ffree(ho);
            free(ref);
        }
    }
    { // FC with B vectors
        const int n = 200, B = 8;
        float *hA = host_random((long)n * n, 51), *hX = host_random((long)n * B, 52);
        double *ref = dalloc((long)n * B);
        float *A = to_dev(hA, (long)n * n), *X = to_dev(hX, (long)n * B), *Y = dzero((long)n * B);
        ref_gemm(n, B, n, hA, hX, NULL, ref);
        fc_batch_gpu(n, B, A, X, Y);
        KERNEL_CHECK();
        check("fcb n=200 B=8", gpu_vs_ref(Y, ref, (long)n * B), 1e-4);
        ref_gemm(n, 1, n, hA, hX, NULL, ref);
        fc_batch_gpu(n, 1, A, X, Y);
        KERNEL_CHECK();
        check("fcb n=200 B=1 (matvec)", gpu_vs_ref(Y, ref, n), 1e-4);
        dfree(A), dfree(X), dfree(Y);
        ffree(hA), ffree(hX);
        free(ref);
    }
    say("checks: %s (%d failed)", g_fail ? "FAILED" : "all passed", g_fail);
    return g_fail;
}

// ===================================================================== main

static int attr(cudaDeviceAttr what) {
    int v = -1;
    if (cudaDeviceGetAttribute(&v, what, 0) != cudaSuccess) {
        cudaGetLastError();
        v = -1;
    }
    return v;
}

static void write_env(const Args *a, const cudaDeviceProp &prop) {
    FILE *f = open_in(a->outdir, "env_cuda.json", "w");
    char cpu[256], ts[40], buf[128];
    int rt = 0, drv = 0;
    const int clk = attr(cudaDevAttrClockRate), mclk = attr(cudaDevAttrMemoryClockRate),
              bus = attr(cudaDevAttrGlobalMemoryBusWidth);
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cpu_model(cpu, sizeof cpu);
    iso_utc(ts, sizeof ts);
    fprintf(f, "{\n");
    json_str(f, "program", "bench_cuda (hand-written CUDA kernels, no cuBLAS/cuDNN)", 1);
    json_str(f, "machine", a->machine, 1);
    json_str(f, "recorded_utc", ts, 1);
    json_str(f, "cpu", cpu, 1);
    json_str(f, "gpu", prop.name, 1);
    snprintf(buf, sizeof buf, "%d.%d", prop.major, prop.minor);
    json_str(f, "gpu_compute_capability", buf, 1);
    snprintf(buf, sizeof buf, "%d.%d", rt / 1000, (rt % 1000) / 10);
    json_str(f, "cuda_runtime", buf, 1);
    snprintf(buf, sizeof buf, "%d.%d", drv / 1000, (drv % 1000) / 10);
    json_str(f, "cuda_driver_api", buf, 1);
    snprintf(buf, sizeof buf, "%d.%d.%d", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
    json_str(f, "nvcc", buf, 1);
#ifdef __VERSION__
    json_str(f, "host_compiler", __VERSION__, 1);
#endif
    json_str(f, "experiments", a->exps, 1);
    json_str(f, "batches", a->batches, 1);
    json_str(f, "direct_batches", a->direct_batches, 1);
    json_str(f, "matvec_sizes", a->mv_sizes, 1);
    json_str(f, "gemm_sizes", a->gemm_sizes, 1);
    json_str(f, "gemm_naive_sizes", a->gemm_naive_sizes, 1);
    json_str(f, "bw_sizes", a->bw_sizes, 1);
    json_str(f, "fc_sizes", a->fc_sizes, 1);
    json_str(f, "fc_b", a->fc_b, 1);
    json_str(f, "step_batches", a->step_batches, 1);
    fprintf(f, "  \"gpu_sm_count\": %d,\n  \"gpu_memory_gb\": %.1f,\n  \"gpu_max_sm_clock_mhz\": %d,\n",
            prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0, clk > 0 ? clk / 1000 : -1);
    fprintf(f, "  \"gpu_memory_clock_mhz\": %d,\n  \"gpu_memory_bus_bits\": %d,\n", mclk > 0 ? mclk / 1000 : -1, bus);
    fprintf(f, "  \"gpu_nominal_bandwidth_gbps\": %.1f,\n", (mclk > 0 && bus > 0) ? 2.0 * mclk * 1e3 * (bus / 8) / 1e9 : -1.0);
    fprintf(f, "  \"warm_seconds\": %.1f,\n  \"warmup\": %d,\n  \"repeat\": %d,\n  \"order_seed\": %d,\n  \"quick\": %d\n}\n",
            a->warm_seconds, a->warmup, a->repeat, a->order_seed, a->quick);
    fclose(f);
}

static const char *USAGE =
    "bench_cuda - hand-written CUDA benchmarks (no cuBLAS/cuDNN)\n"
    "  --check-only            run the correctness checks against float64 on the CPU and exit\n"
    "  --machine NAME          label written to every row (required unless --check-only)\n"
    "  --outdir DIR            run folder, created if needed; CSV files are appended (required)\n"
    "  --exps LIST             subset of " ALL_EXPS "\n"
    "  --warmup N --repeat N   untimed / timed runs per point (default 3 / 10)\n"
    "  --order-seed N          0 = ascending sizes, else shuffled (default 1)\n"
    "  --warm-seconds S        GPU busy time before each implementation (default 5)\n"
    "  --quick                 tiny grid, a few seconds\n"
    "  --no-check              skip the checks before timing\n"
    "  --batches, --direct-batches, --matvec-sizes, --gemm-sizes, --gemm-naive-sizes,\n"
    "  --bw-sizes, --fc-sizes, --fc-b, --step-batches LIST   comma separated grids\n";

int main(int argc, char **argv) {
    Args a;
    cudaDeviceProp prop;
    int count = 0;
    double start;
    memset(&a, 0, sizeof a);
    a.exps = ALL_EXPS;
    a.warmup = 3, a.repeat = 10, a.order_seed = 1, a.warm_seconds = 5;
    a.batches = "1,8,16,32,64,96,128,160,192,224,256";
    a.direct_batches = a.batches;
    a.mv_sizes = "256,512,1024,2048,4096,8192,12288,16384";
    a.gemm_sizes = "256,512,1024,1536,2048,3072,4096";
    a.gemm_naive_sizes = a.gemm_sizes;
    a.bw_sizes = "1024,4096,8192,16384";
    a.fc_sizes = "2048,8192";
    a.fc_b = "1,16,64,128";
    a.step_batches = "16,64,256";
    parse_args(argc, argv, &a, USAGE);

    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 1) die("no CUDA device");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_sms = prop.multiProcessorCount;
    g_full_blocks = g_sms * (prop.maxThreadsPerMultiProcessor / 256);
    if (g_full_blocks < 1) g_full_blocks = 1;
    if (g_full_blocks > 1024) g_full_blocks = 1024;
    say("GPU 0: %s, compute capability %d.%d, %d SMs, %.1f GB", prop.name, prop.major, prop.minor, g_sms,
        prop.totalGlobalMem / 1073741824.0);

    if (!a.no_check && run_checks()) return 1;
    if (a.check_only) return 0;

    mkdir_p(a.outdir);
    g_log = open_in(a.outdir, "log_cuda.txt", "a");
    write_env(&a, prop);
    say("bench_cuda: machine %s, outdir %s, warmup %d, repeat %d, order seed %d, warm %.1f s%s", a.machine, a.outdir,
        a.warmup, a.repeat, a.order_seed, a.warm_seconds, a.quick ? " (quick)" : "");
    start = now_s();
    if (has_token(a.exps, "conv")) exp_conv(&a);
    if (has_token(a.exps, "fc")) exp_fc(&a);
    if (has_token(a.exps, "gemm")) exp_gemm(&a);
    if (has_token(a.exps, "bw")) exp_bw(&a);
    if (has_token(a.exps, "fcb")) exp_fcb(&a);
    if (has_token(a.exps, "step")) exp_step(&a);
    say("done in %.1f s -> %s", now_s() - start, a.outdir);
    fclose(g_log);
    return 0;
}
