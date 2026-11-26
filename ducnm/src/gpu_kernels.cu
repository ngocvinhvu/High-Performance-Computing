/*
 * CUDA kernels compiled at runtime via NVRTC for ducnm HPC benchmarks:
 *   - 16x16 Shared-memory tiled GEMM (`kern_tiled_gemm`)
 *   - Batch unfolding (`kern_im2col_unfold`)
 *   - Warp-coalesced shuffle-reduced Matvec/Dense forward (`kern_fc_warp_fwd`)
 *   - Direct 2D convolution, max-pooling, softmax cross-entropy, and momentum SGD
 */
#define GLOBAL_TID ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define TILE_DIM 16

static __device__ inline int d_max(int a, int b) { return a > b ? a : b; }
static __device__ inline int d_min(int a, int b) { return a < b ? a : b; }

extern "C" __global__ void kern_gather_minibatch(const float *all_imgs, const int *indices,
                                                 float *batch_x, int bsz, int hw)
{
    const int tid = GLOBAL_TID;
    if (tid >= bsz * hw) return;
    batch_x[tid] = all_imgs[indices[tid / hw] * hw + tid % hw];
}

extern "C" __global__ void kern_conv2d_fwd(const float *feat_in, float *feat_out, const float *weights,
                                           int bsz, int c_in, int c_out, int side, int ksize)
{
    const int tid = GLOBAL_TID, hw = side * side;
    if (tid >= bsz * c_out * hw) return;
    const int pix = tid % hw, o = (tid / hw) % c_out, n = tid / (hw * c_out);
    const int y = pix / side, x = pix % side, pad = ksize / 2;
    const int ky0 = d_max(0, pad - y), ky1 = d_min(ksize, side + pad - y);
    const int kx0 = d_max(0, pad - x), kx1 = d_min(ksize, side + pad - x);

    float acc = weights[c_out * c_in * ksize * ksize + o];
    for (int c = 0; c < c_in; c++) {
        const float *in_ch = feat_in + (n * c_in + c) * hw;
        const float *flt = weights + (o * c_in + c) * ksize * ksize;
        for (int ky = ky0; ky < ky1; ky++)
            for (int kx = kx0; kx < kx1; kx++)
                acc += flt[ky * ksize + kx] * in_ch[(y + ky - pad) * side + x + kx - pad];
    }
    feat_out[tid] = acc > 0.f ? acc : 0.f;
}

extern "C" __global__ void kern_maxpool2x2_fwd(const float *feat_in, float *feat_out, int *argmax_idx,
                                               int bsz, int channels, int side)
{
    const int tid = GLOBAL_TID, half = side / 2, out_hw = half * half;
    if (tid >= bsz * channels * out_hw) return;
    const int pos = tid % out_hw, plane = tid / out_hw;
    const int tl = (plane * side + 2 * (pos / half)) * side + 2 * (pos % half);

    int best = tl;
    if (feat_in[tl + 1] > feat_in[best]) best = tl + 1;
    if (feat_in[tl + side] > feat_in[best]) best = tl + side;
    if (feat_in[tl + side + 1] > feat_in[best]) best = tl + side + 1;
    feat_out[tid] = feat_in[best];
    argmax_idx[tid] = best;
}

extern "C" __global__ void kern_maxpool_relu_bwd(const float *dout, const int *argmax_idx,
                                                 const float *conv_act, float *dconv,
                                                 int bsz, int channels, int side)
{
    const int tid = GLOBAL_TID, half = side / 2, hw = side * side;
    if (tid >= bsz * channels * hw) return;
    const int pos = tid % hw, plane = tid / hw;
    const int pool_pos = (plane * half + (pos / side) / 2) * half + (pos % side) / 2;
    dconv[tid] = (argmax_idx[pool_pos] == tid && conv_act[tid] > 0.f) ? dout[pool_pos] : 0.f;
}

extern "C" __global__ void kern_conv2d_bwd_input(const float *dout, float *din, const float *weights,
                                                 int bsz, int c_in, int c_out, int side, int ksize)
{
    const int tid = GLOBAL_TID, hw = side * side;
    if (tid >= bsz * c_in * hw) return;
    const int pix = tid % hw, c = (tid / hw) % c_in, n = tid / (hw * c_in);
    const int y = pix / side, x = pix % side, pad = ksize / 2;
    const int ky0 = d_max(0, y + pad - side + 1), ky1 = d_min(ksize, y + pad + 1);
    const int kx0 = d_max(0, x + pad - side + 1), kx1 = d_min(ksize, x + pad + 1);

    float acc = 0.f;
    for (int o = 0; o < c_out; o++) {
        const float *dout_ch = dout + (n * c_out + o) * hw;
        const float *flt = weights + (o * c_in + c) * ksize * ksize;
        for (int ky = ky0; ky < ky1; ky++)
            for (int kx = kx0; kx < kx1; kx++)
                acc += flt[ky * ksize + kx] * dout_ch[(y - ky + pad) * side + x - kx + pad];
    }
    din[tid] = acc;
}

extern "C" __global__ void kern_conv2d_bwd_weight_map(const float *feat_in, const float *dout, float *partial_g,
                                                      int bsz, int c_in, int c_out, int side, int ksize)
{
    const int tid = GLOBAL_TID, n_w = c_out * c_in * ksize * ksize, n_tot = n_w + c_out, hw = side * side;
    if (tid >= bsz * n_tot) return;
    const int n = tid / n_tot, p = tid % n_tot;
    float acc = 0.f;

    if (p >= n_w) {
        const float *dout_ch = dout + (n * c_out + p - n_w) * hw;
        for (int i = 0; i < hw; i++) acc += dout_ch[i];
    } else {
        const int kx = p % ksize, ky = (p / ksize) % ksize, pair = p / (ksize * ksize), pad = ksize / 2;
        const float *dout_ch = dout + (n * c_out + pair / c_in) * hw;
        const float *in_ch = feat_in + (n * c_in + pair % c_in) * hw;
        const int y0 = d_max(0, pad - ky), y1 = d_min(side, side + pad - ky);
        const int x0 = d_max(0, pad - kx), x1 = d_min(side, side + pad - kx);
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                acc += dout_ch[y * side + x] * in_ch[(y + ky - pad) * side + x + kx - pad];
    }
    partial_g[tid] = acc;
}

extern "C" __global__ void kern_reduce_batch_grad(const float *partial_g, float *grad_out, int bsz, int n_tot)
{
    const int p = GLOBAL_TID;
    if (p >= n_tot) return;
    float acc = 0.f;
    for (int n = 0; n < bsz; n++) acc += partial_g[n * n_tot + p];
    grad_out[p] = acc;
}

/* Warp-coalesced FC / Matvec forward: 32 threads in a warp cooperate on 1 row with stride-1 reads */
extern "C" __global__ void kern_fc_warp_fwd(const float *in_vec, float *out_vec, const float *weights,
                                            int bsz, int dim_in, int dim_out, int apply_relu)
{
    const int tid = GLOBAL_TID;
    const int warp_idx = tid >> 5;
    const int lane = tid & 31;
    if (warp_idx >= bsz * dim_out) return;
    const int o = warp_idx % dim_out;
    const float *x = in_vec + (warp_idx / dim_out) * dim_in;
    const float *w_row = weights + (size_t)o * dim_in;

    float acc = 0.f;
    for (int i = lane; i < dim_in; i += 32)
        acc += w_row[i] * x[i];

    acc += __shfl_down(acc, 16);
    acc += __shfl_down(acc, 8);
    acc += __shfl_down(acc, 4);
    acc += __shfl_down(acc, 2);
    acc += __shfl_down(acc, 1);

    if (lane == 0) {
        acc += weights[(size_t)dim_out * dim_in + o];
        out_vec[warp_idx] = (apply_relu && acc < 0.f) ? 0.f : acc;
    }
}

extern "C" __global__ void kern_fc_bwd_input(const float *dout, const float *post_act, float *din,
                                             const float *weights, int bsz, int dim_in, int dim_out)
{
    const int tid = GLOBAL_TID;
    if (tid >= bsz * dim_in) return;
    const int row_base = (tid / dim_in) * dim_out, i = tid % dim_in;
    float acc = 0.f;
    for (int o = 0; o < dim_out; o++)
        if (!post_act || post_act[row_base + o] > 0.f)
            acc += weights[o * dim_in + i] * dout[row_base + o];
    din[tid] = acc;
}

extern "C" __global__ void kern_fc_bwd_weight(const float *in_vec, const float *dout, const float *post_act,
                                              float *grad_out, int bsz, int dim_in, int dim_out)
{
    const int p = GLOBAL_TID, n_w = dim_out * dim_in;
    if (p >= n_w + dim_out) return;
    const int is_bias = (p >= n_w);
    const int o = is_bias ? (p - n_w) : (p / dim_in), i = is_bias ? 0 : (p % dim_in);
    float acc = 0.f;
    for (int n = 0; n < bsz; n++)
        if (!post_act || post_act[n * dim_out + o] > 0.f)
            acc += dout[n * dim_out + o] * (is_bias ? 1.f : in_vec[n * dim_in + i]);
    grad_out[p] = acc;
}

extern "C" __global__ void kern_softmax_xent(const float *logits, const int *all_labels, const int *indices,
                                             float *dlogits, float *sample_loss, int *sample_hit,
                                             int bsz, int num_cls)
{
    const int n = GLOBAL_TID;
    if (n >= bsz) return;
    const float *l_row = logits + n * num_cls;
    const int target = all_labels[indices[n]];
    int pred = 0;
    for (int j = 1; j < num_cls; j++)
        if (l_row[j] > l_row[pred]) pred = j;

    float sum_e = 0.f;
    for (int j = 0; j < num_cls; j++) {
        const float e = expf(l_row[j] - l_row[pred]);
        dlogits[n * num_cls + j] = e;
        sum_e += e;
    }
    for (int j = 0; j < num_cls; j++)
        dlogits[n * num_cls + j] = dlogits[n * num_cls + j] / sum_e - (j == target ? 1.f : 0.f);
    sample_loss[n] = l_row[pred] - l_row[target] + logf(sum_e);
    sample_hit[n] = (pred == target);
}

extern "C" __global__ void kern_sgd_momentum(float *weights, float *vel, const float *grad, int n_params,
                                             float inv_bsz, float lr, float mom)
{
    const int p = GLOBAL_TID;
    if (p >= n_params) return;
    vel[p] = mom * vel[p] + inv_bsz * grad[p];
    weights[p] -= lr * vel[p];
}

extern "C" __global__ void kern_im2col_unfold(const float *feat_in, float *col_mat,
                                              int bsz, int c_in, int side, int ksize)
{
    const int tid = GLOBAL_TID, hw = side * side, kdim = c_in * ksize * ksize, total = bsz * hw * kdim;
    if (tid >= total) return;
    const int k = tid % kdim, row = tid / kdim;
    const int n = row / hw, pix = row % hw;
    const int y = pix / side, x = pix % side, pad = ksize / 2;
    const int c = k / (ksize * ksize), rem = k % (ksize * ksize), ky = rem / ksize, kx = rem % ksize;
    const int iy = y + ky - pad, ix = x + kx - pad;
    float v = 0.f;
    if (iy >= 0 && iy < side && ix >= 0 && ix < side)
        v = feat_in[((n * c_in + c) * side + iy) * side + ix];
    col_mat[tid] = v;
}

extern "C" __global__ void kern_tiled_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    const int r = blockIdx.y * TILE_DIM + threadIdx.y;
    const int c = blockIdx.x * TILE_DIM + threadIdx.x;
    float acc = 0.f;

    const int n_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < n_tiles; t++) {
        const int ac = t * TILE_DIM + threadIdx.x;
        const int br = t * TILE_DIM + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (r < M && ac < K) ? A[r * K + ac] : 0.f;
        tileB[threadIdx.y][threadIdx.x] = (br < K && c < N) ? B[br * N + c] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE_DIM; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (r < M && c < N)
        C[r * N + c] = acc;
}
