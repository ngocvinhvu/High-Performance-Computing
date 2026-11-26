/*
 * Implementation of CNN forward & backward propagation in two execution modes:
 *   1. Direct multi-loop convolution (baseline reference)
 *   2. Unfolded im2col/col2im + cache-friendly i-k-j SIMD matrix multiplication (GEMM)
 */
#include "lenet_gemm.h"
#include "sys_timer.h"

typedef struct {
    scalar_t conv1_act[CONV1_CH * IN_AREA];
    scalar_t pool1_out[CONV1_CH * POOL1_SIDE * POOL1_SIDE];
    scalar_t conv2_act[CONV2_CH * POOL1_SIDE * POOL1_SIDE];
    scalar_t pool2_out[FC1_IN];
    scalar_t fc1_act[FC1_OUT];
    scalar_t logits[NUM_CLASSES];
    int pool1_argmax[CONV1_CH * POOL1_SIDE * POOL1_SIDE];
    int pool2_argmax[FC1_IN];
} FwdWorkspace;

static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int imin(int a, int b) { return a < b ? a : b; }

static void init_tensor_he(scalar_t *buf, int n_weights, int n_bias, double scale, uint64_t *rng)
{
    for (int i = 0; i < n_weights; i++) buf[i] = (scalar_t)(scale * xor64_gaussian(rng));
    for (int i = 0; i < n_bias; i++) buf[n_weights + i] = 0;
}

void init_cnn_weights(scalar_t *weights, int seed)
{
    uint64_t rng = (uint64_t)seed ^ 0xA5A5A5A55A5A5A5AULL;
    init_tensor_he(weights + OFF_CONV1, CONV1_CH * K_SIZE * K_SIZE, CONV1_CH, sqrt(2.0 / (K_SIZE * K_SIZE)), &rng);
    init_tensor_he(weights + OFF_CONV2, CONV2_CH * CONV1_CH * K_SIZE * K_SIZE, CONV2_CH, sqrt(2.0 / (CONV1_CH * K_SIZE * K_SIZE)), &rng);
    init_tensor_he(weights + OFF_FC1, FC1_OUT * FC1_IN, FC1_OUT, sqrt(2.0 / FC1_IN), &rng);
    init_tensor_he(weights + OFF_FC2, NUM_CLASSES * FC1_OUT, NUM_CLASSES, sqrt(1.0 / FC1_OUT), &rng);
}

/* ---------- Matrix Multiplication (GEMM) & Unfolding (im2col/col2im) ---------- */

/* C[M x N] += A[M x K] * B[K x N] in stride-1 i-k-j order for AVX2 vectorization */
void op_matmul_nn(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        scalar_t *c_row = C + i * N;
        for (int k = 0; k < K; k++) {
            const scalar_t a_val = A[i * K + k];
            const scalar_t *b_row = B + k * N;
            for (int j = 0; j < N; j++)
                c_row[j] += a_val * b_row[j];
        }
    }
}

/* C[M x N] += A[M x K] * B^T[N x K] */
static void op_matmul_nt(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        const scalar_t *a_row = A + i * K;
        for (int j = 0; j < N; j++) {
            const scalar_t *b_row = B + j * K;
            scalar_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += a_row[k] * b_row[k];
            C[i * N + j] += acc;
        }
    }
}

/* C[M x N] = A^T[K x M] * B[K x N] */
static void op_matmul_tn(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K)
{
    memset(C, 0, sizeof(scalar_t) * (size_t)M * N);
    for (int k = 0; k < K; k++) {
        const scalar_t *a_col = A + k * M;
        const scalar_t *b_row = B + k * N;
        for (int i = 0; i < M; i++) {
            const scalar_t a_val = a_col[i];
            scalar_t *c_row = C + i * N;
            for (int j = 0; j < N; j++)
                c_row[j] += a_val * b_row[j];
        }
    }
}

static void unfold_im2col(const scalar_t *feat_in, scalar_t *col_mat, int c_in, int side)
{
    const int hw = side * side;
    memset(col_mat, 0, sizeof(scalar_t) * (size_t)(c_in * K_SIZE * K_SIZE * hw));
    for (int c = 0; c < c_in; c++) {
        for (int ky = 0; ky < K_SIZE; ky++) {
            for (int kx = 0; kx < K_SIZE; kx++) {
                const int row_idx = (c * K_SIZE + ky) * K_SIZE + kx;
                scalar_t *dst = col_mat + row_idx * hw;
                const int y_beg = imax(0, K_PAD - ky), y_end = imin(side, side + K_PAD - ky);
                const int x_beg = imax(0, K_PAD - kx), x_end = imin(side, side + K_PAD - kx);
                const int offset = c * hw + (ky - K_PAD) * side + (kx - K_PAD);
                for (int y = y_beg; y < y_end; y++)
                    for (int x = x_beg; x < x_end; x++)
                        dst[y * side + x] = feat_in[y * side + x + offset];
            }
        }
    }
}

static void fold_col2im(const scalar_t *dcol_mat, scalar_t *dfeat_in, int c_in, int side)
{
    const int hw = side * side;
    memset(dfeat_in, 0, sizeof(scalar_t) * (size_t)(c_in * hw));
    for (int c = 0; c < c_in; c++) {
        for (int ky = 0; ky < K_SIZE; ky++) {
            for (int kx = 0; kx < K_SIZE; kx++) {
                const int row_idx = (c * K_SIZE + ky) * K_SIZE + kx;
                const scalar_t *src = dcol_mat + row_idx * hw;
                const int y_beg = imax(0, K_PAD - ky), y_end = imin(side, side + K_PAD - ky);
                const int x_beg = imax(0, K_PAD - kx), x_end = imin(side, side + K_PAD - kx);
                const int offset = c * hw + (ky - K_PAD) * side + (kx - K_PAD);
                for (int y = y_beg; y < y_end; y++)
                    for (int x = x_beg; x < x_end; x++)
                        dfeat_in[y * side + x + offset] += src[y * side + x];
            }
        }
    }
}

/* ---------- Direct Convolution, MaxPool, Dense & Softmax ---------- */

void op_conv_naive_fwd(const scalar_t *in_map, scalar_t *out_map, const scalar_t *w_layer, int c_in, int c_out, int side)
{
    const int hw = side * side;
    const scalar_t *bias = w_layer + c_out * c_in * K_SIZE * K_SIZE;

    for (int o = 0; o < c_out; o++) {
        scalar_t *dst = out_map + o * hw;
        for (int i = 0; i < hw; i++) dst[i] = bias[o];
        for (int c = 0; c < c_in; c++) {
            for (int ky = 0; ky < K_SIZE; ky++) {
                for (int kx = 0; kx < K_SIZE; kx++) {
                    const scalar_t k_val = w_layer[((o * c_in + c) * K_SIZE + ky) * K_SIZE + kx];
                    const int y_beg = imax(0, K_PAD - ky), y_end = imin(side, side + K_PAD - ky);
                    const int x_beg = imax(0, K_PAD - kx), x_end = imin(side, side + K_PAD - kx);
                    const int offset = c * hw + (ky - K_PAD) * side + (kx - K_PAD);
                    for (int y = y_beg; y < y_end; y++)
                        for (int x = x_beg; x < x_end; x++)
                            dst[y * side + x] += k_val * in_map[y * side + x + offset];
                }
            }
        }
        for (int i = 0; i < hw; i++) dst[i] = dst[i] > 0 ? dst[i] : 0;
    }
}

static void op_conv_naive_bwd(const scalar_t *in_map, const scalar_t *dout_map, scalar_t *din_map,
                              const scalar_t *w_layer, scalar_t *g_layer, int c_in, int c_out, int side)
{
    const int hw = side * side, n_weights = c_out * c_in * K_SIZE * K_SIZE;
    if (din_map) memset(din_map, 0, sizeof(scalar_t) * c_in * hw);

    for (int o = 0; o < c_out; o++) {
        const scalar_t *src = dout_map + o * hw;
        scalar_t b_grad = 0;
        for (int i = 0; i < hw; i++) b_grad += src[i];
        g_layer[n_weights + o] += b_grad;

        for (int c = 0; c < c_in; c++) {
            for (int ky = 0; ky < K_SIZE; ky++) {
                for (int kx = 0; kx < K_SIZE; kx++) {
                    const int k_idx = ((o * c_in + c) * K_SIZE + ky) * K_SIZE + kx;
                    const int y_beg = imax(0, K_PAD - ky), y_end = imin(side, side + K_PAD - ky);
                    const int x_beg = imax(0, K_PAD - kx), x_end = imin(side, side + K_PAD - kx);
                    const int offset = c * hw + (ky - K_PAD) * side + (kx - K_PAD);
                    const scalar_t k_val = w_layer[k_idx];
                    scalar_t w_grad = 0;
                    for (int y = y_beg; y < y_end; y++) {
                        for (int x = x_beg; x < x_end; x++)
                            w_grad += src[y * side + x] * in_map[y * side + x + offset];
                        if (din_map) {
                            for (int x = x_beg; x < x_end; x++)
                                din_map[y * side + x + offset] += k_val * src[y * side + x];
                        }
                    }
                    g_layer[k_idx] += w_grad;
                }
            }
        }
    }
}

static void op_maxpool2x2_fwd(const scalar_t *in_map, scalar_t *out_map, int *argmax_buf, int channels, int side)
{
    const int half = side / 2;
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < half; y++) {
            for (int x = 0; x < half; x++) {
                const int tl = (c * side + 2 * y) * side + 2 * x;
                const int candidates[4] = {tl, tl + 1, tl + side, tl + side + 1};
                int best = tl;
                for (int k = 1; k < 4; k++)
                    if (in_map[candidates[k]] > in_map[best]) best = candidates[k];
                const int out_idx = (c * half + y) * half + x;
                out_map[out_idx] = in_map[best];
                argmax_buf[out_idx] = best;
            }
        }
    }
}

static void op_maxpool_relu_bwd(const scalar_t *dout, const int *argmax_buf, const scalar_t *conv_act,
                                scalar_t *dconv, int channels, int side)
{
    const int half = side / 2;
    memset(dconv, 0, sizeof(scalar_t) * channels * side * side);
    for (int i = 0; i < channels * half * half; i++) {
        int win = argmax_buf[i];
        if (conv_act[win] > 0) dconv[win] = dout[i];
    }
}

void op_fc_matvec_fwd(const scalar_t *in_vec, scalar_t *out_vec, const scalar_t *w_layer, int dim_in, int dim_out, int apply_relu)
{
    for (int o = 0; o < dim_out; o++) {
        const scalar_t *w_row = w_layer + o * dim_in;
        scalar_t acc = w_layer[dim_out * dim_in + o];
        for (int i = 0; i < dim_in; i++) acc += w_row[i] * in_vec[i];
        out_vec[o] = (apply_relu && acc < 0) ? 0 : acc;
    }
}

static void op_fc_matvec_bwd(const scalar_t *in_vec, const scalar_t *dout_vec, scalar_t *din_vec,
                             const scalar_t *w_layer, scalar_t *g_layer, int dim_in, int dim_out)
{
    memset(din_vec, 0, sizeof(scalar_t) * dim_in);
    for (int o = 0; o < dim_out; o++) {
        const scalar_t d = dout_vec[o];
        if (d == 0) continue;
        g_layer[dim_out * dim_in + o] += d;
        for (int i = 0; i < dim_in; i++) g_layer[o * dim_in + i] += d * in_vec[i];
        for (int i = 0; i < dim_in; i++) din_vec[i] += d * w_layer[o * dim_in + i];
    }
}

static void run_fwd_pass_naive(const scalar_t *weights, const scalar_t *img, FwdWorkspace *ws)
{
    op_conv_naive_fwd(img, ws->conv1_act, weights + OFF_CONV1, 1, CONV1_CH, IN_SIDE);
    op_maxpool2x2_fwd(ws->conv1_act, ws->pool1_out, ws->pool1_argmax, CONV1_CH, IN_SIDE);
    op_conv_naive_fwd(ws->pool1_out, ws->conv2_act, weights + OFF_CONV2, CONV1_CH, CONV2_CH, POOL1_SIDE);
    op_maxpool2x2_fwd(ws->conv2_act, ws->pool2_out, ws->pool2_argmax, CONV2_CH, POOL1_SIDE);
    op_fc_matvec_fwd(ws->pool2_out, ws->fc1_act, weights + OFF_FC1, FC1_IN, FC1_OUT, 1);
    op_fc_matvec_fwd(ws->fc1_act, ws->logits, weights + OFF_FC2, FC1_OUT, NUM_CLASSES, 0);
}

static scalar_t compute_softmax_xent(const scalar_t *logits, int label, scalar_t *dlogits)
{
    scalar_t max_l = logits[0], sum_exp = 0, probs[NUM_CLASSES];
    for (int j = 1; j < NUM_CLASSES; j++)
        if (logits[j] > max_l) max_l = logits[j];
    for (int j = 0; j < NUM_CLASSES; j++)
        sum_exp += (probs[j] = (scalar_t)exp(logits[j] - max_l));
    if (dlogits) {
        for (int j = 0; j < NUM_CLASSES; j++)
            dlogits[j] = probs[j] / sum_exp - (j == label);
    }
    return max_l - logits[label] + (scalar_t)log(sum_exp);
}

scalar_t backward_sample_naive(const scalar_t *weights, const scalar_t *img, int label, scalar_t *grad_acc)
{
    FwdWorkspace ws;
    scalar_t dlogits[NUM_CLASSES], dfc1[FC1_OUT], dpool2[FC1_IN];
    scalar_t dconv2[CONV2_CH * POOL1_SIDE * POOL1_SIDE], dpool1[CONV1_CH * POOL1_SIDE * POOL1_SIDE];
    scalar_t dconv1[CONV1_CH * IN_AREA];

    run_fwd_pass_naive(weights, img, &ws);
    const scalar_t loss = compute_softmax_xent(ws.logits, label, dlogits);

    op_fc_matvec_bwd(ws.fc1_act, dlogits, dfc1, weights + OFF_FC2, grad_acc + OFF_FC2, FC1_OUT, NUM_CLASSES);
    for (int i = 0; i < FC1_OUT; i++)
        if (ws.fc1_act[i] <= 0) dfc1[i] = 0;
    op_fc_matvec_bwd(ws.pool2_out, dfc1, dpool2, weights + OFF_FC1, grad_acc + OFF_FC1, FC1_IN, FC1_OUT);

    op_maxpool_relu_bwd(dpool2, ws.pool2_argmax, ws.conv2_act, dconv2, CONV2_CH, POOL1_SIDE);
    op_conv_naive_bwd(ws.pool1_out, dconv2, dpool1, weights + OFF_CONV2, grad_acc + OFF_CONV2, CONV1_CH, CONV2_CH, POOL1_SIDE);

    op_maxpool_relu_bwd(dpool1, ws.pool1_argmax, ws.conv1_act, dconv1, CONV1_CH, IN_SIDE);
    op_conv_naive_bwd(img, dconv1, NULL, weights + OFF_CONV1, grad_acc + OFF_CONV1, 1, CONV1_CH, IN_SIDE);
    return loss;
}

scalar_t backward_sample_im2col(const scalar_t *weights, const scalar_t *img, int label, scalar_t *grad_acc, GemmProfile *prof)
{
    FwdWorkspace ws;
    scalar_t col1_buf[K_SIZE * K_SIZE * IN_AREA];
    scalar_t col2_buf[CONV1_CH * K_SIZE * K_SIZE * POOL1_SIDE * POOL1_SIDE];
    scalar_t dcol2_buf[CONV1_CH * K_SIZE * K_SIZE * POOL1_SIDE * POOL1_SIDE];
    scalar_t dlogits[NUM_CLASSES], dfc1[FC1_OUT], dpool2[FC1_IN];
    scalar_t dconv2[CONV2_CH * POOL1_SIDE * POOL1_SIDE], dpool1[CONV1_CH * POOL1_SIDE * POOL1_SIDE];
    scalar_t dconv1[CONV1_CH * IN_AREA];
    double t0, t1;

    /* Block 1: im2col + GEMM + ReLU/MaxPool */
    t0 = wall_time_sec();
    unfold_im2col(img, col1_buf, 1, IN_SIDE);
    t1 = wall_time_sec();
    if (prof) prof->t_unfold += t1 - t0;

    t0 = t1;
    for (int o = 0; o < CONV1_CH; o++) {
        scalar_t b = weights[OFF_CONV1 + CONV1_CH * K_SIZE * K_SIZE + o];
        for (int i = 0; i < IN_AREA; i++) ws.conv1_act[o * IN_AREA + i] = b;
    }
    op_matmul_nn(weights + OFF_CONV1, col1_buf, ws.conv1_act, CONV1_CH, IN_AREA, K_SIZE * K_SIZE);
    t1 = wall_time_sec();
    if (prof) prof->t_conv_mm += t1 - t0;

    t0 = t1;
    for (int i = 0; i < CONV1_CH * IN_AREA; i++)
        ws.conv1_act[i] = ws.conv1_act[i] > 0 ? ws.conv1_act[i] : 0;
    op_maxpool2x2_fwd(ws.conv1_act, ws.pool1_out, ws.pool1_argmax, CONV1_CH, IN_SIDE);
    t1 = wall_time_sec();
    if (prof) prof->t_act_pool += t1 - t0;

    /* Block 2: im2col + GEMM + ReLU/MaxPool */
    t0 = t1;
    unfold_im2col(ws.pool1_out, col2_buf, CONV1_CH, POOL1_SIDE);
    t1 = wall_time_sec();
    if (prof) prof->t_unfold += t1 - t0;

    t0 = t1;
    for (int o = 0; o < CONV2_CH; o++) {
        scalar_t b = weights[OFF_CONV2 + CONV2_CH * CONV1_CH * K_SIZE * K_SIZE + o];
        for (int i = 0; i < POOL1_SIDE * POOL1_SIDE; i++)
            ws.conv2_act[o * POOL1_SIDE * POOL1_SIDE + i] = b;
    }
    op_matmul_nn(weights + OFF_CONV2, col2_buf, ws.conv2_act, CONV2_CH, POOL1_SIDE * POOL1_SIDE, CONV1_CH * K_SIZE * K_SIZE);
    t1 = wall_time_sec();
    if (prof) prof->t_conv_mm += t1 - t0;

    t0 = t1;
    for (int i = 0; i < CONV2_CH * POOL1_SIDE * POOL1_SIDE; i++)
        ws.conv2_act[i] = ws.conv2_act[i] > 0 ? ws.conv2_act[i] : 0;
    op_maxpool2x2_fwd(ws.conv2_act, ws.pool2_out, ws.pool2_argmax, CONV2_CH, POOL1_SIDE);
    t1 = wall_time_sec();
    if (prof) prof->t_act_pool += t1 - t0;

    /* Block 3: Dense layers fwd + loss + Dense layers bwd */
    t0 = t1;
    op_fc_matvec_fwd(ws.pool2_out, ws.fc1_act, weights + OFF_FC1, FC1_IN, FC1_OUT, 1);
    op_fc_matvec_fwd(ws.fc1_act, ws.logits, weights + OFF_FC2, FC1_OUT, NUM_CLASSES, 0);
    t1 = wall_time_sec();
    if (prof) prof->t_dense_mm += t1 - t0;

    t0 = t1;
    const scalar_t loss = compute_softmax_xent(ws.logits, label, dlogits);
    t1 = wall_time_sec();
    if (prof) prof->t_loss_other += t1 - t0;

    t0 = t1;
    op_fc_matvec_bwd(ws.fc1_act, dlogits, dfc1, weights + OFF_FC2, grad_acc + OFF_FC2, FC1_OUT, NUM_CLASSES);
    for (int i = 0; i < FC1_OUT; i++)
        if (ws.fc1_act[i] <= 0) dfc1[i] = 0;
    op_fc_matvec_bwd(ws.pool2_out, dfc1, dpool2, weights + OFF_FC1, grad_acc + OFF_FC1, FC1_IN, FC1_OUT);
    t1 = wall_time_sec();
    if (prof) prof->t_dense_mm += t1 - t0;

    /* Backward Block 2: MaxPool2 + Conv2 GEMMs + col2im */
    t0 = t1;
    op_maxpool_relu_bwd(dpool2, ws.pool2_argmax, ws.conv2_act, dconv2, CONV2_CH, POOL1_SIDE);
    for (int o = 0; o < CONV2_CH; o++) {
        scalar_t acc = 0;
        for (int i = 0; i < POOL1_SIDE * POOL1_SIDE; i++)
            acc += dconv2[o * POOL1_SIDE * POOL1_SIDE + i];
        grad_acc[OFF_CONV2 + CONV2_CH * CONV1_CH * K_SIZE * K_SIZE + o] += acc;
    }
    t1 = wall_time_sec();
    if (prof) prof->t_act_pool += t1 - t0;

    t0 = t1;
    op_matmul_nt(dconv2, col2_buf, grad_acc + OFF_CONV2, CONV2_CH, CONV1_CH * K_SIZE * K_SIZE, POOL1_SIDE * POOL1_SIDE);
    op_matmul_tn(weights + OFF_CONV2, dconv2, dcol2_buf, CONV1_CH * K_SIZE * K_SIZE, POOL1_SIDE * POOL1_SIDE, CONV2_CH);
    t1 = wall_time_sec();
    if (prof) prof->t_conv_mm += t1 - t0;

    t0 = t1;
    fold_col2im(dcol2_buf, dpool1, CONV1_CH, POOL1_SIDE);
    t1 = wall_time_sec();
    if (prof) prof->t_unfold += t1 - t0;

    /* Backward Block 1: MaxPool1 + Conv1 GEMM */
    t0 = t1;
    op_maxpool_relu_bwd(dpool1, ws.pool1_argmax, ws.conv1_act, dconv1, CONV1_CH, IN_SIDE);
    for (int o = 0; o < CONV1_CH; o++) {
        scalar_t acc = 0;
        for (int i = 0; i < IN_AREA; i++) acc += dconv1[o * IN_AREA + i];
        grad_acc[OFF_CONV1 + CONV1_CH * K_SIZE * K_SIZE + o] += acc;
    }
    t1 = wall_time_sec();
    if (prof) prof->t_act_pool += t1 - t0;

    t0 = t1;
    op_matmul_nt(dconv1, col1_buf, grad_acc + OFF_CONV1, CONV1_CH, K_SIZE * K_SIZE, IN_AREA);
    t1 = wall_time_sec();
    if (prof) prof->t_conv_mm += t1 - t0;

    return loss;
}

scalar_t eval_sample_loss(const scalar_t *weights, const scalar_t *img, int label)
{
    FwdWorkspace ws;
    run_fwd_pass_naive(weights, img, &ws);
    return compute_softmax_xent(ws.logits, label, NULL);
}

int predict_sample_class(const scalar_t *weights, const scalar_t *img)
{
    FwdWorkspace ws;
    run_fwd_pass_naive(weights, img, &ws);
    int best_cls = 0;
    for (int j = 1; j < NUM_CLASSES; j++)
        if (ws.logits[j] > ws.logits[best_cls]) best_cls = j;
    return best_cls;
}

void apply_sgd_momentum(scalar_t *weights, scalar_t *vel, const scalar_t *grad, scalar_t inv_batch, scalar_t lr, scalar_t mom)
{
    for (int i = 0; i < TOTAL_PARAMS; i++) {
        vel[i] = mom * vel[i] + inv_batch * grad[i];
        weights[i] -= lr * vel[i];
    }
}
