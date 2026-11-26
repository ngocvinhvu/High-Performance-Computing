/*
 * LeNet-style CNN definitions and GEMM/im2col profiling interface.
 * Network topology (54,314 parameters = 217,256 bytes FP32):
 *   Input (1x28x28)
 *     -> Conv1 (8 filters 5x5, pad 2) + ReLU + MaxPool 2x2  => (8x14x14)
 *     -> Conv2 (16 filters 5x5, pad 2) + ReLU + MaxPool 2x2 => (16x7x7 = 784)
 *     -> FC1   (784 -> 64) + ReLU
 *     -> FC2   (64 -> 10)  + Softmax Cross-Entropy
 */
#ifndef LENET_GEMM_H
#define LENET_GEMM_H

#ifdef VERIFY_FP64
typedef double scalar_t;
#else
typedef float scalar_t;
#endif

enum {
    IN_SIDE = 28,
    IN_AREA = IN_SIDE * IN_SIDE,
    K_SIZE = 5,
    K_PAD = K_SIZE / 2,

    CONV1_CH = 8,
    POOL1_SIDE = IN_SIDE / 2,
    CONV2_CH = 16,
    POOL2_SIDE = POOL1_SIDE / 2,

    FC1_IN = CONV2_CH * POOL2_SIDE * POOL2_SIDE,
    FC1_OUT = 64,
    NUM_CLASSES = 10,

    OFF_CONV1 = 0,
    OFF_CONV2 = OFF_CONV1 + CONV1_CH * K_SIZE * K_SIZE + CONV1_CH,
    OFF_FC1   = OFF_CONV2 + CONV2_CH * CONV1_CH * K_SIZE * K_SIZE + CONV2_CH,
    OFF_FC2   = OFF_FC1 + FC1_OUT * FC1_IN + FC1_OUT,
    TOTAL_PARAMS = OFF_FC2 + NUM_CLASSES * FC1_OUT + NUM_CLASSES
};

typedef struct {
    double t_unfold;
    double t_conv_mm;
    double t_dense_mm;
    double t_act_pool;
    double t_loss_other;
} GemmProfile;

void init_cnn_weights(scalar_t *weights, int seed);
scalar_t backward_sample_naive(const scalar_t *weights, const scalar_t *img, int label, scalar_t *grad_acc);
scalar_t backward_sample_im2col(const scalar_t *weights, const scalar_t *img, int label, scalar_t *grad_acc, GemmProfile *prof);
scalar_t eval_sample_loss(const scalar_t *weights, const scalar_t *img, int label);
int predict_sample_class(const scalar_t *weights, const scalar_t *img);
void apply_sgd_momentum(scalar_t *weights, scalar_t *vel, const scalar_t *grad, scalar_t inv_batch, scalar_t lr, scalar_t mom);

/* Low-level layer operators used by CPU vs GPU microbenchmarks */
void op_conv_naive_fwd(const scalar_t *in_map, scalar_t *out_map, const scalar_t *w_layer, int c_in, int c_out, int side);
void op_fc_matvec_fwd(const scalar_t *in_vec, scalar_t *out_vec, const scalar_t *w_layer, int dim_in, int dim_out, int apply_relu);
void op_matmul_nn(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K);

#endif
