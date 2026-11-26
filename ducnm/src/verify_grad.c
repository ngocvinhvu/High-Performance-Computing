/*
 * Numerical gradient verification (FP64 central difference) and equivalence check
 * between direct multi-loop convolution and im2col + GEMM.
 */
#include "core/fmnist_loader.h"

int main(void)
{
    FMnistData ds = alloc_fmnist_data();
    scalar_t *w = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    scalar_t *g_naive = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    scalar_t *g_gemm  = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    const int bounds[5] = {OFF_CONV1, OFF_CONV2, OFF_FC1, OFF_FC2, TOTAL_PARAMS};
    const double eps = 1e-6;
    uint64_t rng = 42;
    int status = 0;

    load_fmnist_idx("data", &ds);
    init_cnn_weights(w, 2026);
    for (int s = 0; s < 4; s++) {
        backward_sample_naive(w, ds.images + s * IN_AREA, ds.labels[s], g_naive);
        backward_sample_im2col(w, ds.images + s * IN_AREA, ds.labels[s], g_gemm, NULL);
    }

    for (int blk = 0; blk < 4; blk++) {
        double max_rel = 0;
        const int span = bounds[blk + 1] - bounds[blk];
        for (int probe = 0; probe < 50; probe++) {
            const int idx = bounds[blk] + (probe < 45 ? (int)(xor64_next(&rng) % span) : span - 1 - (probe - 45));
            const scalar_t saved = w[idx];
            double loss_pos = 0, loss_neg = 0;
            w[idx] = saved + eps;
            for (int s = 0; s < 4; s++) loss_pos += eval_sample_loss(w, ds.images + s * IN_AREA, ds.labels[s]);
            w[idx] = saved - eps;
            for (int s = 0; s < 4; s++) loss_neg += eval_sample_loss(w, ds.images + s * IN_AREA, ds.labels[s]);
            w[idx] = saved;

            const double fd_grad = (loss_pos - loss_neg) / (2.0 * eps);
            const double rel_err = fabs(fd_grad - g_naive[idx]) / fmax(1e-3, fabs(fd_grad) + fabs(g_naive[idx]));
            if (rel_err > max_rel) max_rel = rel_err;
        }
        printf("[VERIFY_FD] block=%d max_rel_err=%.3e status=%s\n", blk + 1, max_rel, max_rel < 1e-4 ? "OK" : "FAIL");
        if (max_rel >= 1e-4) status = 1;
    }

    double l2_diff = 0, l2_ref = 0;
    for (int i = 0; i < TOTAL_PARAMS; i++) {
        double d = (double)g_naive[i] - (double)g_gemm[i];
        l2_diff += d * d;
        l2_ref += (double)g_naive[i] * (double)g_naive[i];
    }
    double eq_err = sqrt(l2_diff / fmax(1e-12, l2_ref));
    printf("[VERIFY_GEMM] im2col_vs_naive_rel_err=%.3e status=%s\n", eq_err, eq_err < 1e-9 ? "OK" : "FAIL");
    if (eq_err >= 1e-9) status = 1;

    free_fmnist_data(&ds);
    free(w); free(g_naive); free(g_gemm);
    return status;
}
