/*
 * Single-core CPU trainer and GEMM bottleneck phase profiler (cnn_cpu.exe).
 */
#include "core/fmnist_loader.h"

static void profile_cpu_gemm_phases(const FMnistData *ds, const scalar_t *w_base, int seed)
{
    static const int batch_grid[] = {16, 64, 128, 256};
    scalar_t *w = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    scalar_t *v = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    scalar_t *g = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    int *order = zalloc_buf(TRAIN_LIMIT, sizeof(int));
    shuffle_indices(order, TRAIN_LIMIT, seed, 1);

    for (int b_idx = 0; b_idx < 4; b_idx++) {
        const int bsz = batch_grid[b_idx];
        const int trials = (bsz <= 64) ? 5 : 3;
        double naive_sec = 0, gemm_step_sec = 0, upd_sec = 0;
        GemmProfile prof = {0, 0, 0, 0, 0};

        memcpy(w, w_base, sizeof(scalar_t) * TOTAL_PARAMS);
        memset(v, 0, sizeof(scalar_t) * TOTAL_PARAMS);
        for (int t = 0; t < trials; t++) {
            double t0 = wall_time_sec();
            accumulate_minibatch_naive(w, ds, order + t * bsz, bsz, g);
            apply_sgd_momentum(w, v, g, (scalar_t)1 / bsz, (scalar_t)0.02, (scalar_t)0.9);
            naive_sec += wall_time_sec() - t0;
        }
        naive_sec /= trials;

        memcpy(w, w_base, sizeof(scalar_t) * TOTAL_PARAMS);
        memset(v, 0, sizeof(scalar_t) * TOTAL_PARAMS);
        for (int t = 0; t < trials; t++) {
            double t0 = wall_time_sec();
            accumulate_minibatch_im2col(w, ds, order + t * bsz, bsz, g, &prof);
            double t1 = wall_time_sec();
            apply_sgd_momentum(w, v, g, (scalar_t)1 / bsz, (scalar_t)0.02, (scalar_t)0.9);
            double t2 = wall_time_sec();
            gemm_step_sec += t2 - t0;
            upd_sec += t2 - t1;
        }
        gemm_step_sec /= trials;
        upd_sec /= trials;
        prof.t_unfold /= trials;
        prof.t_conv_mm /= trials;
        prof.t_dense_mm /= trials;
        prof.t_act_pool /= trials;
        prof.t_loss_other /= trials;

        double flops_conv_mm = (double)bsz * (
            2.0 * CONV1_CH * (K_SIZE * K_SIZE) * IN_AREA * 2.0 +
            2.0 * CONV2_CH * (CONV1_CH * K_SIZE * K_SIZE) * (POOL1_SIDE * POOL1_SIDE) * 3.0
        );
        double flops_dense_mm = (double)bsz * (
            2.0 * FC1_IN * FC1_OUT * 3.0 +
            2.0 * FC1_OUT * NUM_CLASSES * 3.0
        );
        double flops_misc = (double)bsz * (
            CONV1_CH * IN_AREA * 3.0 + CONV1_CH * POOL1_SIDE * POOL1_SIDE * 3.0 +
            CONV2_CH * POOL1_SIDE * POOL1_SIDE * 3.0 + FC1_IN * 3.0 +
            FC1_OUT * 3.0 + NUM_CLASSES * 6.0
        ) + 4.0 * TOTAL_PARAMS;

        printf("[CPU_PROF] batch=%d naive_ms=%.3f gemm_ms=%.3f conv_mm_ms=%.3f dense_mm_ms=%.3f "
               "unfold_ms=%.3f act_pool_ms=%.3f loss_sgd_ms=%.3f "
               "conv_mm_flops=%.0f dense_mm_flops=%.0f misc_flops=%.0f\n",
               bsz, naive_sec * 1e3, gemm_step_sec * 1e3, prof.t_conv_mm * 1e3, prof.t_dense_mm * 1e3,
               prof.t_unfold * 1e3, prof.t_act_pool * 1e3, (prof.t_loss_other + upd_sec) * 1e3,
               flops_conv_mm, flops_dense_mm, flops_misc);
        fflush(stdout);
    }

    free(w); free(v); free(g); free(order);
}

int main(int argc, char **argv)
{
    CnnConfig cfg = parse_cli_args(argc, argv);
    FMnistData ds = alloc_fmnist_data();
    scalar_t *w = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    scalar_t *v = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    scalar_t *g = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    int *order  = zalloc_buf(cfg.train_samples, sizeof(int));

    load_fmnist_idx(cfg.data_dir, &ds);
    init_cnn_weights(w, cfg.seed);

    if (cfg.flag_prof_gemm) {
        profile_cpu_gemm_phases(&ds, w, cfg.seed);
        free_fmnist_data(&ds);
        free(w); free(v); free(g); free(order);
        return 0;
    }

    printf("[CONFIG] runner=cpu params=%d train=%d test=%d batch=%d mode=%s\n",
           (int)TOTAL_PARAMS, cfg.train_samples, (int)TEST_COUNT, cfg.batch_size,
           cfg.flag_naive_conv ? "naive_conv" : "im2col_gemm");

    double cum_sec = 0, ep_loss = 0, val_acc = 0, test_acc = 0;
    for (int ep = 1; ep <= cfg.epochs; ep++) {
        const double t0 = wall_time_sec();
        shuffle_indices(order, cfg.train_samples, cfg.seed, ep);
        ep_loss = 0;
        for (int pos = 0; pos < cfg.train_samples; pos += cfg.batch_size) {
            const int cur_b = (cfg.train_samples - pos < cfg.batch_size) ? (cfg.train_samples - pos) : cfg.batch_size;
            if (cfg.flag_naive_conv)
                ep_loss += accumulate_minibatch_naive(w, &ds, order + pos, cur_b, g);
            else
                ep_loss += accumulate_minibatch_im2col(w, &ds, order + pos, cur_b, g, NULL);
            apply_sgd_momentum(w, v, g, (scalar_t)1 / cur_b, cfg.lr, cfg.momentum);
        }
        const double dt = wall_time_sec() - t0;
        cum_sec += dt;
        ep_loss /= cfg.train_samples;
        val_acc = eval_classification_acc(w, &ds, VAL_OFFSET, VAL_COUNT);
        test_acc = eval_classification_acc(w, &ds, TEST_OFFSET, TEST_COUNT);
        log_epoch_metrics(ep, ep_loss, val_acc, test_acc, dt);
    }
    printf("[SUMMARY] total_s=%.4f comm_s=0.0000 loss=%.4f val_acc=%.2f test_acc=%.2f\n",
           cum_sec, ep_loss, val_acc, test_acc);

    free_fmnist_data(&ds);
    free(w); free(v); free(g); free(order);
    return 0;
}
