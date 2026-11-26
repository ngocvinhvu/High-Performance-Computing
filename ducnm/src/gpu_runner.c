/*
 * CUDA GPU trainer, layer microbenchmark, GEMM kernel profiler, and delayed-gradient
 * staleness simulator (cnn_gpu.exe).
 */
#include "core/fmnist_loader.h"
#include "gpu_driver.h"

#define CU_CALL(stmt) check_cu_err((stmt), #stmt)
#define THREADS_PER_BLK 128
#define TILE_DIM 16
#define DEV_OFFSET(base, elems) ((base) + (CUdeviceptr)(elems) * sizeof(float))

enum {
    FN_GATHER, FN_CONV_FWD, FN_POOL_FWD, FN_POOL_BWD, FN_CONV_BWD_IN, FN_CONV_BWD_W, FN_REDUCE_G,
    FN_FC_FWD, FN_FC_BWD_IN, FN_FC_BWD_W, FN_SOFTMAX, FN_SGD,
    FN_IM2COL, FN_TILED_GEMM, FN_TOTAL
};

static const char *g_kern_symbols[FN_TOTAL] = {
    "kern_gather_minibatch", "kern_conv2d_fwd", "kern_maxpool2x2_fwd", "kern_maxpool_relu_bwd",
    "kern_conv2d_bwd_input", "kern_conv2d_bwd_weight_map", "kern_reduce_batch_grad",
    "kern_fc_warp_fwd", "kern_fc_bwd_input", "kern_fc_bwd_weight", "kern_softmax_xent",
    "kern_sgd_momentum", "kern_im2col_unfold", "kern_tiled_gemm"
};

static CUfunction g_fns[FN_TOTAL];

typedef struct {
    CUdeviceptr d_imgs, d_lbls, d_order, d_w, d_v, d_g;
    CUdeviceptr d_x, d_c1, d_p1, d_arg1, d_c2, d_p2, d_arg2, d_fc1, d_logits;
    CUdeviceptr d_dlogits, d_dfc1, d_dp2, d_dc2, d_dp1, d_dc1, d_part_g;
    CUdeviceptr d_losses, d_hits;
} GpuBuffers;

static void check_cu_err(CUresult res, const char *call_site)
{
    if (res == CUDA_OK) return;
    const char *err_str = NULL;
    p_cuGetErrorString(res, &err_str);
    fprintf(stderr, "[CUDA_ERR] %s -> %s\n", call_site, err_str ? err_str : "unknown");
    exit(EXIT_FAILURE);
}

static CUdeviceptr alloc_dev_floats(size_t count)
{
    CUdeviceptr ptr;
    CU_CALL(p_cuMemAlloc(&ptr, (count ? count : 1) * sizeof(float)));
    return ptr;
}

static void invoke_1d(int fn_id, int n_threads, void **params)
{
    unsigned int grid = (unsigned int)((n_threads + THREADS_PER_BLK - 1) / THREADS_PER_BLK);
    CU_CALL(p_cuLaunchKernel(g_fns[fn_id], grid, 1, 1, THREADS_PER_BLK, 1, 1, 0, 0, params, NULL));
}

static void invoke_tiled_gemm(CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC, int M, int N, int K)
{
    void *params[] = {&dA, &dB, &dC, &M, &N, &K};
    unsigned int gx = (unsigned int)((N + TILE_DIM - 1) / TILE_DIM);
    unsigned int gy = (unsigned int)((M + TILE_DIM - 1) / TILE_DIM);
    CU_CALL(p_cuLaunchKernel(g_fns[FN_TILED_GEMM], gx, gy, 1, TILE_DIM, TILE_DIM, 1, 0, 0, params, NULL));
}

static CUmodule compile_and_load_ptx(const char *cu_path, CUdevice dev)
{
    FILE *fp = fopen(cu_path, "rb");
    if (!fp) fatal_err("Cannot open CUDA source file:", cu_path);
    fseek(fp, 0, SEEK_END);
    const long fsize = ftell(fp);
    rewind(fp);
    char *src_code = zalloc_buf(fsize + 1, 1);
    if (fread(src_code, 1, fsize, fp) != (size_t)fsize) fatal_err("Read failed:", cu_path);
    fclose(fp);

    int cc_maj = 5, cc_min = 0;
    CU_CALL(p_cuDeviceComputeCapability(&cc_maj, &cc_min, dev));
    char arch_flag[64];
    snprintf(arch_flag, sizeof arch_flag, "--gpu-architecture=compute_%d%d", cc_maj, cc_min);
    const char *compile_opts[] = {arch_flag};

    nvrtcProgram prog;
    if (p_nvrtcCreateProgram(&prog, src_code, cu_path, 0, NULL, NULL) != NVRTC_OK)
        fatal_err("nvrtcCreateProgram failed", NULL);
    nvrtcResult comp_res = p_nvrtcCompileProgram(prog, 1, compile_opts);
    size_t log_sz = 0;
    if (comp_res != NVRTC_OK) {
        p_nvrtcGetProgramLogSize(prog, &log_sz);
        char *log_buf = zalloc_buf(log_sz + 1, 1);
        p_nvrtcGetProgramLog(prog, log_buf);
        fatal_err("NVRTC compile error:\n", log_buf);
    }
    size_t ptx_sz = 0;
    p_nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx_code = zalloc_buf(ptx_sz + 1, 1);
    p_nvrtcGetPTX(prog, ptx_code);
    p_nvrtcDestroyProgram(&prog);

    CUmodule mod;
    CU_CALL(p_cuModuleLoadData(&mod, ptx_code));
    for (int i = 0; i < FN_TOTAL; i++)
        CU_CALL(p_cuModuleGetFunction(&g_fns[i], mod, g_kern_symbols[i]));
    free(src_code);
    free(ptx_code);
    return mod;
}

static void launch_conv_fwd(CUdeviceptr d_in, CUdeviceptr d_out, CUdeviceptr d_w, int bsz, int cin, int cout, int side)
{
    int ks = K_SIZE;
    void *args[] = {&d_in, &d_out, &d_w, &bsz, &cin, &cout, &side, &ks};
    invoke_1d(FN_CONV_FWD, bsz * cout * side * side, args);
}

static void launch_pool_fwd(CUdeviceptr d_in, CUdeviceptr d_out, CUdeviceptr d_arg, int bsz, int ch, int side)
{
    void *args[] = {&d_in, &d_out, &d_arg, &bsz, &ch, &side};
    invoke_1d(FN_POOL_FWD, bsz * ch * (side / 2) * (side / 2), args);
}

static void launch_pool_bwd(CUdeviceptr d_dout, CUdeviceptr d_arg, CUdeviceptr d_act, CUdeviceptr d_din, int bsz, int ch, int side)
{
    void *args[] = {&d_dout, &d_arg, &d_act, &d_din, &bsz, &ch, &side};
    invoke_1d(FN_POOL_BWD, bsz * ch * side * side, args);
}

static void launch_conv_bwd(const GpuBuffers *gb, CUdeviceptr d_in, CUdeviceptr d_dout, CUdeviceptr d_din,
                            int w_off, int bsz, int cin, int cout, int side)
{
    int ks = K_SIZE, n_tot = cout * cin * K_SIZE * K_SIZE + cout;
    CUdeviceptr d_w = DEV_OFFSET(gb->d_w, w_off), d_g = DEV_OFFSET(gb->d_g, w_off), d_part = gb->d_part_g;
    if (d_din) {
        void *a_in[] = {&d_dout, &d_din, &d_w, &bsz, &cin, &cout, &side, &ks};
        invoke_1d(FN_CONV_BWD_IN, bsz * cin * side * side, a_in);
    }
    void *a_map[] = {&d_in, &d_dout, &d_part, &bsz, &cin, &cout, &side, &ks};
    invoke_1d(FN_CONV_BWD_W, bsz * n_tot, a_map);
    void *a_red[] = {&d_part, &d_g, &bsz, &n_tot};
    invoke_1d(FN_REDUCE_G, n_tot, a_red);
}

static void launch_fc_fwd(CUdeviceptr d_in, CUdeviceptr d_out, CUdeviceptr d_w, int bsz, int nin, int nout, int relu)
{
    void *args[] = {&d_in, &d_out, &d_w, &bsz, &nin, &nout, &relu};
    invoke_1d(FN_FC_FWD, bsz * nout * 32, args);
}

static void launch_fc_bwd(const GpuBuffers *gb, CUdeviceptr d_in, CUdeviceptr d_dout, CUdeviceptr d_act,
                          CUdeviceptr d_din, int w_off, int bsz, int nin, int nout)
{
    CUdeviceptr d_w = DEV_OFFSET(gb->d_w, w_off), d_g = DEV_OFFSET(gb->d_g, w_off);
    void *a_in[] = {&d_dout, &d_act, &d_din, &d_w, &bsz, &nin, &nout};
    invoke_1d(FN_FC_BWD_IN, bsz * nin, a_in);
    void *a_w[] = {&d_in, &d_dout, &d_act, &d_g, &bsz, &nin, &nout};
    invoke_1d(FN_FC_BWD_W, nout * nin + nout, a_w);
}

static void gpu_step_forward(const GpuBuffers *gb, int offset, int bsz)
{
    int hw = IN_AREA, n_cls = NUM_CLASSES;
    CUdeviceptr d_imgs = gb->d_imgs, d_x = gb->d_x, d_idx = DEV_OFFSET(gb->d_order, offset);
    void *a_gather[] = {&d_imgs, &d_idx, &d_x, &bsz, &hw};
    invoke_1d(FN_GATHER, bsz * IN_AREA, a_gather);

    launch_conv_fwd(gb->d_x, gb->d_c1, DEV_OFFSET(gb->d_w, OFF_CONV1), bsz, 1, CONV1_CH, IN_SIDE);
    launch_pool_fwd(gb->d_c1, gb->d_p1, gb->d_arg1, bsz, CONV1_CH, IN_SIDE);
    launch_conv_fwd(gb->d_p1, gb->d_c2, DEV_OFFSET(gb->d_w, OFF_CONV2), bsz, CONV1_CH, CONV2_CH, POOL1_SIDE);
    launch_pool_fwd(gb->d_c2, gb->d_p2, gb->d_arg2, bsz, CONV2_CH, POOL1_SIDE);
    launch_fc_fwd(gb->d_p2, gb->d_fc1, DEV_OFFSET(gb->d_w, OFF_FC1), bsz, FC1_IN, FC1_OUT, 1);
    launch_fc_fwd(gb->d_fc1, gb->d_logits, DEV_OFFSET(gb->d_w, OFF_FC2), bsz, FC1_OUT, NUM_CLASSES, 0);

    CUdeviceptr d_log = gb->d_logits, d_lbl = gb->d_lbls, d_dl = gb->d_dlogits;
    CUdeviceptr d_loss = DEV_OFFSET(gb->d_losses, offset), d_hit = DEV_OFFSET(gb->d_hits, offset);
    void *a_sm[] = {&d_log, &d_lbl, &d_idx, &d_dl, &d_loss, &d_hit, &bsz, &n_cls};
    invoke_1d(FN_SOFTMAX, bsz, a_sm);
}

static void gpu_step_backward(const GpuBuffers *gb, int bsz)
{
    launch_fc_bwd(gb, gb->d_fc1, gb->d_dlogits, 0, gb->d_dfc1, OFF_FC2, bsz, FC1_OUT, NUM_CLASSES);
    launch_fc_bwd(gb, gb->d_p2, gb->d_dfc1, gb->d_fc1, gb->d_dp2, OFF_FC1, bsz, FC1_IN, FC1_OUT);
    launch_pool_bwd(gb->d_dp2, gb->d_arg2, gb->d_c2, gb->d_dc2, bsz, CONV2_CH, POOL1_SIDE);
    launch_conv_bwd(gb, gb->d_p1, gb->d_dc2, gb->d_dp1, OFF_CONV2, bsz, CONV1_CH, CONV2_CH, POOL1_SIDE);
    launch_pool_bwd(gb->d_dp1, gb->d_arg1, gb->d_c1, gb->d_dc1, bsz, CONV1_CH, IN_SIDE);
    launch_conv_bwd(gb, gb->d_x, gb->d_dc1, 0, OFF_CONV1, bsz, 1, CONV1_CH, IN_SIDE);
}

static void gpu_step_sgd(const GpuBuffers *gb, int bsz, float lr, float mom)
{
    int n_p = TOTAL_PARAMS;
    float inv_b = 1.f / bsz;
    CUdeviceptr d_w = gb->d_w, d_v = gb->d_v, d_g = gb->d_g;
    void *args[] = {&d_w, &d_v, &d_g, &n_p, &inv_b, &lr, &mom};
    invoke_1d(FN_SGD, TOTAL_PARAMS, args);
}

static GpuBuffers init_gpu_buffers(const FMnistData *ds, const scalar_t *w, int max_b)
{
    GpuBuffers gb;
    gb.d_imgs    = alloc_dev_floats((size_t)TOTAL_SAMPLES * IN_AREA);
    gb.d_lbls    = alloc_dev_floats(TOTAL_SAMPLES);
    gb.d_order   = alloc_dev_floats(TOTAL_SAMPLES);
    gb.d_losses  = alloc_dev_floats(TOTAL_SAMPLES);
    gb.d_hits    = alloc_dev_floats(TOTAL_SAMPLES);
    gb.d_w       = alloc_dev_floats(TOTAL_PARAMS);
    gb.d_v       = alloc_dev_floats(TOTAL_PARAMS);
    gb.d_g       = alloc_dev_floats(TOTAL_PARAMS);
    gb.d_x       = alloc_dev_floats(max_b * IN_AREA);
    gb.d_c1      = alloc_dev_floats(max_b * CONV1_CH * IN_AREA);
    gb.d_dc1     = alloc_dev_floats(max_b * CONV1_CH * IN_AREA);
    gb.d_p1      = alloc_dev_floats(max_b * CONV1_CH * POOL1_SIDE * POOL1_SIDE);
    gb.d_dp1     = alloc_dev_floats(max_b * CONV1_CH * POOL1_SIDE * POOL1_SIDE);
    gb.d_arg1    = alloc_dev_floats(max_b * CONV1_CH * POOL1_SIDE * POOL1_SIDE);
    gb.d_c2      = alloc_dev_floats(max_b * CONV2_CH * POOL1_SIDE * POOL1_SIDE);
    gb.d_dc2     = alloc_dev_floats(max_b * CONV2_CH * POOL1_SIDE * POOL1_SIDE);
    gb.d_p2      = alloc_dev_floats(max_b * FC1_IN);
    gb.d_dp2     = alloc_dev_floats(max_b * FC1_IN);
    gb.d_arg2    = alloc_dev_floats(max_b * FC1_IN);
    gb.d_fc1     = alloc_dev_floats(max_b * FC1_OUT);
    gb.d_dfc1    = alloc_dev_floats(max_b * FC1_OUT);
    gb.d_logits  = alloc_dev_floats(max_b * NUM_CLASSES);
    gb.d_dlogits = alloc_dev_floats(max_b * NUM_CLASSES);
    gb.d_part_g  = alloc_dev_floats(max_b * (CONV2_CH * CONV1_CH * K_SIZE * K_SIZE + CONV2_CH));

    CU_CALL(p_cuMemcpyHtoD(gb.d_imgs, ds->images, sizeof(float) * TOTAL_SAMPLES * IN_AREA));
    CU_CALL(p_cuMemcpyHtoD(gb.d_lbls, ds->labels, sizeof(int) * TOTAL_SAMPLES));
    CU_CALL(p_cuMemcpyHtoD(gb.d_w, w, sizeof(float) * TOTAL_PARAMS));
    CU_CALL(p_cuMemsetD32(gb.d_v, 0, TOTAL_PARAMS));
    return gb;
}

static double eval_gpu_split(const GpuBuffers *gb, int bsz, int start_idx, int count, int *order_buf, int *hits_buf)
{
    for (int i = 0; i < count; i++) order_buf[i] = start_idx + i;
    CU_CALL(p_cuMemcpyHtoD(gb->d_order, order_buf, sizeof(int) * count));
    for (int pos = 0; pos < count; pos += bsz)
        gpu_step_forward(gb, pos, (count - pos < bsz) ? (count - pos) : bsz);
    CU_CALL(p_cuMemcpyDtoH(hits_buf, gb->d_hits, sizeof(int) * count));
    int total_hits = 0;
    for (int i = 0; i < count; i++) total_hits += hits_buf[i];
    return 100.0 * total_hits / count;
}

static void run_gpu_training(const CnnConfig *cfg, GpuBuffers *gb)
{
    CUdeviceptr delay_ring[MAX_STALE_GAP + 1];
    int delay_bsz[MAX_STALE_GAP + 1];
    const int ring_len = cfg->staleness + 1;
    long step_no = 0;
    delay_ring[0] = gb->d_g;
    for (int i = 1; i < ring_len; i++) delay_ring[i] = alloc_dev_floats(TOTAL_PARAMS);

    int *order_buf = zalloc_buf(TOTAL_SAMPLES, sizeof(int));
    int *hits_buf  = zalloc_buf(TOTAL_SAMPLES, sizeof(int));
    float *loss_buf = zalloc_buf(TOTAL_SAMPLES, sizeof(float));
    double cum_sec = 0, ep_loss = 0, val_acc = 0, test_acc = 0;

    for (int ep = 1; ep <= cfg->epochs; ep++) {
        const double t0 = wall_time_sec();
        shuffle_indices(order_buf, cfg->train_samples, cfg->seed, ep);
        CU_CALL(p_cuMemcpyHtoD(gb->d_order, order_buf, sizeof(int) * cfg->train_samples));

        for (int pos = 0; pos < cfg->train_samples; pos += cfg->batch_size) {
            const int cur_b = (cfg->train_samples - pos < cfg->batch_size) ? (cfg->train_samples - pos) : cfg->batch_size;
            gb->d_g = delay_ring[step_no % ring_len];
            delay_bsz[step_no % ring_len] = cur_b;
            gpu_step_forward(gb, pos, cur_b);
            gpu_step_backward(gb, cur_b);
            if (step_no >= cfg->staleness) {
                const long slot = (step_no - cfg->staleness) % ring_len;
                gb->d_g = delay_ring[slot];
                gpu_step_sgd(gb, delay_bsz[slot], cfg->lr, cfg->momentum);
            }
            step_no++;
        }
        CU_CALL(p_cuCtxSynchronize());
        CU_CALL(p_cuMemcpyDtoH(loss_buf, gb->d_losses, sizeof(float) * cfg->train_samples));
        const double dt = wall_time_sec() - t0;
        cum_sec += dt;

        ep_loss = 0;
        for (int i = 0; i < cfg->train_samples; i++) ep_loss += loss_buf[i];
        ep_loss /= cfg->train_samples;

        val_acc  = eval_gpu_split(gb, cfg->batch_size, VAL_OFFSET, VAL_COUNT, order_buf, hits_buf);
        test_acc = eval_gpu_split(gb, cfg->batch_size, TEST_OFFSET, TEST_COUNT, order_buf, hits_buf);
        log_epoch_metrics(ep, ep_loss, val_acc, test_acc, dt);
    }
    printf("[SUMMARY] total_s=%.4f comm_s=0.0000 loss=%.4f val_acc=%.2f test_acc=%.2f\n",
           cum_sec, ep_loss, val_acc, test_acc);
    free(order_buf); free(hits_buf); free(loss_buf);
}

static int verify_gpu_vs_cpu_grad(const CnnConfig *cfg, GpuBuffers *gb, const FMnistData *ds, const scalar_t *w)
{
    const int bsz = cfg->batch_size;
    int *order = zalloc_buf(cfg->train_samples, sizeof(int));
    float *g_cpu = zalloc_buf(TOTAL_PARAMS, sizeof(float));
    float *g_gpu = zalloc_buf(TOTAL_PARAMS, sizeof(float));

    shuffle_indices(order, cfg->train_samples, cfg->seed, 1);
    CU_CALL(p_cuMemcpyHtoD(gb->d_order, order, sizeof(int) * bsz));
    gpu_step_forward(gb, 0, bsz);
    gpu_step_backward(gb, bsz);
    CU_CALL(p_cuMemcpyDtoH(g_gpu, gb->d_g, sizeof(float) * TOTAL_PARAMS));
    accumulate_minibatch_im2col(w, ds, order, bsz, g_cpu, NULL);

    double diff_sq = 0, ref_sq = 0;
    for (int i = 0; i < TOTAL_PARAMS; i++) {
        double d = (double)g_cpu[i] - (double)g_gpu[i];
        diff_sq += d * d;
        ref_sq  += (double)g_cpu[i] * (double)g_cpu[i];
    }
    const double rel_err = sqrt(diff_sq / ref_sq);
    printf("[VERIFY_GPU] gpu_vs_cpu_rel_err=%.3e status=%s\n", rel_err, rel_err < 1e-4 ? "OK" : "FAIL");
    free(order); free(g_cpu); free(g_gpu);
    return rel_err < 1e-4 ? 0 : 1;
}

static void run_layer_microbenchmarks(const FMnistData *ds, const scalar_t *w)
{
    static const int conv_batches[] = {1, 4, 16, 64, 256, 1024};
    static const int mv_dims[]      = {128, 256, 512, 1024, 2048, 4096};
    static const int mm_dims[]      = {128, 256, 512, 1024, 2048};

    float *host_out = zalloc_buf(1024 * CONV1_CH * IN_AREA, sizeof(float));
    CUdeviceptr d_x = alloc_dev_floats(1024 * IN_AREA);
    CUdeviceptr d_y = alloc_dev_floats(1024 * CONV1_CH * IN_AREA);
    CUdeviceptr d_w = alloc_dev_floats(OFF_CONV2);
    CU_CALL(p_cuMemcpyHtoD(d_x, ds->images, sizeof(float) * 1024 * IN_AREA));
    CU_CALL(p_cuMemcpyHtoD(d_w, w, sizeof(float) * OFF_CONV2));
    launch_conv_fwd(d_x, d_y, d_w, 64, 1, CONV1_CH, IN_SIDE);
    CU_CALL(p_cuCtxSynchronize());

    for (int i = 0; i < 6; i++) {
        const int b = conv_batches[i];
        const int iters = (2048 / b < 100) ? ((2048 / b < 10) ? 10 : 2048 / b) : 100;
        double t0 = wall_time_sec();
        for (int r = 0; r < iters; r++)
            for (int s = 0; s < b; s++)
                op_conv_naive_fwd(ds->images + s * IN_AREA, host_out + s * CONV1_CH * IN_AREA, w, 1, CONV1_CH, IN_SIDE);
        const double cpu_s = (wall_time_sec() - t0) / iters;

        t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) {
            launch_conv_fwd(d_x, d_y, d_w, b, 1, CONV1_CH, IN_SIDE);
            CU_CALL(p_cuCtxSynchronize());
        }
        const double gpu_k_s = (wall_time_sec() - t0) / iters;

        t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) {
            CU_CALL(p_cuMemcpyHtoD(d_x, ds->images, sizeof(float) * b * IN_AREA));
            launch_conv_fwd(d_x, d_y, d_w, b, 1, CONV1_CH, IN_SIDE);
            CU_CALL(p_cuMemcpyDtoH(host_out, d_y, sizeof(float) * b * CONV1_CH * IN_AREA));
        }
        const double gpu_e2e_s = (wall_time_sec() - t0) / iters;
        printf("[LAYER_BENCH] op=conv2d dim=%d cpu_ms=%.4f gpu_kern_ms=%.4f gpu_e2e_ms=%.4f\n",
               b, cpu_s * 1e3, gpu_k_s * 1e3, gpu_e2e_s * 1e3);
        fflush(stdout);
    }
    CU_CALL(p_cuMemFree(d_x)); CU_CALL(p_cuMemFree(d_y)); CU_CALL(p_cuMemFree(d_w));

    for (int i = 0; i < 6; i++) {
        const int n = mv_dims[i], iters = (n < 1024) ? 100 : 10;
        float *h_mat = zalloc_buf((size_t)n * n + n, sizeof(float));
        float *h_vec = zalloc_buf(n, sizeof(float));
        uint64_t rng = 900 + n;
        for (int j = 0; j < n * n + n; j++) h_mat[j] = (float)(xor64_uniform(&rng) - 0.5);
        for (int j = 0; j < n; j++) h_vec[j] = (float)xor64_uniform(&rng);

        CUdeviceptr d_mat = alloc_dev_floats((size_t)n * n + n);
        CUdeviceptr d_vec = alloc_dev_floats(n);
        CUdeviceptr d_res = alloc_dev_floats(n);
        CU_CALL(p_cuMemcpyHtoD(d_mat, h_mat, sizeof(float) * ((size_t)n * n + n)));
        CU_CALL(p_cuMemcpyHtoD(d_vec, h_vec, sizeof(float) * n));
        launch_fc_fwd(d_vec, d_res, d_mat, 1, n, n, 0);
        CU_CALL(p_cuCtxSynchronize());

        double t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) op_fc_matvec_fwd(h_vec, host_out, h_mat, n, n, 0);
        const double cpu_s = (wall_time_sec() - t0) / iters;

        t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) {
            launch_fc_fwd(d_vec, d_res, d_mat, 1, n, n, 0);
            CU_CALL(p_cuCtxSynchronize());
        }
        const double gpu_k_s = (wall_time_sec() - t0) / iters;

        t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) {
            CU_CALL(p_cuMemcpyHtoD(d_vec, h_vec, sizeof(float) * n));
            launch_fc_fwd(d_vec, d_res, d_mat, 1, n, n, 0);
            CU_CALL(p_cuMemcpyDtoH(host_out, d_res, sizeof(float) * n));
        }
        const double gpu_e2e_s = (wall_time_sec() - t0) / iters;
        printf("[LAYER_BENCH] op=matvec dim=%d cpu_ms=%.4f gpu_kern_ms=%.4f gpu_e2e_ms=%.4f\n",
               n, cpu_s * 1e3, gpu_k_s * 1e3, gpu_e2e_s * 1e3);
        fflush(stdout);
        CU_CALL(p_cuMemFree(d_mat)); CU_CALL(p_cuMemFree(d_vec)); CU_CALL(p_cuMemFree(d_res));
        free(h_mat); free(h_vec);
    }

    for (int i = 0; i < 5; i++) {
        const int n = mm_dims[i], iters = (n <= 512) ? 10 : 3;
        float *hA = zalloc_buf((size_t)n * n, sizeof(float));
        float *hB = zalloc_buf((size_t)n * n, sizeof(float));
        float *hC = zalloc_buf((size_t)n * n, sizeof(float));
        uint64_t rng = 2000 + n;
        for (int j = 0; j < n * n; j++) {
            hA[j] = (float)(xor64_uniform(&rng) - 0.5);
            hB[j] = (float)(xor64_uniform(&rng) - 0.5);
        }
        CUdeviceptr dA = alloc_dev_floats((size_t)n * n);
        CUdeviceptr dB = alloc_dev_floats((size_t)n * n);
        CUdeviceptr dC = alloc_dev_floats((size_t)n * n);
        CU_CALL(p_cuMemcpyHtoD(dA, hA, sizeof(float) * n * n));
        CU_CALL(p_cuMemcpyHtoD(dB, hB, sizeof(float) * n * n));
        invoke_tiled_gemm(dA, dB, dC, n, n, n);
        CU_CALL(p_cuCtxSynchronize());

        double t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) {
            memset(hC, 0, sizeof(float) * n * n);
            op_matmul_nn(hA, hB, hC, n, n, n);
        }
        const double cpu_s = (wall_time_sec() - t0) / iters;

        t0 = wall_time_sec();
        for (int r = 0; r < iters; r++) {
            invoke_tiled_gemm(dA, dB, dC, n, n, n);
            CU_CALL(p_cuCtxSynchronize());
        }
        const double gpu_k_s = (wall_time_sec() - t0) / iters;
        printf("[LAYER_BENCH] op=gemm dim=%d cpu_ms=%.4f gpu_kern_ms=%.4f gpu_e2e_ms=%.4f\n",
               n, cpu_s * 1e3, gpu_k_s * 1e3, gpu_k_s * 1e3);
        fflush(stdout);
        CU_CALL(p_cuMemFree(dA)); CU_CALL(p_cuMemFree(dB)); CU_CALL(p_cuMemFree(dC));
        free(hA); free(hB); free(hC);
    }
    free(host_out);
}

static void profile_gpu_gemm_kernels(const FMnistData *ds, const scalar_t *w)
{
    static const int batch_grid[] = {16, 64, 128, 256};
    GpuBuffers gb = init_gpu_buffers(ds, w, 256);
    CUdeviceptr d_col1 = alloc_dev_floats(256 * IN_AREA * (K_SIZE * K_SIZE));
    CUdeviceptr d_col2 = alloc_dev_floats(256 * POOL1_SIDE * POOL1_SIDE * (CONV1_CH * K_SIZE * K_SIZE));

    for (int i = 0; i < 4; i++) {
        const int bsz = batch_grid[i], trials = 10;
        int ks = K_SIZE, cin1 = 1, side1 = IN_SIDE, cin2 = CONV1_CH, side2 = POOL1_SIDE;
        int hw = IN_AREA, n_cls = NUM_CLASSES;
        double s_unfold = 0, s_conv_mm = 0, s_dense_mm = 0, s_pool = 0, s_loss_sgd = 0;

        gpu_step_forward(&gb, 0, bsz);
        gpu_step_backward(&gb, bsz);
        CU_CALL(p_cuCtxSynchronize());

        for (int t = 0; t < trials; t++) {
            double t0 = wall_time_sec();
            void *a_u1[] = {&gb.d_x, &d_col1, (void *)&bsz, &cin1, &side1, &ks};
            invoke_1d(FN_IM2COL, bsz * IN_AREA * K_SIZE * K_SIZE, a_u1);
            void *a_u2[] = {&gb.d_p1, &d_col2, (void *)&bsz, &cin2, &side2, &ks};
            invoke_1d(FN_IM2COL, bsz * POOL1_SIDE * POOL1_SIDE * CONV1_CH * K_SIZE * K_SIZE, a_u2);
            CU_CALL(p_cuCtxSynchronize());
            s_unfold += wall_time_sec() - t0;

            t0 = wall_time_sec();
            invoke_tiled_gemm(DEV_OFFSET(gb.d_w, OFF_CONV1), d_col1, gb.d_c1, CONV1_CH, bsz * IN_AREA, K_SIZE * K_SIZE);
            invoke_tiled_gemm(DEV_OFFSET(gb.d_w, OFF_CONV1), gb.d_dc1, d_col1, K_SIZE * K_SIZE, bsz * IN_AREA, CONV1_CH);
            invoke_tiled_gemm(DEV_OFFSET(gb.d_w, OFF_CONV2), d_col2, gb.d_c2, CONV2_CH, bsz * POOL1_SIDE * POOL1_SIDE, CONV1_CH * K_SIZE * K_SIZE);
            invoke_tiled_gemm(gb.d_dc2, d_col2, DEV_OFFSET(gb.d_g, OFF_CONV2), CONV2_CH, CONV1_CH * K_SIZE * K_SIZE, bsz * POOL1_SIDE * POOL1_SIDE);
            invoke_tiled_gemm(DEV_OFFSET(gb.d_w, OFF_CONV2), gb.d_dc2, d_col2, CONV1_CH * K_SIZE * K_SIZE, bsz * POOL1_SIDE * POOL1_SIDE, CONV2_CH);
            CU_CALL(p_cuCtxSynchronize());
            s_conv_mm += wall_time_sec() - t0;

            t0 = wall_time_sec();
            invoke_tiled_gemm(gb.d_p2, DEV_OFFSET(gb.d_w, OFF_FC1), gb.d_fc1, bsz, FC1_OUT, FC1_IN);
            invoke_tiled_gemm(gb.d_dfc1, DEV_OFFSET(gb.d_w, OFF_FC1), gb.d_dp2, bsz, FC1_IN, FC1_OUT);
            invoke_tiled_gemm(gb.d_dfc1, gb.d_p2, DEV_OFFSET(gb.d_g, OFF_FC1), FC1_OUT, FC1_IN, bsz);
            invoke_tiled_gemm(gb.d_fc1, DEV_OFFSET(gb.d_w, OFF_FC2), gb.d_logits, bsz, NUM_CLASSES, FC1_OUT);
            invoke_tiled_gemm(gb.d_dlogits, DEV_OFFSET(gb.d_w, OFF_FC2), gb.d_dfc1, bsz, FC1_OUT, NUM_CLASSES);
            invoke_tiled_gemm(gb.d_dlogits, gb.d_fc1, DEV_OFFSET(gb.d_g, OFF_FC2), NUM_CLASSES, FC1_OUT, bsz);
            CU_CALL(p_cuCtxSynchronize());
            s_dense_mm += wall_time_sec() - t0;

            t0 = wall_time_sec();
            launch_pool_fwd(gb.d_c1, gb.d_p1, gb.d_arg1, bsz, CONV1_CH, IN_SIDE);
            launch_pool_fwd(gb.d_c2, gb.d_p2, gb.d_arg2, bsz, CONV2_CH, POOL1_SIDE);
            launch_pool_bwd(gb.d_dp2, gb.d_arg2, gb.d_c2, gb.d_dc2, bsz, CONV2_CH, POOL1_SIDE);
            launch_pool_bwd(gb.d_dp1, gb.d_arg1, gb.d_c1, gb.d_dc1, bsz, CONV1_CH, IN_SIDE);
            CU_CALL(p_cuCtxSynchronize());
            s_pool += wall_time_sec() - t0;

            t0 = wall_time_sec();
            CUdeviceptr d_idx0 = gb.d_order;
            void *a_g[] = {&gb.d_imgs, &d_idx0, &gb.d_x, (void *)&bsz, &hw};
            invoke_1d(FN_GATHER, bsz * IN_AREA, a_g);
            void *a_s[] = {&gb.d_logits, &gb.d_lbls, &d_idx0, &gb.d_dlogits, &gb.d_losses, &gb.d_hits, (void *)&bsz, &n_cls};
            invoke_1d(FN_SOFTMAX, bsz, a_s);
            gpu_step_sgd(&gb, bsz, 0.02f, 0.9f);
            CU_CALL(p_cuCtxSynchronize());
            s_loss_sgd += wall_time_sec() - t0;
        }

        printf("[GPU_PROF] batch=%d conv_mm_ms=%.4f dense_mm_ms=%.4f unfold_ms=%.4f act_pool_ms=%.4f loss_sgd_ms=%.4f\n",
               bsz, (s_conv_mm / trials) * 1e3, (s_dense_mm / trials) * 1e3, (s_unfold / trials) * 1e3,
               (s_pool / trials) * 1e3, (s_loss_sgd / trials) * 1e3);
        fflush(stdout);
    }

    CU_CALL(p_cuMemFree(d_col1));
    CU_CALL(p_cuMemFree(d_col2));
}

int main(int argc, char **argv)
{
    CnnConfig cfg = parse_cli_args(argc, argv);
    FMnistData ds = alloc_fmnist_data();
    scalar_t *w = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
    load_fmnist_idx(cfg.data_dir, &ds);
    init_cnn_weights(w, cfg.seed);

    bind_cuda_driver_and_nvrtc();
    CUdevice dev;
    CUcontext ctx;
    char gpu_name[128];
    int cc_maj = 0, cc_min = 0;
    CU_CALL(p_cuInit(0));
    CU_CALL(p_cuDeviceGet(&dev, 0));
    CU_CALL(p_cuDeviceGetName(gpu_name, sizeof gpu_name, dev));
    CU_CALL(p_cuDeviceComputeCapability(&cc_maj, &cc_min, dev));
    CU_CALL(p_cuCtxCreate(&ctx, 0, dev));

    const double t0 = wall_time_sec();
    CUmodule mod = compile_and_load_ptx("src/gpu_kernels.cu", dev);
    printf("[GPU_INFO] device=\"%s\" sm=%d.%d jit_sec=%.3f\n", gpu_name, cc_maj, cc_min, wall_time_sec() - t0);
    fflush(stdout);

    int rc = 0;
    if (cfg.flag_microbench) {
        run_layer_microbenchmarks(&ds, w);
    } else if (cfg.flag_prof_gemm) {
        profile_gpu_gemm_kernels(&ds, w);
    } else {
        GpuBuffers gb = init_gpu_buffers(&ds, w, cfg.batch_size);
        size_t free_b = 0, tot_b = 0;
        CU_CALL(p_cuMemGetInfo(&free_b, &tot_b));
        printf("[GPU_MEM] free_gb=%.2f total_gb=%.2f\n", free_b / 1073741824.0, tot_b / 1073741824.0);
        if (cfg.flag_verify) rc = verify_gpu_vs_cpu_grad(&cfg, &gb, &ds, w);
        else run_gpu_training(&cfg, &gb);
    }

    CU_CALL(p_cuModuleUnload(mod));
    CU_CALL(p_cuCtxDestroy(ctx));
    free_fmnist_data(&ds);
    free(w);
    return rc;
}
