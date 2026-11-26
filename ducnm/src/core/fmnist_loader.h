/*
 * Fashion-MNIST IDX reader, batch gradient helpers, and CLI option parser.
 */
#ifndef FMNIST_LOADER_H
#define FMNIST_LOADER_H

#include "lenet_gemm.h"
#include "sys_timer.h"

enum {
    TOTAL_SAMPLES = 10000,
    TRAIN_LIMIT   = 7000,
    VAL_COUNT     = 1000,
    VAL_OFFSET    = TRAIN_LIMIT,
    TEST_OFFSET   = VAL_OFFSET + VAL_COUNT,
    TEST_COUNT    = TOTAL_SAMPLES - TEST_OFFSET,
    MAX_STALE_GAP = 64
};

typedef struct {
    const char *data_dir, *strategy, *collective;
    int epochs, batch_size, train_samples, seed, workers;
    scalar_t lr, momentum;
    int flag_microbench, flag_verify, flag_prof_gemm, flag_comm_bench, flag_naive_conv;
    int staleness;
} CnnConfig;

typedef struct {
    scalar_t *images;
    int *labels;
} FMnistData;

static inline CnnConfig parse_cli_args(int argc, char **argv)
{
    CnnConfig cfg = {
        "data", "sync", "ring",
        3, 64, TRAIN_LIMIT, 2026, 1,
        (scalar_t)0.02, (scalar_t)0.9,
        0, 0, 0, 0, 0, 0
    };
    int custom_lr = 0, custom_mom = 0;
    disable_power_throttling();

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--bench")) { cfg.flag_microbench = 1; continue; }
        else if (!strcmp(arg, "--verify") || !strcmp(arg, "--check")) { cfg.flag_verify = 1; continue; }
        else if (!strcmp(arg, "--profile-gemm")) { cfg.flag_prof_gemm = 1; continue; }
        else if (!strcmp(arg, "--bench-comm")) { cfg.flag_comm_bench = 1; continue; }
        else if (!strcmp(arg, "--naive-conv") || !strcmp(arg, "--direct")) { cfg.flag_naive_conv = 1; continue; }
        if (i + 1 == argc) fatal_err("Missing value for argument:", arg);
        const char *val = argv[++i];
        if (!strcmp(arg, "--data")) cfg.data_dir = val;
        else if (!strcmp(arg, "--strategy") || !strcmp(arg, "--mode")) cfg.strategy = val;
        else if (!strcmp(arg, "--collective") || !strcmp(arg, "--allreduce")) cfg.collective = val;
        else if (!strcmp(arg, "--epochs")) cfg.epochs = atoi(val);
        else if (!strcmp(arg, "--batch")) cfg.batch_size = atoi(val);
        else if (!strcmp(arg, "--train")) cfg.train_samples = atoi(val);
        else if (!strcmp(arg, "--seed")) cfg.seed = atoi(val);
        else if (!strcmp(arg, "--workers") || !strcmp(arg, "--ranks") || !strcmp(arg, "-n")) cfg.workers = atoi(val);
        else if (!strcmp(arg, "--staleness") || !strcmp(arg, "--delay")) cfg.staleness = atoi(val);
        else if (!strcmp(arg, "--lr")) { cfg.lr = (scalar_t)atof(val); custom_lr = 1; }
        else if (!strcmp(arg, "--momentum")) { cfg.momentum = (scalar_t)atof(val); custom_mom = 1; }
        else fatal_err("Unrecognized CLI option:", arg);
    }
    if (!strcmp(cfg.strategy, "async")) {
        if (!custom_lr) cfg.lr = (scalar_t)0.1;
        if (!custom_mom) cfg.momentum = 0;
    }
    return cfg;
}

static inline FMnistData alloc_fmnist_data(void)
{
    FMnistData ds = {
        zalloc_buf((size_t)TOTAL_SAMPLES * IN_AREA, sizeof(scalar_t)),
        zalloc_buf(TOTAL_SAMPLES, sizeof(int))
    };
    return ds;
}

static inline void free_fmnist_data(FMnistData *ds)
{
    free(ds->images);
    free(ds->labels);
}

static inline void load_binary_blob(const char *folder, const char *filename, unsigned char *dst, size_t expected_bytes)
{
    char fullpath[1024];
    snprintf(fullpath, sizeof fullpath, "%s/%s", folder, filename);
    FILE *fp = fopen(fullpath, "rb");
    if (!fp) fatal_err("Cannot open dataset file:", fullpath);
    if (fread(dst, 1, expected_bytes, fp) != expected_bytes) fatal_err("Truncated dataset file:", fullpath);
    fclose(fp);
}

static inline void load_fmnist_idx(const char *folder, FMnistData *ds)
{
    const size_t img_bytes = 16 + (size_t)TOTAL_SAMPLES * IN_AREA;
    const size_t lbl_bytes = 8 + TOTAL_SAMPLES;
    unsigned char *raw_img = zalloc_buf(img_bytes, 1);
    unsigned char *raw_lbl = zalloc_buf(lbl_bytes, 1);

    load_binary_blob(folder, "t10k-images-idx3-ubyte", raw_img, img_bytes);
    load_binary_blob(folder, "t10k-labels-idx1-ubyte", raw_lbl, lbl_bytes);
    if (raw_img[2] != 8 || raw_img[3] != 3 || raw_lbl[2] != 8 || raw_lbl[3] != 1)
        fatal_err("Invalid IDX header in folder:", folder);

    for (int i = 0; i < TOTAL_SAMPLES * IN_AREA; i++)
        ds->images[i] = (scalar_t)((raw_img[16 + i] / 255.0 - 0.2860) / 0.3530);
    for (int i = 0; i < TOTAL_SAMPLES; i++)
        ds->labels[i] = raw_lbl[8 + i];

    free(raw_img);
    free(raw_lbl);
}

static inline scalar_t accumulate_minibatch_naive(const scalar_t *w, const FMnistData *ds, const int *indices, int count, scalar_t *grad)
{
    scalar_t sum_loss = 0;
    memset(grad, 0, sizeof(scalar_t) * TOTAL_PARAMS);
    for (int i = 0; i < count; i++)
        sum_loss += backward_sample_naive(w, ds->images + indices[i] * IN_AREA, ds->labels[indices[i]], grad);
    return sum_loss;
}

static inline scalar_t accumulate_minibatch_im2col(const scalar_t *w, const FMnistData *ds, const int *indices, int count, scalar_t *grad, GemmProfile *prof)
{
    scalar_t sum_loss = 0;
    memset(grad, 0, sizeof(scalar_t) * TOTAL_PARAMS);
    for (int i = 0; i < count; i++)
        sum_loss += backward_sample_im2col(w, ds->images + indices[i] * IN_AREA, ds->labels[indices[i]], grad, prof);
    return sum_loss;
}

static inline double eval_classification_acc(const scalar_t *w, const FMnistData *ds, int start_idx, int count)
{
    int hits = 0;
    for (int i = start_idx; i < start_idx + count; i++)
        hits += (predict_sample_class(w, ds->images + i * IN_AREA) == ds->labels[i]);
    return 100.0 * hits / count;
}

static inline void log_epoch_metrics(int ep, double loss, double val_acc, double test_acc, double sec)
{
    printf("[EPOCH] ep=%d loss=%.4f val_acc=%.2f test_acc=%.2f elapsed_s=%.4f\n", ep, loss, val_acc, test_acc, sec);
    fflush(stdout);
}

#endif
