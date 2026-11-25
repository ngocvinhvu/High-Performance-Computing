/* bench_seq.c - sequential C version of the benchmarks (1 thread, no BLAS, no OpenMP, no library).

   Same experiments and CSV columns as the PyTorch code in ../code, written by hand:
     conv  (paper Fig 1-2)  caffe   per image: im2col, then GEMM W(96x121) x col(121x2916) + bias  [main series]
                            direct  the convolution loops themselves, no im2col                    [extension]
     fc    (paper Fig 3-4)  matvec_naive  y = A x, one accumulator (a dependency chain)
                            matvec        y = A x, 32 independent accumulators                    [main series]
           (gemm)           gemm_naive (i,j,k loops), gemm_ikj (loop interchange),
                            gemm (cache + register blocking, packed B panel)                       [main series]
     bw    memory bandwidth: sum, dot, copy, triad (STREAM-like) and matvec, arrays of n*n floats
     fcb   FC layer with B input vectors, (n x n)(n x B): mm1 = matvec, mm16..mm128 = gemm
     step  caffe conv split into im2col only / GEMM only / both

   Build (Linux, Kaggle or WSL):  gcc -O3 -march=native -std=c11 -fno-tree-loop-distribute-patterns \
                                      -o bench_seq bench_seq.c -lm
   -fno-tree-loop-distribute-patterns keeps gcc from turning the copy loop into a call to memcpy.
   No -ffast-math: the compiler may vectorize only where the code itself has independent sums.

   Usage:  ./bench_seq --check-only
           ./bench_seq --machine kaggle --outdir results/kaggle/seq1 --order-seed 1 [--quick] */
#include "common.h"

static volatile float g_sink; /* results go here so that no computation can be dropped */

/* ================================================================== kernels */

/* ---- GEMM, row-major: C (M x N) = A (M x K) * B (K x N) + bias[i] (bias may be NULL) */

static NOINLINE void gemm_naive(int M, int N, int K, const float *RESTRICT A, const float *RESTRICT B,
                                const float *RESTRICT bias, float *RESTRICT C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[(long)i * K + k] * B[(long)k * N + j]; /* B read down a column */
            C[(long)i * N + j] = s + (bias ? bias[i] : 0.f);
        }
}

static NOINLINE void gemm_ikj(int M, int N, int K, const float *RESTRICT A, const float *RESTRICT B,
                              const float *RESTRICT bias, float *RESTRICT C) {
    for (int i = 0; i < M; i++) {
        float *RESTRICT c = C + (long)i * N;
        const float b0 = bias ? bias[i] : 0.f;
        for (int j = 0; j < N; j++) c[j] = b0;
        for (int k = 0; k < K; k++) {
            const float a = A[(long)i * K + k];
            const float *RESTRICT b = B + (long)k * N;
            for (int j = 0; j < N; j++) c[j] += a * b[j]; /* contiguous rows: vectorizable */
        }
    }
}

/* Blocked GEMM. For each block of KC values of k and each panel of NR = 16 columns, the panel of B is
   copied into a contiguous buffer (so it stays in L1 whatever N is), then every group of MR = 4 rows
   is computed by the micro-kernel, which keeps a 4 x 16 tile of C in registers over the k loop. */
enum { MR = 4, NR = 16, KC = 256 };

static inline void micro_4x16(int kc, int K, int N, const float *RESTRICT A, const float *RESTRICT Bp,
                              float *RESTRICT C) {
    float acc[MR][NR];
    for (int r = 0; r < MR; r++)
        for (int j = 0; j < NR; j++) acc[r][j] = C[(long)r * N + j];
    for (int k = 0; k < kc; k++) {
        const float *RESTRICT b = Bp + k * NR;
        const float a0 = A[k], a1 = A[K + k], a2 = A[2L * K + k], a3 = A[3L * K + k];
        for (int j = 0; j < NR; j++) {
            const float bj = b[j];
            acc[0][j] += a0 * bj;
            acc[1][j] += a1 * bj;
            acc[2][j] += a2 * bj;
            acc[3][j] += a3 * bj;
        }
    }
    for (int r = 0; r < MR; r++)
        for (int j = 0; j < NR; j++) C[(long)r * N + j] = acc[r][j];
}

static NOINLINE void gemm_blocked(int M, int N, int K, const float *RESTRICT A, const float *RESTRICT B,
                                  const float *RESTRICT bias, float *RESTRICT C) {
    static float Bp[KC * NR]; /* 16 KB packed panel of B */
    const int Mf = M - M % MR, Nf = N - N % NR;
    for (int i = 0; i < M; i++) {
        const float b0 = bias ? bias[i] : 0.f;
        float *RESTRICT c = C + (long)i * N;
        for (int j = 0; j < N; j++) c[j] = b0;
    }
    for (int k0 = 0; k0 < K; k0 += KC) {
        const int kc = (K - k0 < KC) ? K - k0 : KC;
        for (int j0 = 0; j0 < Nf; j0 += NR) {
            for (int k = 0; k < kc; k++)
                for (int j = 0; j < NR; j++) Bp[k * NR + j] = B[(long)(k0 + k) * N + j0 + j];
            for (int i0 = 0; i0 < Mf; i0 += MR)
                micro_4x16(kc, K, N, A + (long)i0 * K + k0, Bp, C + (long)i0 * N + j0);
        }
        /* edges: the last N % 16 columns of every row, and all columns of the last M % 4 rows */
        for (int i = 0; i < M; i++) {
            const int jstart = (i < Mf) ? Nf : 0;
            float *RESTRICT c = C + (long)i * N;
            if (jstart >= N) continue;
            for (int k = k0; k < k0 + kc; k++) {
                const float a = A[(long)i * K + k];
                const float *RESTRICT b = B + (long)k * N;
                for (int j = jstart; j < N; j++) c[j] += a * b[j];
            }
        }
    }
}

/* ---- convolution */

/* Caffe: one image at a time, im2col into one reused column buffer, then one GEMM with the bias. */
static NOINLINE void conv_caffe(int nb, const float *x, const float *w, const float *bias, float *col, float *out) {
    for (int b = 0; b < nb; b++) {
        im2col_cpu(x + b * IMG_ELEMS, col);
        gemm_blocked(C_OUT, PATCHES, PATCH_LEN, w, col, bias, out + b * OUT_ELEMS);
    }
}

/* Direct convolution, loops ordered so that the innermost loop runs along an output row. */
static NOINLINE void conv_direct(int nb, const float *RESTRICT x, const float *RESTRICT w,
                                 const float *RESTRICT bias, float *RESTRICT out) {
    for (int b = 0; b < nb; b++)
        for (int oc = 0; oc < C_OUT; oc++) {
            float *RESTRICT o = out + b * OUT_ELEMS + (long)oc * PATCHES;
            for (int p = 0; p < PATCHES; p++) o[p] = bias[oc];
            for (int c = 0; c < C_IN; c++)
                for (int kh = 0; kh < KS; kh++)
                    for (int kw = 0; kw < KS; kw++) {
                        const float wv = w[((oc * C_IN + c) * KS + kh) * KS + kw];
                        for (int oh = 0; oh < OUT_HW; oh++) {
                            const float *RESTRICT src =
                                x + b * IMG_ELEMS + (long)c * IMG * IMG + (long)(oh * STRIDE + kh) * IMG + kw;
                            float *RESTRICT orow = o + oh * OUT_HW;
                            for (int ow = 0; ow < OUT_HW; ow++) orow[ow] += wv * src[ow * STRIDE];
                        }
                    }
        }
}

/* ---- matrix-vector and STREAM-like kernels */

static NOINLINE void matvec_naive(int n, const float *RESTRICT A, const float *RESTRICT x, float *RESTRICT y) {
    for (int i = 0; i < n; i++) {
        float s = 0.f; /* one accumulator: each add waits for the previous one */
        for (int k = 0; k < n; k++) s += A[(long)i * n + k] * x[k];
        y[i] = s;
    }
}

enum { NACC = 32 }; /* independent partial sums: enough to hide the add latency and to vectorize */

static inline float dot32(const float *RESTRICT a, const float *RESTRICT b, long n) {
    float s[NACC] = {0}, t = 0.f;
    long k = 0;
    for (; k + NACC <= n; k += NACC)
        for (int u = 0; u < NACC; u++) s[u] += a[k + u] * b[k + u];
    for (; k < n; k++) t += a[k] * b[k];
    for (int u = 0; u < NACC; u++) t += s[u];
    return t;
}

static NOINLINE void matvec(int n, const float *RESTRICT A, const float *RESTRICT x, float *RESTRICT y) {
    for (int i = 0; i < n; i++) y[i] = dot32(A + (long)i * n, x, n);
}

static NOINLINE float sum_seq(const float *RESTRICT x, long n) {
    float s[NACC] = {0}, t = 0.f;
    long k = 0;
    for (; k + NACC <= n; k += NACC)
        for (int u = 0; u < NACC; u++) s[u] += x[k + u];
    for (; k < n; k++) t += x[k];
    for (int u = 0; u < NACC; u++) t += s[u];
    return t;
}

static NOINLINE float dot_seq(const float *RESTRICT x, const float *RESTRICT y, long n) { return dot32(x, y, n); }

static NOINLINE void copy_seq(const float *RESTRICT x, float *RESTRICT y, long n) {
    for (long i = 0; i < n; i++) y[i] = x[i];
}

#define TRIAD_S 3.0f
static NOINLINE void triad_seq(const float *RESTRICT b, const float *RESTRICT c, float *RESTRICT o, long n) {
    for (long i = 0; i < n; i++) o[i] = b[i] + TRIAD_S * c[i];
}

/* FC layer with B input vectors: Y (n x B) = A (n x n) X (n x B); B = 1 is the matvec. */
static void fc_batch(int n, int B, const float *A, const float *X, float *Y) {
    if (B == 1)
        matvec(n, A, X, Y);
    else
        gemm_blocked(n, B, n, A, X, NULL, Y);
}

/* ================================================================== timing */

typedef struct Job Job;
struct Job {
    void (*run)(Job *);
    int n, nb, B;
    long N;
    const float *x, *y, *w, *bias, *a;
    float *col, *out;
};

static void run_caffe(Job *j) { conv_caffe(j->nb, j->x, j->w, j->bias, j->col, j->out); }
static void run_direct(Job *j) { conv_direct(j->nb, j->x, j->w, j->bias, j->out); }
static void run_matvec_naive(Job *j) { matvec_naive(j->n, j->a, j->x, j->out); }
static void run_matvec(Job *j) { matvec(j->n, j->a, j->x, j->out); }
static void run_gemm_naive(Job *j) { gemm_naive(j->n, j->n, j->n, j->a, j->x, NULL, j->out); }
static void run_gemm_ikj(Job *j) { gemm_ikj(j->n, j->n, j->n, j->a, j->x, NULL, j->out); }
static void run_gemm(Job *j) { gemm_blocked(j->n, j->n, j->n, j->a, j->x, NULL, j->out); }
static void run_sum(Job *j) { g_sink = sum_seq(j->x, j->N); }
static void run_dot(Job *j) { g_sink = dot_seq(j->x, j->y, j->N); }
static void run_copy(Job *j) { copy_seq(j->x, j->out, j->N); }
static void run_triad(Job *j) { triad_seq(j->x, j->y, j->out, j->N); }
static void run_fcb(Job *j) { fc_batch(j->n, j->B, j->a, j->x, j->out); }
static void run_step_im2col(Job *j) {
    for (int b = 0; b < j->nb; b++) im2col_cpu(j->x + b * IMG_ELEMS, j->col);
}
static void run_step_gemm(Job *j) {
    for (int b = 0; b < j->nb; b++) gemm_blocked(C_OUT, PATCHES, PATCH_LEN, j->w, j->col, j->bias, j->out + b * OUT_ELEMS);
}

/* warmup untimed runs, then repeat timed runs (wall clock, ms). */
static Stats bench(Job *j, int warmup, int repeat) {
    double *ms = dalloc(repeat), t0, t1;
    Stats s;
    for (int i = 0; i < warmup; i++) j->run(j);
    t0 = epoch_s();
    for (int i = 0; i < repeat; i++) {
        double start = now_s();
        j->run(j);
        ms[i] = (now_s() - start) * 1e3;
    }
    t1 = epoch_s();
    s = make_stats(ms, ms, repeat, t0, t1);
    free(ms);
    return s;
}

static float *random_array(long n, unsigned long long seed) {
    float *p = falloc(n);
    rng_seed(seed);
    fill_random(p, n);
    return p;
}

static float *zero_array(long n) {
    float *p = falloc(n);
    fzero(p, n);
    return p;
}

static void keep(const float *p, long n) { g_sink = p[0] + p[n / 2] + p[n - 1]; }

/* ================================================================== experiments */

static void exp_conv(const Args *a) {
    static const char *impls[] = {"caffe", "direct"};
    FILE *csv = csv_open(a->outdir, "conv.csv");
    float *w = random_array((long)C_OUT * PATCH_LEN, 1), *bias = random_array(C_OUT, 2);
    float *col = zero_array(COL_ELEMS);
    for (int ii = 0; ii < 2; ii++) {
        long list[MAX_LIST];
        int cnt = parse_list(ii ? a->direct_batches : a->batches, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        for (int t = 0; t < cnt; t++) {
            Job j;
            Stats s;
            int nb = (int)list[t];
            memset(&j, 0, sizeof j);
            j.nb = nb, j.w = w, j.bias = bias, j.col = col;
            j.x = random_array(nb * IMG_ELEMS, 100 + nb);
            j.out = zero_array(nb * OUT_ELEMS);
            j.run = ii ? run_direct : run_caffe;
            s = bench(&j, a->warmup, a->repeat);
            keep(j.out, nb * OUT_ELEMS);
            csv_row(csv, "conv", impls[ii], "cpu", 1, nb, &s, a->warmup, a->repeat, FLOPS_PER_IMAGE * nb,
                    4.0 * nb * (IMG_ELEMS + OUT_ELEMS), a->machine);
            say("conv  cpu %-13s batch=%-5d %11.3f ms +- %.3f  (%.3f ms/img, %.1f GFLOP/s)", impls[ii], nb, s.mean,
                s.std, s.mean / nb, FLOPS_PER_IMAGE * nb / (s.mean * 1e6));
            ffree((float *)j.x);
            ffree(j.out);
        }
    }
    ffree(w), ffree(bias), ffree(col);
    fclose(csv);
}

static void exp_fc(const Args *a) {
    static const char *impls[] = {"matvec_naive", "matvec"};
    FILE *csv = csv_open(a->outdir, "fc.csv");
    for (int ii = 0; ii < 2; ii++) {
        long list[MAX_LIST];
        int cnt = parse_list(a->mv_sizes, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        for (int t = 0; t < cnt; t++) {
            Job j;
            Stats s;
            int n = (int)list[t];
            memset(&j, 0, sizeof j);
            j.n = n;
            j.a = random_array((long)n * n, 200 + n);
            j.x = random_array(n, 300 + n);
            j.out = zero_array(n);
            j.run = ii ? run_matvec : run_matvec_naive;
            s = bench(&j, a->warmup, a->repeat);
            keep(j.out, n);
            csv_row(csv, "fc", impls[ii], "cpu", 1, n, &s, a->warmup, a->repeat, 2.0 * n * n,
                    4.0 * ((double)n * n + 2.0 * n), a->machine);
            say("fc    cpu %-13s n=%-6d %11.3f ms +- %.3f  (%.2f GFLOP/s, %.2f GB/s)", impls[ii], n, s.mean, s.std,
                2.0 * n * n / (s.mean * 1e6), 4.0 * n * n / (s.mean * 1e6));
            ffree((float *)j.a), ffree((float *)j.x), ffree(j.out);
        }
    }
    fclose(csv);
}

static void exp_gemm(const Args *a) {
    static const char *impls[] = {"gemm_naive", "gemm_ikj", "gemm"};
    FILE *csv = csv_open(a->outdir, "fc.csv");
    for (int ii = 0; ii < 3; ii++) {
        long list[MAX_LIST];
        int cnt = parse_list(ii == 0 ? a->gemm_naive_sizes : a->gemm_sizes, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        for (int t = 0; t < cnt; t++) {
            Job j;
            Stats s;
            int n = (int)list[t];
            double flops = 2.0 * n * (double)n * n;
            memset(&j, 0, sizeof j);
            j.n = n;
            j.a = random_array((long)n * n, 400 + n);
            j.x = random_array((long)n * n, 500 + n);
            j.out = zero_array((long)n * n);
            j.run = ii == 0 ? run_gemm_naive : ii == 1 ? run_gemm_ikj : run_gemm;
            s = bench(&j, a->warmup, a->repeat);
            keep(j.out, (long)n * n);
            csv_row(csv, "fc", impls[ii], "cpu", 1, n, &s, a->warmup, a->repeat, flops, 12.0 * n * n, a->machine);
            say("gemm  cpu %-13s n=%-6d %11.3f ms +- %.3f  (%.2f GFLOP/s)", impls[ii], n, s.mean, s.std,
                flops / (s.mean * 1e6));
            ffree((float *)j.a), ffree((float *)j.x), ffree(j.out);
        }
    }
    fclose(csv);
}

static void exp_bw(const Args *a) {
    static const char *kernels[] = {"sum", "dot", "copy", "triad", "matvec"};
    static const double bytes_per[] = {4, 8, 8, 12, 4}, flops_per[] = {1, 2, 0, 2, 2};
    FILE *csv = csv_open(a->outdir, "bw.csv");
    for (int kk = 0; kk < 5; kk++) {
        long list[MAX_LIST];
        int cnt = parse_list(a->bw_sizes, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        for (int t = 0; t < cnt; t++) {
            Job j;
            Stats s;
            int n = (int)list[t];
            long N = (long)n * n;
            memset(&j, 0, sizeof j);
            j.n = n, j.N = N;
            if (kk == 4) { /* matvec over the same n*n floats */
                j.a = random_array(N, 600 + n);
                j.x = random_array(n, 700 + n);
                j.out = zero_array(n);
                j.run = run_matvec;
            } else {
                j.x = random_array(N, 600 + n);
                if (kk == 1 || kk == 3) j.y = random_array(N, 700 + n);
                if (kk >= 2) j.out = zero_array(N);
                j.run = kk == 0 ? run_sum : kk == 1 ? run_dot : kk == 2 ? run_copy : run_triad;
            }
            s = bench(&j, a->warmup, a->repeat);
            if (j.out) keep(j.out, kk == 4 ? n : N);
            csv_row(csv, "bw", kernels[kk], "cpu", 1, n, &s, a->warmup, a->repeat, flops_per[kk] * N,
                    bytes_per[kk] * N, a->machine);
            say("bw    cpu %-13s n=%-6d %11.3f ms +- %.3f  (%.2f GB/s)", kernels[kk], n, s.mean, s.std,
                bytes_per[kk] * N / (s.mean * 1e6));
            if (j.a) ffree((float *)j.a);
            if (j.x) ffree((float *)j.x);
            if (j.y) ffree((float *)j.y);
            if (j.out) ffree(j.out);
        }
    }
    fclose(csv);
}

static void exp_fcb(const Args *a) {
    long sizes[MAX_LIST], bs[MAX_LIST], order[MAX_LIST];
    int ns = parse_list(a->fc_sizes, sizes, MAX_LIST), nB = parse_list(a->fc_b, bs, MAX_LIST), cnt = 0;
    FILE *csv = csv_open(a->outdir, "fcb.csv");
    if (ns * nB > MAX_LIST) die("too many (n, B) pairs");
    for (int i = 0; i < ns * nB; i++) order[cnt++] = i;
    shuffle(order, cnt, a->order_seed);
    for (int t = 0; t < cnt; t++) {
        Job j;
        Stats s;
        char impl[32];
        int n = (int)sizes[order[t] / nB], B = (int)bs[order[t] % nB];
        double flops = 2.0 * n * (double)n * B;
        memset(&j, 0, sizeof j);
        j.n = n, j.B = B;
        j.a = random_array((long)n * n, 800 + n);
        j.x = random_array((long)n * B, 900 + B);
        j.out = zero_array((long)n * B);
        j.run = run_fcb;
        s = bench(&j, a->warmup, a->repeat);
        keep(j.out, (long)n * B);
        snprintf(impl, sizeof impl, "mm%d", B);
        csv_row(csv, "fcb", impl, "cpu", 1, n, &s, a->warmup, a->repeat, flops, 4.0 * n * n, a->machine);
        say("fcb   cpu B=%-4d n=%-6d %11.3f ms +- %.3f  (%.2f GFLOP/s)", B, n, s.mean, s.std, flops / (s.mean * 1e6));
        ffree((float *)j.a), ffree((float *)j.x), ffree(j.out);
    }
    fclose(csv);
}

static void exp_step(const Args *a) {
    static const char *steps[] = {"im2col", "gemm", "caffe"};
    FILE *csv = csv_open(a->outdir, "step.csv");
    float *w = random_array((long)C_OUT * PATCH_LEN, 1), *bias = random_array(C_OUT, 2);
    float *col = zero_array(COL_ELEMS);
    for (int ss = 0; ss < 3; ss++) {
        long list[MAX_LIST];
        int cnt = parse_list(a->step_batches, list, MAX_LIST);
        shuffle(list, cnt, a->order_seed);
        for (int t = 0; t < cnt; t++) {
            Job j;
            Stats s;
            int nb = (int)list[t];
            memset(&j, 0, sizeof j);
            j.nb = nb, j.w = w, j.bias = bias, j.col = col;
            j.x = random_array(nb * IMG_ELEMS, 100 + nb);
            j.out = zero_array(nb * OUT_ELEMS);
            im2col_cpu(j.x, col); /* GEMM-only needs valid columns */
            j.run = ss == 0 ? run_step_im2col : ss == 1 ? run_step_gemm : run_caffe;
            s = bench(&j, a->warmup, a->repeat);
            keep(j.out, nb * OUT_ELEMS);
            csv_row(csv, "step", steps[ss], "cpu", 1, nb, &s, a->warmup, a->repeat, ss ? FLOPS_PER_IMAGE * nb : 0, 0,
                    a->machine);
            say("step  cpu %-13s batch=%-5d %11.3f ms +- %.3f  (%.1f us/img)", steps[ss], nb, s.mean, s.std,
                s.mean / nb * 1e3);
            ffree((float *)j.x), ffree(j.out);
        }
    }
    ffree(w), ffree(bias), ffree(col);
    fclose(csv);
}

/* ================================================================== correctness */

static void check_gemm_shape(int M, int N, int K, int with_bias) {
    float *A = random_array((long)M * K, 11), *B = random_array((long)K * N, 12), *bias = random_array(M, 13);
    float *C = falloc((long)M * N);
    double *ref = dalloc((long)M * N);
    char what[96];
    const float *bb = with_bias ? bias : NULL;
    ref_gemm(M, N, K, A, B, bb, ref);
    gemm_naive(M, N, K, A, B, bb, C);
    snprintf(what, sizeof what, "gemm_naive   %dx%dx%d%s", M, N, K, with_bias ? " +bias" : "");
    check(what, rel_err(C, ref, (long)M * N), 1e-4);
    gemm_ikj(M, N, K, A, B, bb, C);
    snprintf(what, sizeof what, "gemm_ikj     %dx%dx%d%s", M, N, K, with_bias ? " +bias" : "");
    check(what, rel_err(C, ref, (long)M * N), 1e-4);
    gemm_blocked(M, N, K, A, B, bb, C);
    snprintf(what, sizeof what, "gemm         %dx%dx%d%s", M, N, K, with_bias ? " +bias" : "");
    check(what, rel_err(C, ref, (long)M * N), 1e-4);
    ffree(A), ffree(B), ffree(bias), ffree(C);
    free(ref);
}

static int run_checks(void) {
    g_fail = 0;
    /* GEMM: odd shapes (edges of the 4 x 16 tiles), more than one k block (K > 256), the conv shape */
    check_gemm_shape(37, 53, 29, 1);
    check_gemm_shape(5, 17, 3, 0);
    check_gemm_shape(64, 80, 300, 0);
    check_gemm_shape(C_OUT, PATCHES, PATCH_LEN, 1);
    { /* convolution, 2 images */
        const int nb = 2;
        float *x = random_array(nb * IMG_ELEMS, 21), *w = random_array((long)C_OUT * PATCH_LEN, 22);
        float *bias = random_array(C_OUT, 23), *col = falloc(COL_ELEMS), *out = falloc(nb * OUT_ELEMS);
        double *ref = dalloc(nb * OUT_ELEMS);
        ref_conv(nb, x, w, bias, ref);
        conv_caffe(nb, x, w, bias, col, out);
        check("conv caffe (im2col + gemm), 2 images", rel_err(out, ref, nb * OUT_ELEMS), 1e-4);
        conv_direct(nb, x, w, bias, out);
        check("conv direct, 2 images", rel_err(out, ref, nb * OUT_ELEMS), 1e-4);
        ffree(x), ffree(w), ffree(bias), ffree(col), ffree(out);
        free(ref);
    }
    { /* matvec */
        static const int sizes[] = {1, 37, 300, 1000};
        for (int t = 0; t < 4; t++) {
            int n = sizes[t];
            float *A = random_array((long)n * n, 31), *x = random_array(n, 32), *y = falloc(n);
            double *ref = dalloc(n);
            char what[64];
            ref_matvec(n, A, x, ref);
            matvec_naive(n, A, x, y);
            snprintf(what, sizeof what, "matvec_naive n=%d", n);
            check(what, rel_err(y, ref, n), 1e-4);
            matvec(n, A, x, y);
            snprintf(what, sizeof what, "matvec       n=%d", n);
            check(what, rel_err(y, ref, n), 1e-4);
            ffree(A), ffree(x), ffree(y);
            free(ref);
        }
    }
    { /* STREAM-like kernels on an odd length */
        const long N = 1000003;
        float *x = random_array(N, 41), *y = random_array(N, 42), *o = falloc(N);
        double s = 0, d = 0, sabs = 0, dabs = 0, got;
        double *ref = dalloc(N);
        for (long i = 0; i < N; i++) {
            s += x[i], d += (double)x[i] * y[i];
            sabs += fabs(x[i]), dabs += fabs((double)x[i] * y[i]);
        }
        /* error relative to the sum of magnitudes (the scale of float rounding in a long sum) */
        got = sum_seq(x, N);
        check("sum N=1000003", fabs(got - s) / sabs, 1e-6);
        got = dot_seq(x, y, N);
        check("dot N=1000003", fabs(got - d) / dabs, 1e-6);
        copy_seq(x, o, N);
        check("copy N=1000003 (exact)", memcmp(x, o, (size_t)N * sizeof(float)) ? 1.0 : 0.0, 0.0);
        triad_seq(x, y, o, N);
        for (long i = 0; i < N; i++) ref[i] = (double)x[i] + 3.0 * y[i];
        check("triad N=1000003", rel_err(o, ref, N), 1e-6);
        ffree(x), ffree(y), ffree(o);
        free(ref);
    }
    { /* FC with B vectors */
        const int n = 200, B = 8;
        float *A = random_array((long)n * n, 51), *X = random_array((long)n * B, 52), *Y = falloc((long)n * B);
        double *ref = dalloc((long)n * B);
        ref_gemm(n, B, n, A, X, NULL, ref);
        fc_batch(n, B, A, X, Y);
        check("fcb n=200 B=8", rel_err(Y, ref, (long)n * B), 1e-4);
        ref_gemm(n, 1, n, A, X, NULL, ref);
        fc_batch(n, 1, A, X, Y);
        check("fcb n=200 B=1 (matvec)", rel_err(Y, ref, n), 1e-4);
        ffree(A), ffree(X), ffree(Y);
        free(ref);
    }
    say("checks: %s (%d failed)", g_fail ? "FAILED" : "all passed", g_fail);
    return g_fail;
}

/* ================================================================== main */

static void write_env(const Args *a) {
    FILE *f = open_in(a->outdir, "env_seq.json", "w");
    char cpu[256], ts[40], buf[64];
    cpu_model(cpu, sizeof cpu);
    iso_utc(ts, sizeof ts);
    fprintf(f, "{\n");
    json_str(f, "program", "bench_seq (sequential C, 1 thread, no external library)", 1);
    json_str(f, "machine", a->machine, 1);
    json_str(f, "recorded_utc", ts, 1);
    json_str(f, "cpu", cpu, 1);
#ifdef __VERSION__
    json_str(f, "compiler", __VERSION__, 1);
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
    snprintf(buf, sizeof buf, "%d", process_threads());
    json_str(f, "process_threads", buf, 1);
    fprintf(f, "  \"warmup\": %d,\n  \"repeat\": %d,\n  \"order_seed\": %d,\n  \"quick\": %d\n}\n", a->warmup,
            a->repeat, a->order_seed, a->quick);
    fclose(f);
}

static const char *USAGE =
    "bench_seq - sequential C benchmarks (1 thread, no library)\n"
    "  --check-only            run the correctness checks and exit\n"
    "  --machine NAME          label written to every row (required unless --check-only)\n"
    "  --outdir DIR            run folder, created if needed; CSV files are appended (required)\n"
    "  --exps LIST             subset of " ALL_EXPS "\n"
    "  --warmup N --repeat N   untimed / timed runs per point (default 3 / 10)\n"
    "  --order-seed N          0 = ascending sizes, else shuffled (default 1)\n"
    "  --quick                 tiny grid, a few seconds\n"
    "  --no-check              skip the checks before timing\n"
    "  --batches, --direct-batches, --matvec-sizes, --gemm-sizes, --gemm-naive-sizes,\n"
    "  --bw-sizes, --fc-sizes, --fc-b, --step-batches LIST   comma separated grids\n";

int main(int argc, char **argv) {
    Args a;
    double start;
    memset(&a, 0, sizeof a);
    a.exps = ALL_EXPS;
    a.warmup = 3, a.repeat = 10, a.order_seed = 1;
    a.batches = "1,8,16,32,64,96,128,160,192,224,256";
    a.direct_batches = "1,16,64,256";
    a.mv_sizes = "256,512,1024,2048,4096,8192,12288,16384";
    a.gemm_sizes = "256,512,1024,1536,2048";
    a.gemm_naive_sizes = "256,512";
    a.bw_sizes = "1024,4096,8192,16384";
    a.fc_sizes = "2048,8192";
    a.fc_b = "1,16,64,128";
    a.step_batches = "16,64,256";
    parse_args(argc, argv, &a, USAGE);

    if (!a.no_check && run_checks()) return 1;
    if (a.check_only) return 0;

    mkdir_p(a.outdir);
    g_log = open_in(a.outdir, "log_seq.txt", "a");
    write_env(&a);
    say("bench_seq: machine %s, outdir %s, warmup %d, repeat %d, order seed %d%s", a.machine, a.outdir, a.warmup,
        a.repeat, a.order_seed, a.quick ? " (quick)" : "");
    start = now_s();
    if (has_token(a.exps, "conv")) exp_conv(&a);
    if (has_token(a.exps, "fc")) exp_fc(&a);
    if (has_token(a.exps, "gemm")) exp_gemm(&a);
    if (has_token(a.exps, "bw")) exp_bw(&a);
    if (has_token(a.exps, "fcb")) exp_fcb(&a);
    if (has_token(a.exps, "step")) exp_step(&a);
    say("threads in this process: %d (1 = sequential)", process_threads());
    say("done in %.1f s -> %s", now_s() - start, a.outdir);
    fclose(g_log);
    return 0;
}
