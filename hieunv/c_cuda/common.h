/* common.h - shared by bench_seq.c (sequential C) and the host side of bench_cuda.cu (CUDA).

   Plain C, no threads, no BLAS, no external library: only the C standard library, plus the
   POSIX (or Win32) calls for the clocks and mkdir. Compiles as C11 and as C++ (nvcc host code).

   Contents: layer constants (the paper's first conv layer), clocks, random data, statistics,
   CSV/JSON output, command-line parsing, and the float64 reference kernels used by --check. */
#ifndef HPC_COMMON_H
#define HPC_COMMON_H

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__cplusplus)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#if defined(__GNUC__)
#define MAYBE_UNUSED __attribute__((unused))
#define NOINLINE __attribute__((noinline))
#define NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define MAYBE_UNUSED
#define NOINLINE __declspec(noinline)
#define NORETURN __declspec(noreturn)
#else
#define MAYBE_UNUSED
#define NOINLINE
#define NORETURN
#endif
#define API static MAYBE_UNUSED

/* ------------------------------------------------------------------ the paper's conv layer
   Greyscale 224x224 input, 96 filters of 11x11, stride 4, no padding: output 96 x 54 x 54.
   Caffe unrolls each image (im2col) into a 121 x 2916 matrix and multiplies W (96 x 121) by it. */
#define IMG 224
#define C_IN 1
#define C_OUT 96
#define KS 11
#define STRIDE 4
#define OUT_HW ((IMG - KS) / STRIDE + 1)                  /* 54 */
#define PATCHES (OUT_HW * OUT_HW)                         /* 2916 */
#define PATCH_LEN (C_IN * KS * KS)                        /* 121 */
#define IMG_ELEMS ((long)C_IN * IMG * IMG)
#define OUT_ELEMS ((long)C_OUT * PATCHES)
#define COL_ELEMS ((long)PATCH_LEN * PATCHES)
#define FLOPS_PER_IMAGE (2.0 * C_OUT * PATCH_LEN * PATCHES) /* multiply-adds of the GEMM, x2 */

#define MAX_LIST 64

/* ------------------------------------------------------------------ logging */
static FILE *g_log = NULL;

API void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
    if (g_log) {
        va_start(ap, fmt);
        vfprintf(g_log, fmt, ap);
        va_end(ap);
        fputc('\n', g_log);
        fflush(g_log);
    }
}

API NORETURN void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

/* ------------------------------------------------------------------ clocks */
/* Monotonic seconds, for measuring durations. */
API double now_s(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

/* Seconds since 1970 (same clock as Python's time.time()), to line rows up with gpu_clocks.csv. */
API double epoch_s(void) {
#ifdef _WIN32
    FILETIME ft;
    ULARGE_INTEGER u;
    GetSystemTimePreciseAsFileTime(&ft);
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    return (double)(u.QuadPart - 116444736000000000ULL) * 1e-7;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

API void iso_utc(char *buf, size_t n) {
    time_t t = time(NULL);
    struct tm g;
#ifdef _WIN32
    gmtime_s(&g, &t);
#else
    gmtime_r(&t, &g);
#endif
    strftime(buf, n, "%Y-%m-%dT%H:%M:%S+00:00", &g);
}

/* ------------------------------------------------------------------ random data (splitmix64) */
API unsigned long long splitmix64(unsigned long long *s) {
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static unsigned long long g_rng = 1;
API void rng_seed(unsigned long long seed) { g_rng = seed * 0x2545F4914F6CDD1DULL + 12345; }
/* Uniform in [-1, 1), 24 random bits. */
API float rng_float(void) { return (float)((double)(splitmix64(&g_rng) >> 40) * (2.0 / 16777216.0) - 1.0); }
API void fill_random(float *p, long n) {
    for (long i = 0; i < n; i++) p[i] = rng_float();
}

/* Fisher-Yates with its own generator. Seed 0 keeps the list as given (ascending). */
API void shuffle(long *a, int n, int seed) {
    unsigned long long s = 0x853C49E6748FEA9BULL ^ ((unsigned long long)seed * 0x9E3779B97F4A7C15ULL);
    if (!seed) return;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(splitmix64(&s) % (unsigned long long)(i + 1));
        long t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

/* ------------------------------------------------------------------ memory */
API float *falloc(long n) {
    size_t bytes = (size_t)(n > 0 ? n : 1) * sizeof(float);
    void *p = NULL;
#ifdef _WIN32
    p = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&p, 64, bytes)) p = NULL;
#endif
    if (!p) die("out of memory (%zu bytes)", bytes);
    return (float *)p;
}

API void ffree(float *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

API void fzero(float *p, long n) { memset(p, 0, (size_t)n * sizeof(float)); }

API double *dalloc(long n) {
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) die("out of memory");
    return p;
}

/* ------------------------------------------------------------------ statistics */
typedef struct {
    double mean, std, min, median, issue, t_start, t_end;
} Stats;

API int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ms[i]: wall time of timed run i; issue[i]: time until the host call returned (= ms on the CPU). */
API Stats make_stats(const double *ms, const double *issue, int n, double t_start, double t_end) {
    Stats s;
    double sum = 0, isum = 0, var = 0;
    double *tmp;
    if (n < 1) die("no timed runs");
    tmp = dalloc(n);
    s.min = ms[0];
    for (int i = 0; i < n; i++) {
        sum += ms[i];
        isum += issue[i];
        if (ms[i] < s.min) s.min = ms[i];
        tmp[i] = ms[i];
    }
    s.mean = sum / n;
    s.issue = isum / n;
    for (int i = 0; i < n; i++) var += (ms[i] - s.mean) * (ms[i] - s.mean);
    s.std = n > 1 ? sqrt(var / (n - 1)) : 0.0;
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    s.median = (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    s.t_start = t_start;
    s.t_end = t_end;
    return s;
}

/* ------------------------------------------------------------------ files */
API void mkdir_p(const char *path) {
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof buf) die("path too long: %s", path);
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\\' || buf[i] == '\0') {
            char c = buf[i];
            buf[i] = '\0';
#ifdef _WIN32
            _mkdir(buf);
#else
            mkdir(buf, 0755);
#endif
            buf[i] = c;
        }
    }
}

API FILE *open_in(const char *dir, const char *name, const char *mode) {
    char path[1200];
    FILE *f;
    snprintf(path, sizeof path, "%s/%s", dir, name);
    f = fopen(path, mode);
    if (!f) die("cannot open %s", path);
    return f;
}

/* Same columns as the PyTorch version (code/common.py), so the same analysis scripts can read both. */
#define CSV_HEADER "exp,impl,device,threads,size,mean_ms,std_ms,min_ms,median_ms,issue_ms,warmup,repeats," \
                   "gflops,gbps,order,t_start,t_end,machine,timestamp"

API FILE *csv_open(const char *dir, const char *name) {
    char path[1200];
    int fresh = 1;
    FILE *probe, *f;
    snprintf(path, sizeof path, "%s/%s", dir, name);
    probe = fopen(path, "r");
    if (probe) {
        fresh = fgetc(probe) == EOF;
        fclose(probe);
    }
    f = fopen(path, "a");
    if (!f) die("cannot open %s", path);
    if (fresh) fprintf(f, "%s\n", CSV_HEADER);
    return f;
}

static int g_order = 0;

API void csv_row(FILE *f, const char *exp, const char *impl, const char *device, int threads, long size,
                 const Stats *s, int warmup, int repeat, double flops, double bytes, const char *machine) {
    char ts[40];
    double sec = s->mean / 1000.0;
    iso_utc(ts, sizeof ts);
    fprintf(f, "%s,%s,%s,%d,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,", exp, impl, device, threads, size, s->mean,
            s->std, s->min, s->median, s->issue, warmup, repeat);
    if (flops > 0) fprintf(f, "%.3f", flops / sec / 1e9);
    fputc(',', f);
    if (bytes > 0) fprintf(f, "%.3f", bytes / sec / 1e9);
    fprintf(f, ",%d,%.4f,%.4f,%s,%s\n", ++g_order, s->t_start, s->t_end, machine, ts);
    fflush(f);
}

API void json_str(FILE *f, const char *key, const char *val, int comma) {
    fprintf(f, "  \"%s\": \"", key);
    for (const char *p = val; *p; p++) {
        if (*p == '"' || *p == '\\') fputc('\\', f);
        if ((unsigned char)*p >= 32) fputc(*p, f);
    }
    fprintf(f, "\"%s\n", comma ? "," : "");
}

API void cpu_model(char *buf, size_t n) {
    snprintf(buf, n, "unknown");
#ifndef _WIN32
    {
        FILE *f = fopen("/proc/cpuinfo", "r");
        char line[512];
        if (!f) return;
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "model name", 10)) {
                char *c = strchr(line, ':');
                if (c) {
                    c++;
                    while (*c == ' ' || *c == '\t') c++;
                    c[strcspn(c, "\n")] = 0;
                    snprintf(buf, n, "%s", c);
                }
                break;
            }
        }
        fclose(f);
    }
#else
    {
        const char *p = getenv("PROCESSOR_IDENTIFIER");
        if (p) snprintf(buf, n, "%s", p);
    }
#endif
}

/* Number of threads of this process (Linux /proc/self/status), -1 if unknown. */
API int process_threads(void) {
    int t = -1;
#ifndef _WIN32
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "Threads:", 8)) t = atoi(line + 8);
    fclose(f);
#endif
    return t;
}

/* ------------------------------------------------------------------ command line */
typedef struct {
    const char *machine, *outdir, *exps;
    int warmup, repeat, order_seed, quick, check_only, no_check;
    double warm_seconds;
    const char *batches, *direct_batches, *mv_sizes, *gemm_sizes, *gemm_naive_sizes;
    const char *bw_sizes, *fc_sizes, *fc_b, *step_batches;
} Args;

#define ALL_EXPS "conv,fc,gemm,bw,fcb,step"

API void apply_quick(Args *a) {
    a->batches = "1,3";
    a->direct_batches = "1,2";
    a->mv_sizes = "256,1000";
    a->gemm_sizes = "128,200";
    a->gemm_naive_sizes = "128";
    a->bw_sizes = "256,1000";
    a->fc_sizes = "256,300";
    a->fc_b = "1,8";
    a->step_batches = "2";
    a->warmup = 1;
    a->repeat = 2;
    a->warm_seconds = 0.5;
}

API int parse_list(const char *s, long *out, int max) {
    int n = 0;
    const char *p = s;
    while (*p) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0) die("bad list '%s' (comma separated positive integers)", s);
        if (n == max) die("list too long: %s", s);
        out[n++] = v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

API int has_token(const char *list, const char *tok) {
    size_t len_tok = strlen(tok);
    const char *p = list;
    while (*p) {
        const char *e = strchr(p, ',');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (len == len_tok && !strncmp(p, tok, len)) return 1;
        if (!e) break;
        p = e + 1;
    }
    return 0;
}

API void parse_args(int argc, char **argv, Args *a, const char *usage) {
    /* --quick first sets the tiny grid; options given explicitly then override it, in any order */
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--quick")) apply_quick(a);
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        int takes = 1;
        if (!strcmp(k, "--help") || !strcmp(k, "-h")) {
            printf("%s", usage);
            exit(0);
        } else if (!strcmp(k, "--quick")) {
            a->quick = 1, takes = 0;
        } else if (!strcmp(k, "--check-only")) {
            a->check_only = 1, takes = 0;
        } else if (!strcmp(k, "--no-check")) {
            a->no_check = 1, takes = 0;
        } else if (!v) {
            die("missing value for %s", k);
        } else if (!strcmp(k, "--machine")) a->machine = v;
        else if (!strcmp(k, "--outdir")) a->outdir = v;
        else if (!strcmp(k, "--exps")) a->exps = v;
        else if (!strcmp(k, "--warmup")) a->warmup = atoi(v);
        else if (!strcmp(k, "--repeat")) a->repeat = atoi(v);
        else if (!strcmp(k, "--order-seed")) a->order_seed = atoi(v);
        else if (!strcmp(k, "--warm-seconds")) a->warm_seconds = atof(v);
        else if (!strcmp(k, "--batches")) a->batches = v;
        else if (!strcmp(k, "--direct-batches")) a->direct_batches = v;
        else if (!strcmp(k, "--matvec-sizes")) a->mv_sizes = v;
        else if (!strcmp(k, "--gemm-sizes")) a->gemm_sizes = v;
        else if (!strcmp(k, "--gemm-naive-sizes")) a->gemm_naive_sizes = v;
        else if (!strcmp(k, "--bw-sizes")) a->bw_sizes = v;
        else if (!strcmp(k, "--fc-sizes")) a->fc_sizes = v;
        else if (!strcmp(k, "--fc-b")) a->fc_b = v;
        else if (!strcmp(k, "--step-batches")) a->step_batches = v;
        else die("unknown option %s (try --help)", k);
        i += takes;
    }
    if (a->repeat < 1 || a->warmup < 0) die("--repeat must be >= 1 and --warmup >= 0");
    if (!a->check_only && (!a->machine || !a->outdir)) die("--machine and --outdir are required (try --help)");
}

/* ------------------------------------------------------------------ float64 references for --check */
static int g_fail = 0;

/* max |got - ref| / max |ref|: 0 for identical results, about 1 for an indexing bug. NaN counts as a failure. */
API double rel_err(const float *got, const double *ref, long n) {
    double maxref = 0, maxerr = 0;
    for (long i = 0; i < n; i++) {
        double r = fabs(ref[i]), e = fabs((double)got[i] - ref[i]);
        if (r > maxref) maxref = r;
        if (e != e) e = INFINITY;
        if (e > maxerr) maxerr = e;
    }
    return maxerr / (maxref > 0 ? maxref : 1.0);
}

API void check(const char *what, double err, double tol) {
    int ok = err <= tol;
    say("check %-44s rel. error %.2e  %s", what, err, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}

/* C = A B (+ bias per row of C); row-major, A is M x K, B is K x N. */
API void ref_gemm(int M, int N, int K, const float *A, const float *B, const float *bias, double *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) C[(long)i * N + j] = bias ? bias[i] : 0.0;
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            double a = A[(long)i * K + k];
            for (int j = 0; j < N; j++) C[(long)i * N + j] += a * B[(long)k * N + j];
        }
}

/* out[b][oc][oh*54 + ow] for nb images, straight from the definition of the convolution. */
API void ref_conv(int nb, const float *x, const float *w, const float *bias, double *out) {
    for (int b = 0; b < nb; b++)
        for (int oc = 0; oc < C_OUT; oc++)
            for (int oh = 0; oh < OUT_HW; oh++)
                for (int ow = 0; ow < OUT_HW; ow++) {
                    double s = bias[oc];
                    for (int c = 0; c < C_IN; c++)
                        for (int kh = 0; kh < KS; kh++)
                            for (int kw = 0; kw < KS; kw++)
                                s += (double)w[((oc * C_IN + c) * KS + kh) * KS + kw] *
                                     x[b * IMG_ELEMS + (long)c * IMG * IMG + (long)(oh * STRIDE + kh) * IMG +
                                       ow * STRIDE + kw];
                    out[(long)b * OUT_ELEMS + (long)oc * PATCHES + oh * OUT_HW + ow] = s;
                }
}

API void ref_matvec(int n, const float *A, const float *x, double *y) {
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int k = 0; k < n; k++) s += (double)A[(long)i * n + k] * x[k];
        y[i] = s;
    }
}

/* Caffe's im2col for one image: col[r][p], r = (c*11 + kh)*11 + kw, p = oh*54 + ow.
   Used by the sequential version and as the reference for the GPU im2col kernel. */
API NOINLINE void im2col_cpu(const float *RESTRICT x, float *RESTRICT col) {
    for (int c = 0; c < C_IN; c++)
        for (int kh = 0; kh < KS; kh++)
            for (int kw = 0; kw < KS; kw++) {
                float *RESTRICT dst = col + (long)((c * KS + kh) * KS + kw) * PATCHES;
                for (int oh = 0; oh < OUT_HW; oh++) {
                    const float *src = x + (long)c * IMG * IMG + (long)(oh * STRIDE + kh) * IMG + kw;
                    for (int ow = 0; ow < OUT_HW; ow++) dst[oh * OUT_HW + ow] = src[ow * STRIDE];
                }
            }
}

#endif /* HPC_COMMON_H */
