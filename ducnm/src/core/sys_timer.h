/*
 * High-resolution timer, memory allocation helpers, Xorshift64* PRNG,
 * and CLI configuration parser for ducnm HPC benchmarks.
 */
#ifndef SYS_TIMER_H
#define SYS_TIMER_H

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

static inline void disable_power_throttling(void)
{
#ifdef _WIN32
    PROCESS_POWER_THROTTLING_STATE st = {
        PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        0
    };
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof st);
#endif
}

static inline void fatal_err(const char *msg, const char *ctx)
{
    fprintf(stderr, "[FATAL] %s %s\n", msg, ctx ? ctx : "");
    exit(EXIT_FAILURE);
}

static inline void *zalloc_buf(size_t count, size_t elem_bytes)
{
    void *ptr = calloc(count ? count : 1, elem_bytes);
    if (!ptr) fatal_err("Memory allocation failed", NULL);
    return ptr;
}

static inline double wall_time_sec(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);
    return (double)ticks.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* Xorshift64* PRNG: reproducible across single-thread CPU, multi-worker, and CUDA host. */
static inline uint64_t xor64_next(uint64_t *state)
{
    uint64_t x = *state;
    if (x == 0) x = 0x2545F4914F6CDD1DULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double xor64_uniform(uint64_t *state)
{
    return ((xor64_next(state) >> 11) + 0.5) / 9007199254740992.0;
}

static inline double xor64_gaussian(uint64_t *state)
{
    double u1 = xor64_uniform(state);
    double u2 = xor64_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
}

static inline void shuffle_indices(int *order, int n, int seed, int epoch)
{
    uint64_t st = ((uint64_t)seed * 0x9E3779B185EBCA87ULL) ^ ((uint64_t)epoch * 0xC2B2AE3D27D4EB4FULL);
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(xor64_next(&st) % (uint64_t)(i + 1));
        int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

static inline int shard_offset(int total, int num_workers, int worker_id)
{
    int base = total / num_workers;
    int rem = total % num_workers;
    return base * worker_id + (worker_id < rem ? worker_id : rem);
}

#endif
