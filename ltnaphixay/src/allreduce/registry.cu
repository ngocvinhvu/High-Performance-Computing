/* registry.cu - maps method names to implementations. */
#include "allreduce.h"
#include <stdio.h>
#include <string.h>

typedef struct { const char *name; allreduce_fn fn; const char *desc; } Entry;

static const Entry TABLE[] = {
    { "host_ref", allreduce_host_ref, "reference: copy to host, sum on CPU, copy back (correct, slow)" },
    { "tree",     allreduce_tree,     "flat: reduce to GPU 0 (driver), then broadcast" },
    { "ring",     allreduce_ring,     "ring: reduce-scatter + allgather" },
    { "rd",       allreduce_rd,       "recursive doubling / halving" },
};
static const int N_ENTRIES = sizeof(TABLE) / sizeof(TABLE[0]);
static int g_sel = 0;

int allreduce_select(const char *name) {
    for (int i = 0; i < N_ENTRIES; i++)
        if (!strcmp(TABLE[i].name, name)) { g_sel = i; return 0; }
    fprintf(stderr, "unknown allreduce method '%s'\n", name);
    allreduce_list();
    return -1;
}

const char *allreduce_selected_name(void) { return TABLE[g_sel].name; }

void allreduce_list(void) {
    fprintf(stderr, "available methods:\n");
    for (int i = 0; i < N_ENTRIES; i++) fprintf(stderr, "  %-9s %s\n", TABLE[i].name, TABLE[i].desc);
}

void allreduce(float **grads, size_t n, int k, cudaStream_t *streams) {
    if (k <= 1 || n == 0) return;
    TABLE[g_sel].fn(grads, n, k, streams);
}
