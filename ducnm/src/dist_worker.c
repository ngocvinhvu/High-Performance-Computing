/*
 * Distributed & Multi-worker Gradient Descent simulator (cnn_dist.exe).
 * Implements 4 point-to-point collective algorithms (ring, tree, rd, ps_star)
 * and 3 distributed SGD strategies (sync, local_avg, ps_async).
 */
#include "core/fmnist_loader.h"

#define MAX_WORKERS 16
#define ANY_SENDER  (-1)

enum {
    CH_BCAST = 10,
    CH_TREE,
    CH_RING,
    CH_RDOUB,
    CH_STAR,
    CH_TASK,
    CH_PUSH_GRAD
};

enum {
    COLL_RING = 0,
    COLL_TREE,
    COLL_RDOUB,
    COLL_STAR
};

typedef struct Packet {
    int sender_id, channel, nbytes;
    void *payload;
    struct Packet *next;
} Packet;

typedef struct {
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE cond;
    Packet *first, *last;
} WorkerQueue;

static WorkerQueue g_queues[MAX_WORKERS];
static CRITICAL_SECTION g_sync_lock;
static CONDITION_VARIABLE g_sync_cond;
static int g_sync_arrived = 0, g_sync_epoch = 0;

static void init_msg_bus(int nw)
{
    InitializeCriticalSection(&g_sync_lock);
    InitializeConditionVariable(&g_sync_cond);
    g_sync_arrived = 0;
    g_sync_epoch = 0;
    for (int i = 0; i < nw; i++) {
        InitializeCriticalSection(&g_queues[i].lock);
        InitializeConditionVariable(&g_queues[i].cond);
        g_queues[i].first = g_queues[i].last = NULL;
    }
}

static void destroy_msg_bus(int nw)
{
    for (int i = 0; i < nw; i++)
        DeleteCriticalSection(&g_queues[i].lock);
    DeleteCriticalSection(&g_sync_lock);
}

static void sync_workers_barrier(int nw)
{
    EnterCriticalSection(&g_sync_lock);
    int phase = g_sync_epoch;
    if (++g_sync_arrived == nw) {
        g_sync_arrived = 0;
        g_sync_epoch++;
        WakeAllConditionVariable(&g_sync_cond);
    } else {
        while (phase == g_sync_epoch)
            SleepConditionVariableCS(&g_sync_cond, &g_sync_lock, INFINITE);
    }
    LeaveCriticalSection(&g_sync_lock);
}

static void bus_send(const void *src_buf, int count, size_t elem_sz, int from_id, int to_id, int channel)
{
    size_t bytes = (size_t)count * elem_sz;
    Packet *pkt = zalloc_buf(1, sizeof(Packet));
    pkt->sender_id = from_id;
    pkt->channel = channel;
    pkt->nbytes = (int)bytes;
    pkt->payload = zalloc_buf(bytes ? bytes : 1, 1);
    memcpy(pkt->payload, src_buf, bytes);
    pkt->next = NULL;

    WorkerQueue *q = &g_queues[to_id];
    EnterCriticalSection(&q->lock);
    if (q->last) q->last->next = pkt;
    else q->first = pkt;
    q->last = pkt;
    WakeAllConditionVariable(&q->cond);
    LeaveCriticalSection(&q->lock);
}

static int bus_recv(void *dst_buf, int count, size_t elem_sz, int expected_from, int my_id, int channel)
{
    size_t bytes = (size_t)count * elem_sz;
    WorkerQueue *q = &g_queues[my_id];
    EnterCriticalSection(&q->lock);
    for (;;) {
        Packet *prev = NULL, *cur = q->first;
        while (cur) {
            if ((expected_from == ANY_SENDER || cur->sender_id == expected_from) && cur->channel == channel) {
                if (prev) prev->next = cur->next;
                else q->first = cur->next;
                if (q->last == cur) q->last = prev;
                int actual_sender = cur->sender_id;
                LeaveCriticalSection(&q->lock);
                memcpy(dst_buf, cur->payload, bytes <= (size_t)cur->nbytes ? bytes : (size_t)cur->nbytes);
                free(cur->payload);
                free(cur);
                return actual_sender;
            }
            prev = cur;
            cur = cur->next;
        }
        SleepConditionVariableCS(&q->cond, &q->lock, INFINITE);
    }
}

typedef struct {
    int id, nw, coll_type;
    double comm_sec;
    CnnConfig cfg;
    const FMnistData *ds;
    scalar_t *w, *v, *g, *scratch, *peer_buf;
    int *order;
} DistNode;

/* ---------- 4 Collective Allreduce Topologies ---------- */

static void coll_bcast_tree(DistNode *nd, void *buf, int count, size_t elem_sz)
{
    const double t0 = wall_time_sec();
    int bit = 1;
    while (bit < nd->nw) {
        if (nd->id & bit) {
            bus_recv(buf, count, elem_sz, nd->id - bit, nd->id, CH_BCAST);
            break;
        }
        bit <<= 1;
    }
    for (bit >>= 1; bit > 0; bit >>= 1) {
        if (nd->id + bit < nd->nw)
            bus_send(buf, count, elem_sz, nd->id, nd->id + bit, CH_BCAST);
    }
    nd->comm_sec += wall_time_sec() - t0;
}

static void coll_allreduce_tree(DistNode *nd, scalar_t *vec, scalar_t *tmp, int count)
{
    const double t0 = wall_time_sec();
    for (int bit = 1; bit < nd->nw; bit <<= 1) {
        if (nd->id & bit) {
            bus_send(vec, count, sizeof(scalar_t), nd->id, nd->id - bit, CH_TREE);
            break;
        }
        if (nd->id + bit < nd->nw) {
            bus_recv(tmp, count, sizeof(scalar_t), nd->id + bit, nd->id, CH_TREE);
            for (int i = 0; i < count; i++) vec[i] += tmp[i];
        }
    }
    nd->comm_sec += wall_time_sec() - t0;
    coll_bcast_tree(nd, vec, count, sizeof(scalar_t));
}

static void coll_allreduce_star(DistNode *nd, scalar_t *vec, scalar_t *tmp, int count)
{
    const double t0 = wall_time_sec();
    if (nd->id == 0) {
        for (int peer = 1; peer < nd->nw; peer++) {
            bus_recv(tmp, count, sizeof(scalar_t), peer, 0, CH_STAR);
            for (int i = 0; i < count; i++) vec[i] += tmp[i];
        }
        for (int peer = 1; peer < nd->nw; peer++)
            bus_send(vec, count, sizeof(scalar_t), 0, peer, CH_STAR);
    } else {
        bus_send(vec, count, sizeof(scalar_t), nd->id, 0, CH_STAR);
        bus_recv(vec, count, sizeof(scalar_t), 0, nd->id, CH_STAR);
    }
    nd->comm_sec += wall_time_sec() - t0;
}

static void coll_allreduce_ring(DistNode *nd, scalar_t *vec, scalar_t *tmp, int count)
{
    if (nd->nw <= 1) return;
    const double t0 = wall_time_sec();
    const int p = nd->nw, me = nd->id;
    const int prev_peer = (me - 1 + p) % p;
    const int next_peer = (me + 1) % p;

    for (int s = 0; s < p - 1; s++) {
        int tx_seg = (me - s + p) % p;
        int rx_seg = (me - s - 1 + p) % p;
        int tx_off = shard_offset(count, p, tx_seg), tx_len = shard_offset(count, p, tx_seg + 1) - tx_off;
        int rx_off = shard_offset(count, p, rx_seg), rx_len = shard_offset(count, p, rx_seg + 1) - rx_off;
        bus_send(vec + tx_off, tx_len, sizeof(scalar_t), me, next_peer, CH_RING + s);
        bus_recv(tmp, rx_len, sizeof(scalar_t), prev_peer, me, CH_RING + s);
        for (int i = 0; i < rx_len; i++) vec[rx_off + i] += tmp[i];
    }

    for (int s = 0; s < p - 1; s++) {
        int tx_seg = (me + 1 - s + p) % p;
        int rx_seg = (me - s + p) % p;
        int tx_off = shard_offset(count, p, tx_seg), tx_len = shard_offset(count, p, tx_seg + 1) - tx_off;
        int rx_off = shard_offset(count, p, rx_seg), rx_len = shard_offset(count, p, rx_seg + 1) - rx_off;
        bus_send(vec + tx_off, tx_len, sizeof(scalar_t), me, next_peer, CH_RING + 128 + s);
        bus_recv(vec + rx_off, rx_len, sizeof(scalar_t), prev_peer, me, CH_RING + 128 + s);
    }
    nd->comm_sec += wall_time_sec() - t0;
}

static void coll_allreduce_rdoub(DistNode *nd, scalar_t *vec, scalar_t *tmp, int count)
{
    if (nd->nw <= 1) return;
    if ((nd->nw & (nd->nw - 1)) != 0) {
        coll_allreduce_tree(nd, vec, tmp, count);
        return;
    }
    const double t0 = wall_time_sec();
    int stage = 0;
    for (int bit = 1; bit < nd->nw; bit <<= 1, stage++) {
        int partner = nd->id ^ bit;
        bus_send(vec, count, sizeof(scalar_t), nd->id, partner, CH_RDOUB + stage);
        bus_recv(tmp, count, sizeof(scalar_t), partner, nd->id, CH_RDOUB + stage);
        for (int i = 0; i < count; i++) vec[i] += tmp[i];
    }
    nd->comm_sec += wall_time_sec() - t0;
}

static void execute_allreduce(DistNode *nd, scalar_t *vec, scalar_t *tmp, int count)
{
    if (nd->coll_type == COLL_RING) coll_allreduce_ring(nd, vec, tmp, count);
    else if (nd->coll_type == COLL_RDOUB) coll_allreduce_rdoub(nd, vec, tmp, count);
    else if (nd->coll_type == COLL_STAR) coll_allreduce_star(nd, vec, tmp, count);
    else coll_allreduce_tree(nd, vec, tmp, count);
}

/* ---------- Distributed SGD Execution Strategies ---------- */

static double run_epoch_sync_sgd(DistNode *nd)
{
    const CnnConfig *c = &nd->cfg;
    double tot_loss = 0;
    for (int pos = 0; pos < c->train_samples; pos += c->batch_size) {
        const int cur_b = (c->train_samples - pos < c->batch_size) ? (c->train_samples - pos) : c->batch_size;
        const int beg = shard_offset(cur_b, nd->nw, nd->id);
        const int end = shard_offset(cur_b, nd->nw, nd->id + 1);

        nd->g[TOTAL_PARAMS] = c->flag_naive_conv
            ? accumulate_minibatch_naive(nd->w, nd->ds, nd->order + pos + beg, end - beg, nd->g)
            : accumulate_minibatch_im2col(nd->w, nd->ds, nd->order + pos + beg, end - beg, nd->g, NULL);
        execute_allreduce(nd, nd->g, nd->peer_buf, TOTAL_PARAMS + 1);
        apply_sgd_momentum(nd->w, nd->v, nd->g, (scalar_t)1 / cur_b, c->lr, c->momentum);
        tot_loss += nd->g[TOTAL_PARAMS];
    }
    return tot_loss / c->train_samples;
}

static double run_epoch_local_avg(DistNode *nd)
{
    const CnnConfig *c = &nd->cfg;
    const int beg = shard_offset(c->train_samples, nd->nw, nd->id);
    const int end = shard_offset(c->train_samples, nd->nw, nd->id + 1);
    scalar_t my_loss = 0;

    for (int pos = beg; pos < end; pos += c->batch_size) {
        const int cur_b = (end - pos < c->batch_size) ? (end - pos) : c->batch_size;
        my_loss += c->flag_naive_conv
            ? accumulate_minibatch_naive(nd->w, nd->ds, nd->order + pos, cur_b, nd->g)
            : accumulate_minibatch_im2col(nd->w, nd->ds, nd->order + pos, cur_b, nd->g, NULL);
        apply_sgd_momentum(nd->w, nd->v, nd->g, (scalar_t)1 / cur_b, c->lr, c->momentum);
    }
    memcpy(nd->scratch, nd->w, sizeof(scalar_t) * TOTAL_PARAMS);
    memcpy(nd->scratch + TOTAL_PARAMS, nd->v, sizeof(scalar_t) * TOTAL_PARAMS);
    nd->scratch[2 * TOTAL_PARAMS] = my_loss;
    execute_allreduce(nd, nd->scratch, nd->peer_buf, 2 * TOTAL_PARAMS + 1);
    for (int i = 0; i < TOTAL_PARAMS; i++) {
        nd->w[i] = nd->scratch[i] / nd->nw;
        nd->v[i] = nd->scratch[TOTAL_PARAMS + i] / nd->nw;
    }
    return nd->scratch[2 * TOTAL_PARAMS] / c->train_samples;
}

static double run_epoch_ps_master(DistNode *nd)
{
    const CnnConfig *c = &nd->cfg;
    const int num_batches = (c->train_samples + c->batch_size - 1) / c->batch_size;
    const scalar_t eff_lr = c->lr / (scalar_t)sqrt(nd->nw - 1);
    int cursor = 0, busy_workers = 0;
    double tot_loss = 0;

    for (int w_id = 1; w_id < nd->nw; w_id++) {
        memcpy(nd->scratch, nd->w, sizeof(scalar_t) * TOTAL_PARAMS);
        nd->scratch[TOTAL_PARAMS] = (cursor < num_batches) ? (scalar_t)cursor++ : (scalar_t)-1;
        busy_workers += (nd->scratch[TOTAL_PARAMS] >= 0);
        bus_send(nd->scratch, TOTAL_PARAMS + 1, sizeof(scalar_t), 0, w_id, CH_TASK);
    }
    while (busy_workers > 0) {
        int worker_id = bus_recv(nd->g, TOTAL_PARAMS + 2, sizeof(scalar_t), ANY_SENDER, 0, CH_PUSH_GRAD);
        apply_sgd_momentum(nd->w, nd->v, nd->g, (scalar_t)1 / nd->g[TOTAL_PARAMS + 1], eff_lr, c->momentum);
        tot_loss += nd->g[TOTAL_PARAMS];

        memcpy(nd->scratch, nd->w, sizeof(scalar_t) * TOTAL_PARAMS);
        nd->scratch[TOTAL_PARAMS] = (cursor < num_batches) ? (scalar_t)cursor++ : (scalar_t)-1;
        busy_workers -= (nd->scratch[TOTAL_PARAMS] < 0);
        bus_send(nd->scratch, TOTAL_PARAMS + 1, sizeof(scalar_t), 0, worker_id, CH_TASK);
    }
    return tot_loss / c->train_samples;
}

static void run_epoch_ps_client(DistNode *nd)
{
    const CnnConfig *c = &nd->cfg;
    for (;;) {
        double t0 = wall_time_sec();
        bus_recv(nd->scratch, TOTAL_PARAMS + 1, sizeof(scalar_t), 0, nd->id, CH_TASK);
        nd->comm_sec += wall_time_sec() - t0;

        const int b_idx = (int)nd->scratch[TOTAL_PARAMS];
        if (b_idx < 0) return;
        const int pos = b_idx * c->batch_size;
        const int cur_b = (c->train_samples - pos < c->batch_size) ? (c->train_samples - pos) : c->batch_size;
        nd->g[TOTAL_PARAMS] = c->flag_naive_conv
            ? accumulate_minibatch_naive(nd->scratch, nd->ds, nd->order + pos, cur_b, nd->g)
            : accumulate_minibatch_im2col(nd->scratch, nd->ds, nd->order + pos, cur_b, nd->g, NULL);
        nd->g[TOTAL_PARAMS + 1] = (scalar_t)cur_b;

        t0 = wall_time_sec();
        bus_send(nd->g, TOTAL_PARAMS + 2, sizeof(scalar_t), nd->id, 0, CH_PUSH_GRAD);
        nd->comm_sec += wall_time_sec() - t0;
    }
}

static double g_worker_ep_sec[MAX_WORKERS];
static double g_cum_time = 0, g_final_loss = 0, g_final_val = 0, g_final_test = 0;

static DWORD WINAPI dist_node_entry(LPVOID param)
{
    DistNode *nd = (DistNode *)param;
    const int is_sync = !strcmp(nd->cfg.strategy, "sync");
    const int is_avg  = !strcmp(nd->cfg.strategy, "avg") || !strcmp(nd->cfg.strategy, "local_avg");

    coll_bcast_tree(nd, nd->w, TOTAL_PARAMS, sizeof(scalar_t));
    nd->comm_sec = 0;

    for (int ep = 1; ep <= nd->cfg.epochs; ep++) {
        sync_workers_barrier(nd->nw);
        const double t0 = wall_time_sec();
        shuffle_indices(nd->order, nd->cfg.train_samples, nd->cfg.seed, ep);
        double loss = 0;
        if (is_sync) loss = run_epoch_sync_sgd(nd);
        else if (is_avg) loss = run_epoch_local_avg(nd);
        else if (nd->id == 0) loss = run_epoch_ps_master(nd);
        else run_epoch_ps_client(nd);

        g_worker_ep_sec[nd->id] = wall_time_sec() - t0;
        sync_workers_barrier(nd->nw);

        if (nd->id == 0) {
            double max_sec = 0;
            for (int r = 0; r < nd->nw; r++)
                if (g_worker_ep_sec[r] > max_sec) max_sec = g_worker_ep_sec[r];
            g_cum_time += max_sec;
            g_final_loss = loss;
            g_final_val = eval_classification_acc(nd->w, nd->ds, VAL_OFFSET, VAL_COUNT);
            g_final_test = eval_classification_acc(nd->w, nd->ds, TEST_OFFSET, TEST_COUNT);
            log_epoch_metrics(ep, g_final_loss, g_final_val, g_final_test, max_sec);
        }
        sync_workers_barrier(nd->nw);
    }
    return 0;
}

/* ---------- Collective Allreduce Microbenchmark (--bench-comm) ---------- */

typedef struct {
    int id, nw, coll_type, elems, iters;
    scalar_t *vec, *tmp;
    double avg_sec;
} CollBenchJob;

static DWORD WINAPI coll_bench_worker(LPVOID param)
{
    CollBenchJob *job = (CollBenchJob *)param;
    DistNode dummy;
    memset(&dummy, 0, sizeof dummy);
    dummy.id = job->id;
    dummy.nw = job->nw;
    dummy.coll_type = job->coll_type;

    for (int i = 0; i < job->elems; i++) job->vec[i] = (scalar_t)(job->id + 1);
    execute_allreduce(&dummy, job->vec, job->tmp, job->elems);
    sync_workers_barrier(job->nw);

    const double t0 = wall_time_sec();
    for (int it = 0; it < job->iters; it++)
        execute_allreduce(&dummy, job->vec, job->tmp, job->elems);
    sync_workers_barrier(job->nw);
    job->avg_sec = (wall_time_sec() - t0) / job->iters;
    return 0;
}

static void run_collective_microbench(void)
{
    static const int worker_counts[] = {2, 4, 8};
    static const int msg_elems[] = {1024, 4096, 16384, TOTAL_PARAMS, 262144, 1048576, 4194304};
    static const char *coll_names[] = {"ring", "tree", "rd", "naive"};

    for (int wi = 0; wi < 3; wi++) {
        int nw = worker_counts[wi];
        for (int mi = 0; mi < 7; mi++) {
            int elems = msg_elems[mi];
            int iters = (elems <= 65536) ? 40 : ((elems <= 262144) ? 15 : 5);
            for (int ctype = 0; ctype < 4; ctype++) {
                init_msg_bus(nw);
                CollBenchJob jobs[MAX_WORKERS];
                HANDLE th[MAX_WORKERS];
                for (int r = 0; r < nw; r++) {
                    jobs[r].id = r;
                    jobs[r].nw = nw;
                    jobs[r].coll_type = ctype;
                    jobs[r].elems = elems;
                    jobs[r].iters = iters;
                    jobs[r].vec = zalloc_buf(elems, sizeof(scalar_t));
                    jobs[r].tmp = zalloc_buf(elems, sizeof(scalar_t));
                    th[r] = CreateThread(NULL, 0, coll_bench_worker, &jobs[r], 0, NULL);
                }
                WaitForMultipleObjects(nw, th, TRUE, INFINITE);
                double slowest = 0;
                for (int r = 0; r < nw; r++) {
                    if (jobs[r].avg_sec > slowest) slowest = jobs[r].avg_sec;
                    CloseHandle(th[r]);
                    free(jobs[r].vec);
                    free(jobs[r].tmp);
                }
                destroy_msg_bus(nw);
                printf("[COMM_BENCH] method=%s workers=%d msg_bytes=%d latency_us=%.3f\n",
                       coll_names[ctype], nw, elems * (int)sizeof(scalar_t), slowest * 1e6);
                fflush(stdout);
            }
        }
    }
}

int main(int argc, char **argv)
{
    CnnConfig cfg = parse_cli_args(argc, argv);
    if (cfg.flag_comm_bench) {
        run_collective_microbench();
        return 0;
    }

    int coll_type = COLL_RING;
    if (!strcmp(cfg.collective, "tree")) coll_type = COLL_TREE;
    else if (!strcmp(cfg.collective, "rd")) coll_type = COLL_RDOUB;
    else if (!strcmp(cfg.collective, "naive") || !strcmp(cfg.collective, "star")) coll_type = COLL_STAR;

    FMnistData ds = alloc_fmnist_data();
    load_fmnist_idx(cfg.data_dir, &ds);

    printf("[CONFIG] runner=dist params=%d train=%d test=%d batch=%d strategy=%s collective=%s workers=%d\n",
           (int)TOTAL_PARAMS, cfg.train_samples, (int)TEST_COUNT, cfg.batch_size,
           cfg.strategy, cfg.collective, cfg.workers);
    fflush(stdout);

    init_msg_bus(cfg.workers);
    DistNode nodes[MAX_WORKERS];
    HANDLE th[MAX_WORKERS];

    for (int r = 0; r < cfg.workers; r++) {
        nodes[r].id = r;
        nodes[r].nw = cfg.workers;
        nodes[r].coll_type = coll_type;
        nodes[r].comm_sec = 0;
        nodes[r].cfg = cfg;
        nodes[r].ds = &ds;
        nodes[r].w = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
        nodes[r].v = zalloc_buf(TOTAL_PARAMS, sizeof(scalar_t));
        nodes[r].g = zalloc_buf(TOTAL_PARAMS + 2, sizeof(scalar_t));
        nodes[r].scratch  = zalloc_buf(2 * TOTAL_PARAMS + 3, sizeof(scalar_t));
        nodes[r].peer_buf = zalloc_buf(2 * TOTAL_PARAMS + 3, sizeof(scalar_t));
        nodes[r].order    = zalloc_buf(cfg.train_samples, sizeof(int));
        if (r == 0) init_cnn_weights(nodes[r].w, cfg.seed);
    }

    g_cum_time = 0;
    for (int r = 0; r < cfg.workers; r++)
        th[r] = CreateThread(NULL, 0, dist_node_entry, &nodes[r], 0, NULL);
    WaitForMultipleObjects(cfg.workers, th, TRUE, INFINITE);
    for (int r = 0; r < cfg.workers; r++) CloseHandle(th[r]);

    double sum_comm = 0;
    for (int r = 0; r < cfg.workers; r++) {
        sum_comm += nodes[r].comm_sec;
        free(nodes[r].w); free(nodes[r].v); free(nodes[r].g);
        free(nodes[r].scratch); free(nodes[r].peer_buf); free(nodes[r].order);
    }
    destroy_msg_bus(cfg.workers);

    int is_async = !strcmp(cfg.strategy, "async");
    printf("[SUMMARY] total_s=%.4f comm_s=%.4f loss=%.4f val_acc=%.2f test_acc=%.2f\n",
           g_cum_time, sum_comm / (is_async && cfg.workers > 1 ? cfg.workers - 1 : cfg.workers),
           g_final_loss, g_final_val, g_final_test);

    free_fmnist_data(&ds);
    return 0;
}
