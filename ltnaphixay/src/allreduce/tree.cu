/*
 * tree.cu - flat reduce-to-driver allreduce (Hegde & Usmani, Sec. 5, driver model):
 *           reduce all gradients to GPU 0 ("driver"), average, broadcast back.
 *
 *   1. GPU 0 waits for every GPU j's gradients to be ready.
 *   2. For j = 1..k-1: copy grads[j] -> scratch on GPU 0 and ar_add into grads[0].
 *      All of this runs on streams[0], so one scratch buffer is enough.
 *   3. ar_scale(grads[0], 1/k) on GPU 0.
 *   4. Broadcast: GPU j waits for step 3, then copies grads[0] -> grads[j] on streams[j].
 *   5. GPU 0 waits for the k-1 broadcast copies to finish, so the caller cannot
 *      overwrite grads[0] (e.g. the next backward pass) while a peer is still reading it.
 *
 * Every copy goes through GPU 0, so the cost is about 2(k-1)(alpha + n*beta). This alpha-beta
 * form is ours; it equals the paper's summed message cost O(L k) + O(k p / B). The paper's
 * BitTorrent aggregate sends these messages in log2 k parallel rounds; this flat version sends
 * them one after another, so here the summed cost is also the time.
 * Every GPU ends with bitwise the same values.
 */
#include "ar_sync.h"

void allreduce_tree(float **grads, size_t n, int k, cudaStream_t *streams) {
    float *scratch = ar_scratch(0, 0, n * sizeof(float));

    /* 1-2: reduce to GPU 0 */
    for (int j = 1; j < k; j++) ar_mark(j, streams);
    for (int j = 1; j < k; j++) {
        ar_after(0, j, streams);
        ar_copy(scratch, 0, grads[j], j, n, streams[0]);
        ar_add(0, grads[0], scratch, n, streams[0]);
    }
    /* 3: average */
    ar_scale(0, grads[0], 1.0f / k, n, streams[0]);

    /* 4: broadcast */
    ar_mark(0, streams);
    for (int j = 1; j < k; j++) {
        ar_after(j, 0, streams);
        ar_copy(grads[j], j, grads[0], 0, n, streams[j]);
    }
    /* 5: GPU 0 must not reuse grads[0] before the peers finished reading it */
    for (int j = 1; j < k; j++) ar_mark(j, streams);
    for (int j = 1; j < k; j++) ar_after(0, j, streams);
}
