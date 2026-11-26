"""Experiment 2 (Bài toán 2): So sánh hiệu năng GPU (NVIDIA Quadro M620) và CPU (Intel Core i7-6820HQ)
trong từng lớp mạng (Conv2D, Matvec, Tiled GEMM) và trong huấn luyện thực tế tới cùng ngưỡng accuracy trên Fashion-MNIST.
"""
import re
import statistics

from runner_utils import (
    bin_path, calc_mean_std, export_report, find_convergence_point,
    fmt_num, invoke_binary, run_training_job
)

TRIALS = 3
TARGET_VAL_ACC = 84.0
MAX_EPOCHS = 6
EVAL_SEEDS = (11, 12, 13)


def bench_layers():
    pat = re.compile(r'\[LAYER_BENCH\] op=(\w+) dim=(\d+) cpu_ms=(\S+) gpu_kern_ms=(\S+) gpu_e2e_ms=(\S+)')
    grouped = {}
    for _ in range(TRIALS):
        out = invoke_binary(1, [bin_path('cnn_gpu'), '--bench'])
        for op, dim, c_ms, gk_ms, ge_ms in pat.findall(out):
            grouped.setdefault((op, int(dim)), []).append((float(c_ms), float(gk_ms), float(ge_ms)))

    rows = []
    for (op, dim), vals in grouped.items():
        rows.append(dict(
            op=op,
            dim=dim,
            runs=vals,
            cpu_ms=statistics.median(c for c, _, _ in vals),
            gpu_kern_ms=statistics.median(gk for _, gk, _ in vals),
            gpu_e2e_ms=statistics.median(ge for _, _, ge in vals),
            speedup_kern=calc_mean_std(c / gk for c, gk, _ in vals),
            speedup_e2e=calc_mean_std(c / ge for c, _, ge in vals),
        ))
    return rows


def bench_convergence():
    configs = [
        ('CPU 1 nhân (Conv trực tiếp)', 'cpu', 1, 0.02, {'naive_conv': True}),
        ('CPU 1 nhân (im2col + GEMM)',  'cpu', 1, 0.02, {}),
        ('GPU CUDA (Quadro M620)',      'gpu', 1, 0.02, {}),
        ('Đa luồng Sync-Ring (4 worker)', 'dist', 4, 0.02, {'strategy': 'sync', 'collective': 'ring'}),
        ('Đa luồng Local-Avg (4 worker)', 'dist', 4, 0.02, {'strategy': 'avg', 'collective': 'ring'}),
        ('Bất đồng bộ Parameter Server (4 worker)', 'dist', 4, 0.10, {'strategy': 'async'}),
    ]
    all_runs = [[] for _ in configs]
    for s_idx, seed in enumerate(EVAL_SEEDS):
        seq = range(len(configs)) if s_idx % 2 == 0 else reversed(range(len(configs)))
        for i in seq:
            _, rtype, nw, lr, opts = configs[i]
            all_runs[i].append(run_training_job(rtype, nw, epochs=MAX_EPOCHS, lr=lr, seed=seed, **opts))
        print(f'[EXP2] finished seed {seed}', flush=True)

    summary_rows = []
    for (label, _, nw, lr, _), runs in zip(configs, all_runs):
        best_per_seed = [max(r, key=lambda e: e['val']) for r in runs]
        hits = [find_convergence_point(r, TARGET_VAL_ACC) for r in runs]
        valid_hits = [h for h in hits if h is not None]
        summary_rows.append(dict(
            method=label,
            workers=nw,
            lr=lr,
            reached_count=len(valid_hits),
            median_epochs=statistics.median(h[0] for h in valid_hits) if valid_hits else None,
            time_to_target=calc_mean_std(h[1] for h in valid_hits) if valid_hits else None,
            median_epoch_s=statistics.median(e['time'] for r in runs for e in r),
            test_acc=calc_mean_std(e['test'] for e in best_per_seed),
            runs=runs,
        ))
    return summary_rows


def main():
    layer_rows = bench_layers()
    conv_rows = bench_convergence()
    base = conv_rows[0]

    lines = [
        '# Thực nghiệm 2 (Bài toán 2): So sánh GPU (NVIDIA Quadro M620) và CPU (Intel Core i7-6820HQ)',
        '',
        f'Đo thực tế trên cùng một laptop. Phần 1 đo thời gian từng lớp ({TRIALS} lần lặp); '
        f'Phần 2 đo thời gian huấn luyện thực tế tới ngưỡng validation **{fmt_num(TARGET_VAL_ACC, 1)}%** trên Fashion-MNIST ({len(EVAL_SEEDS)} seed).',
    ]

    sections = (
        ('conv2d', '1. Lớp Tích chập 8 bộ lọc 5×5 trên ảnh 28×28', 'Batch'),
        ('matvec', '2. Lớp Kết nối đầy đủ n × n (1 vector Matvec, gộp bộ nhớ theo warp)', 'Kích thước n'),
        ('gemm',   '3. Nhân hai ma trận vuông n × n (Shared-memory Tiled GEMM 16×16)', 'Kích thước n'),
    )
    for op_key, title, col_name in sections:
        lines += [
            '', f'## {title}', '',
            f'| {col_name} | CPU 1 nhân (ms) | GPU kernel (ms) | GPU + PCIe (ms) | Tăng tốc (kernel) | Tăng tốc (+PCIe) |',
            '|--:|--:|--:|--:|--:|--:|',
        ]
        for r in layer_rows:
            if r['op'] == op_key:
                lines.append(
                    f"| {r['dim']} | {fmt_num(r['cpu_ms'], 4)} | {fmt_num(r['gpu_kern_ms'], 4)} "
                    f"| {fmt_num(r['gpu_e2e_ms'], 4)} | {fmt_num(r['speedup_kern'][0])} ± {fmt_num(r['speedup_kern'][1])}× "
                    f"| {fmt_num(r['speedup_e2e'][0])} ± {fmt_num(r['speedup_e2e'][1])}× |"
                )

    lines += [
        '',
        f'## 4. Thời gian huấn luyện tới cùng ngưỡng Validation Accuracy ({fmt_num(TARGET_VAL_ACC, 1)}%) trên Fashion-MNIST',
        '',
        '| Phương pháp | Worker | lr | Seed đạt | Epoch cần | Thời gian tới ngưỡng (s) | Tăng tốc tới ngưỡng | Tăng tốc mỗi epoch | Test Accuracy |',
        '|---|--:|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for r in conv_rows:
        t_str = sp_str = ep_str = '–'
        if r['reached_count'] > 0:
            m, sd = r['time_to_target']
            ep_str = f"{r['median_epochs']:g}".replace('.', ',')
            if r['reached_count'] == len(EVAL_SEEDS):
                t_str = f'{fmt_num(m)} ± {fmt_num(sd)}'
                sp_str = fmt_num(base['time_to_target'][0] / m) + '×'
            else:
                t_str = f"{fmt_num(m)} ({r['reached_count']} seed)"
        ep_sp = base['median_epoch_s'] / r['median_epoch_s']
        lines.append(
            f"| {r['method']} | {r['workers']} | {fmt_num(r['lr'], 2)} | {r['reached_count']}/{len(EVAL_SEEDS)} "
            f"| {ep_str} | {t_str} | {sp_str} | {fmt_num(ep_sp)}× "
            f"| {fmt_num(r['test_acc'][0])} ± {fmt_num(r['test_acc'][1])}% |"
        )

    export_report('exp2_gpu_vs_cpu', dict(layers=layer_rows, convergence=conv_rows), lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
