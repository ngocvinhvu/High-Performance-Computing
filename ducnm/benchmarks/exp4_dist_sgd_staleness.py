"""Experiment 4 (Bài toán 3b): Strong Scaling theo số worker (p = 1, 2, 4) và
ảnh hưởng của độ trễ đạo hàm (Gradient Staleness d x bước hiệu dụng eta) trên GPU Quadro M620.
"""
import statistics

from runner_utils import (
    SUMMARY_PATTERN, bin_path, calc_mean_std, export_report,
    fmt_num, invoke_binary, run_training_job
)

SCALING_TRIALS = 3
SCALING_EPOCHS = 3
STALE_GAPS = (0, 3, 7, 15)
MOMENTA = (0.0, 0.9)
EFF_STEPS = (0.02, 0.05, 0.10, 0.20)
STALE_SEEDS = (11, 12)
STALE_EPOCHS = 4


def measure_strong_scaling():
    configs = [
        ('CPU 1 nhân (Conv trực tiếp)', 1, [bin_path('cnn_cpu'), '--naive-conv']),
        ('CPU 1 nhân (im2col + GEMM)',  1, [bin_path('cnn_cpu')]),
        ('Đa luồng Sync-Ring (p=1)',    1, [bin_path('cnn_dist'), '--workers', '1', '--strategy', 'sync', '--collective', 'ring']),
        ('Đa luồng Sync-Ring (p=2)',    2, [bin_path('cnn_dist'), '--workers', '2', '--strategy', 'sync', '--collective', 'ring']),
        ('Đa luồng Sync-Ring (p=4)',    4, [bin_path('cnn_dist'), '--workers', '4', '--strategy', 'sync', '--collective', 'ring']),
        ('GPU CUDA (Quadro M620)',      1, [bin_path('cnn_gpu')]),
    ]
    samples = [[] for _ in configs]
    for r in range(SCALING_TRIALS):
        seq = range(len(configs)) if r % 2 == 0 else reversed(range(len(configs)))
        for i in seq:
            _, nw, cmd = configs[i]
            out = invoke_binary(nw, cmd + ['--epochs', str(SCALING_EPOCHS), '--lr', '0.02', '--seed', '11'])
            tot_s, comm_s = [float(x) for x in SUMMARY_PATTERN.search(out).groups()[:2]]
            samples[i].append((tot_s, comm_s))
        print(f'[EXP4] scaling trial {r + 1}/{SCALING_TRIALS} done', flush=True)

    rows = []
    t_naive = statistics.median(t for t, _ in samples[0])
    t_gemm  = statistics.median(t for t, _ in samples[1])
    for (label, nw, _), runs in zip(configs, samples):
        med_t = statistics.median(t for t, _ in runs)
        rows.append(dict(
            method=label,
            workers=nw,
            runs=runs,
            median_s=med_t,
            min_s=min(t for t, _ in runs),
            max_s=max(t for t, _ in runs),
            comm_share_pct=100.0 * statistics.median(c / t for t, c in runs),
            speedup_vs_naive=t_naive / med_t,
            speedup_vs_gemm=t_gemm / med_t,
        ))
    return rows


def measure_staleness_grid():
    grid_rows = []
    for mom in MOMENTA:
        for eta in EFF_STEPS:
            for gap in STALE_GAPS:
                raw_lr = round(eta * (1.0 - mom), 8)
                runs = [
                    run_training_job('gpu', epochs=STALE_EPOCHS, lr=raw_lr, momentum=mom, staleness=gap, seed=s)
                    for s in STALE_SEEDS
                ]
                grid_rows.append(dict(
                    momentum=mom,
                    eff_step=eta,
                    lr=raw_lr,
                    staleness=gap,
                    val_acc=calc_mean_std(r[-1]['val'] for r in runs),
                    runs=runs,
                ))
            print(f'[EXP4] staleness mom={mom} eta={eta} done', flush=True)
    return grid_rows


def main():
    scaling_rows = measure_strong_scaling()
    stale_rows = measure_staleness_grid()
    lookup = {(r['momentum'], r['eff_step'], r['staleness']): r for r in stale_rows}

    lines = [
        '# Thực nghiệm 4 (Bài toán 3b): Strong Scaling và độ trễ đạo hàm (Gradient Staleness)',
        '',
        f'## 1. Strong Scaling trên 7.000 ảnh Fashion-MNIST (`batch = 64`, {SCALING_EPOCHS} epoch, {SCALING_TRIALS} lần lặp)',
        '',
        '| Phương pháp | Worker (p) | Thời gian trung vị (s) | Khoảng (min–max) | Tăng tốc (vs trực tiếp) | Tăng tốc (vs GEMM) | Hiệu suất (vs GEMM) | Tỉ trọng chờ/truyền |',
        '|---|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for r in scaling_rows:
        is_dist = 'Sync-Ring' in r['method']
        eff_str = fmt_num(100.0 * r['speedup_vs_gemm'] / r['workers'], 1) + '%' if is_dist else '–'
        comm_str = fmt_num(r['comm_share_pct'], 1) + '%' if is_dist else '–'
        lines.append(
            f"| {r['method']} | {r['workers']} | {fmt_num(r['median_s'], 3)} "
            f"| {fmt_num(r['min_s'], 3)}–{fmt_num(r['max_s'], 3)} | {fmt_num(r['speedup_vs_naive'])}× "
            f"| {fmt_num(r['speedup_vs_gemm'])}× | {eff_str} | {comm_str} |"
        )

    lines += [
        '',
        f'## 2. Ảnh hưởng của độ trễ đạo hàm (Staleness d) theo bước hiệu dụng η trên GPU Quadro M620 ({STALE_EPOCHS} epoch)',
        '',
        'Đạo hàm tính tại bước $t$ được áp dụng tại bước $t + d$ (tương đương $W = d + 1$ worker bất đồng bộ):',
    ]
    for mom in MOMENTA:
        lines += [
            '', f'### Momentum = {fmt_num(mom, 1)}', '',
            '| Bước hiệu dụng η | lr thực | ' + ' | '.join(f'd = {d}' for d in STALE_GAPS) + ' |',
            '|--:|--:|' + '--:|' * len(STALE_GAPS),
        ]
        for eta in EFF_STEPS:
            cells = [lookup[mom, eta, d]['val_acc'] for d in STALE_GAPS]
            lr_val = lookup[mom, eta, 0]['lr']
            lines.append(
                f'| {fmt_num(eta)} | {fmt_num(lr_val, 3)} | '
                + ' | '.join(f'{fmt_num(m, 1)} ± {fmt_num(s, 1)}%' for m, s in cells) + ' |'
            )

    export_report('exp4_dist_sgd_staleness', dict(strong_scaling=scaling_rows, staleness_grid=stale_rows), lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
