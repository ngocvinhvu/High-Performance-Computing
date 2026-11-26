"""Experiment 3 (Bài toán 3a): Phân tích chi phí truyền thông (Communication Cost),
so sánh 4 thuật toán Allreduce (ring, tree, rd, naive), khớp mô hình Hockney alpha-beta
và quét Global Batch Size trên Fashion-MNIST.
"""
import math
import re

from runner_utils import SUMMARY_PATTERN, bin_path, export_report, fmt_num, invoke_binary


def hockney_cost_us(method: str, nw: int, nbytes: int, alpha_us: float, beta_us_per_byte: float) -> float:
    if nw <= 1:
        return 0.0
    steps_lg = math.ceil(math.log2(nw))
    if method == 'ring':
        return 2.0 * (nw - 1) * alpha_us + 2.0 * ((nw - 1) / nw) * nbytes * beta_us_per_byte
    if method == 'rd':
        if (nw & (nw - 1)) != 0:
            return 2.0 * steps_lg * (alpha_us + nbytes * beta_us_per_byte)
        return steps_lg * (alpha_us + nbytes * beta_us_per_byte)
    if method == 'tree':
        return 2.0 * steps_lg * (alpha_us + nbytes * beta_us_per_byte)
    if method == 'naive':
        return 2.0 * (nw - 1) * (alpha_us + nbytes * beta_us_per_byte)
    return 0.0


def run_allreduce_bench():
    pat = re.compile(r'\[COMM_BENCH\] method=(\w+) workers=(\d+) msg_bytes=(\d+) latency_us=(\S+)')
    out = invoke_binary(4, [bin_path('cnn_dist'), '--bench-comm'])
    records = [
        dict(method=m, workers=int(w), msg_bytes=int(b), latency_us=float(t))
        for m, w, b, t in pat.findall(out)
    ]

    p2_ring = {r['msg_bytes']: r['latency_us'] for r in records if r['method'] == 'ring' and r['workers'] == 2}
    alpha_us = max(1.0, p2_ring[4096] / 2.0)
    beta_us = max(1e-6, (p2_ring[16777216] - 2.0 * alpha_us) / 16777216.0)
    bw_gbps = (1.0 / (beta_us * 1e-6)) / 1e9

    for r in records:
        r['pred_us'] = hockney_cost_us(r['method'], r['workers'], r['msg_bytes'], alpha_us, beta_us)
    return records, dict(alpha_us=alpha_us, beta_us_per_byte=beta_us, bandwidth_GBps=bw_gbps)


def run_batch_sweep():
    batch_list = (16, 32, 64, 128, 256)
    rows = []
    for bsz in batch_list:
        out_cpu = invoke_binary(1, [bin_path('cnn_cpu'), '--epochs', '2', '--batch', str(bsz), '--seed', '11'])
        t_cpu = float(SUMMARY_PATTERN.search(out_cpu).group(1))

        per_method = {}
        for m in ('ring', 'tree', 'rd', 'naive'):
            out_d = invoke_binary(4, [
                bin_path('cnn_dist'), '--workers', '4', '--strategy', 'sync',
                '--collective', m, '--epochs', '2', '--batch', str(bsz), '--seed', '11'
            ])
            tot_s, comm_s = [float(x) for x in SUMMARY_PATTERN.search(out_d).groups()[:2]]
            per_method[m] = dict(
                total_s=tot_s,
                comm_s=comm_s,
                comm_share_pct=100.0 * comm_s / tot_s,
                speedup=t_cpu / tot_s,
            )
        rows.append(dict(batch=bsz, cpu_time_s=t_cpu, dist_p4=per_method))
    return rows


def main():
    records, fit = run_allreduce_bench()
    sweep = run_batch_sweep()

    lookup = {(r['workers'], r['msg_bytes'], r['method']): r for r in records}
    msg_sizes = sorted({r['msg_bytes'] for r in records})

    lines = [
        '# Thực nghiệm 3 (Bài toán 3a): Chi phí truyền thông Allreduce và ảnh hưởng của Batch Size',
        '',
        f"Tham số mô hình Hockney $\\alpha$–$\\beta$ khớp từ số đo thực tế: "
        f"độ trễ khởi tạo $\\alpha \\approx {fmt_num(fit['alpha_us'], 2)}$ µs, "
        f"băng thông hiệu dụng $B = 1/\\beta \\approx {fmt_num(fit['bandwidth_GBps'], 2)}$ GB/s.",
        '',
        '## 1. Thời gian Allreduce thực tế (µs) theo kích thước thông điệp và thuật toán',
    ]

    for nw in (2, 4, 8):
        lines += [
            '', f'### Số worker p = {nw}', '',
            '| Kích thước | `ring` (µs) | `tree` (µs) | `rd` (µs) | `naive` (µs) | Dự đoán `ring` α–β (µs) | Nhanh nhất |',
            '|--:|--:|--:|--:|--:|--:|---|',
        ]
        for sz in msg_sizes:
            tag = '212 KB (CNN)' if sz == 217256 else (f'{sz // 1024} KB' if sz < 1048576 else f'{sz // 1048576} MB')
            lat = {m: lookup[nw, sz, m]['latency_us'] for m in ('ring', 'tree', 'rd', 'naive')}
            winner = min(lat, key=lat.get)
            pred_r = lookup[nw, sz, 'ring']['pred_us']
            lines.append(
                f"| {tag} | {fmt_num(lat['ring'], 1)} | {fmt_num(lat['tree'], 1)} "
                f"| {fmt_num(lat['rd'], 1)} | {fmt_num(lat['naive'], 1)} "
                f"| {fmt_num(pred_r, 1)} | **{winner}** |"
            )

    lines += [
        '',
        '## 2. Quét Global Batch Size với 4 worker (Fashion-MNIST, 2 epoch)',
        '',
        r'Khi tăng kích thước batch $B$, số lần đồng bộ đạo hàm mỗi epoch ($\lceil N/B \rceil$) giảm tỉ lệ nghịch với $B$, '
        'làm tỉ trọng truyền thông giảm đơn điệu:',
        '',
        '| Batch | CPU 1 nhân (s) | Sync-Ring 4 worker (s) | Tỉ trọng truyền/chờ Ring (%) | Tăng tốc (`ring`) | Tăng tốc (`tree`) | Tăng tốc (`rd`) | Tăng tốc (`naive`) |',
        '|--:|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for r in sweep:
        d = r['dist_p4']
        lines.append(
            f"| {r['batch']} | {fmt_num(r['cpu_time_s'], 3)} | {fmt_num(d['ring']['total_s'], 3)} "
            f"| **{fmt_num(d['ring']['comm_share_pct'], 1)}%** | **{fmt_num(d['ring']['speedup'], 2)}×** "
            f"| {fmt_num(d['tree']['speedup'], 2)}× | {fmt_num(d['rd']['speedup'], 2)}× "
            f"| {fmt_num(d['naive']['speedup'], 2)}× |"
        )

    export_report('exp3_comm_allreduce', dict(hockney_fit=fit, allreduce=records, batch_sweep=sweep), lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
