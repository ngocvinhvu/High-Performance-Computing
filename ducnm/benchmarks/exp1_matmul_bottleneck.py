"""Experiment 1 (Bài toán 1): Chứng minh phép nhân ma trận (GEMM) là bottleneck trong CNN.
Đo thực tế trên CPU (cnn_cpu.exe --profile-gemm) và GPU Quadro M620 (cnn_gpu.exe --profile-gemm).
"""
import re
import statistics

from runner_utils import bin_path, export_report, fmt_num, invoke_binary

TRIALS = 3


def main():
    cpu_pat = re.compile(
        r'\[CPU_PROF\] batch=(\d+) naive_ms=(\S+) gemm_ms=(\S+) conv_mm_ms=(\S+) '
        r'dense_mm_ms=(\S+) unfold_ms=(\S+) act_pool_ms=(\S+) loss_sgd_ms=(\S+) '
        r'conv_mm_flops=(\S+) dense_mm_flops=(\S+) misc_flops=(\S+)'
    )
    gpu_pat = re.compile(
        r'\[GPU_PROF\] batch=(\d+) conv_mm_ms=(\S+) dense_mm_ms=(\S+) '
        r'unfold_ms=(\S+) act_pool_ms=(\S+) loss_sgd_ms=(\S+)'
    )

    cpu_data, gpu_data = {}, {}
    for _ in range(TRIALS):
        out_c = invoke_binary(1, [bin_path('cnn_cpu'), '--profile-gemm'])
        for m in cpu_pat.findall(out_c):
            cpu_data.setdefault(int(m[0]), []).append([float(x) for x in m[1:]])

        out_g = invoke_binary(1, [bin_path('cnn_gpu'), '--profile-gemm'])
        for m in gpu_pat.findall(out_g):
            gpu_data.setdefault(int(m[0]), []).append([float(x) for x in m[1:]])

    cpu_summary = []
    for bsz, samples in sorted(cpu_data.items()):
        med = [statistics.median(col) for col in zip(*samples)]
        naive_ms, gemm_ms, c_mm, d_mm, unfold, pool, l_sgd, cf, df, mf = med
        tot_mm = c_mm + d_mm
        tot_t = tot_mm + unfold + pool + l_sgd
        tot_f = cf + df + mf
        cpu_summary.append(dict(
            batch=bsz,
            naive_step_ms=naive_ms,
            im2col_gemm_step_ms=gemm_ms,
            im2col_speedup=naive_ms / gemm_ms,
            conv_mm_ms=c_mm,
            dense_mm_ms=d_mm,
            total_mm_ms=tot_mm,
            unfold_ms=unfold,
            act_pool_ms=pool,
            loss_sgd_ms=l_sgd,
            mm_time_share_pct=100.0 * tot_mm / tot_t,
            unfold_time_share_pct=100.0 * unfold / tot_t,
            other_time_share_pct=100.0 * (pool + l_sgd) / tot_t,
            mm_flop_share_pct=100.0 * (cf + df) / tot_f,
        ))

    gpu_summary = []
    for bsz, samples in sorted(gpu_data.items()):
        med = [statistics.median(col) for col in zip(*samples)]
        c_mm, d_mm, unfold, pool, l_sgd = med
        tot_mm = c_mm + d_mm
        tot_t = tot_mm + unfold + pool + l_sgd
        gpu_summary.append(dict(
            batch=bsz,
            conv_mm_ms=c_mm,
            dense_mm_ms=d_mm,
            total_mm_ms=tot_mm,
            unfold_ms=unfold,
            act_pool_ms=pool,
            loss_sgd_ms=l_sgd,
            total_kernel_ms=tot_t,
            mm_time_share_pct=100.0 * tot_mm / tot_t,
            unfold_time_share_pct=100.0 * unfold / tot_t,
            other_time_share_pct=100.0 * (pool + l_sgd) / tot_t,
        ))

    lines = [
        '# Thực nghiệm 1 (Bài toán 1): Phép nhân ma trận (GEMM) là bottleneck trong CNN',
        '',
        f'Đo thực tế {TRIALS} lần (lấy trung vị) cho một bước huấn luyện hoàn chỉnh (Forward + Backward + Momentum SGD) '
        'của mạng CNN 54.314 tham số trên dữ liệu Fashion-MNIST.',
        '',
        '## 1. Tỉ trọng FLOP và phân rã thời gian trên CPU (`cnn_cpu.exe`, C `-O3 -ffast-math`)',
        '',
        '| Batch | Tỉ trọng GEMM FLOPs | Conv trực tiếp (ms) | `im2col + GEMM` (ms) | Tăng tốc nhờ GEMM | Thời gian GEMM (ms) | Tỉ trọng GEMM (%) | `im2col/col2im` (%) | Pool/ReLU/SGD (%) |',
        '|--:|--:|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for r in cpu_summary:
        lines.append(
            f"| {r['batch']} | {fmt_num(r['mm_flop_share_pct'], 2)}% | {fmt_num(r['naive_step_ms'], 2)} "
            f"| {fmt_num(r['im2col_gemm_step_ms'], 2)} | **{fmt_num(r['im2col_speedup'], 2)}×** "
            f"| {fmt_num(r['total_mm_ms'], 2)} | **{fmt_num(r['mm_time_share_pct'], 1)}%** "
            f"| {fmt_num(r['unfold_time_share_pct'], 1)}% | {fmt_num(r['other_time_share_pct'], 1)}% |"
        )

    lines += [
        '',
        '## 2. Phân rã thời gian kernel trên GPU (`cnn_gpu.exe`, NVIDIA Quadro M620, Tiled GEMM 16×16)',
        '',
        '| Batch | Tổng kernel (ms) | Conv GEMM (ms) | Dense GEMM (ms) | Tỉ trọng GEMM (%) | `im2col` (%) | Pool/ReLU/Softmax/SGD (%) |',
        '|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for r in gpu_summary:
        lines.append(
            f"| {r['batch']} | {fmt_num(r['total_kernel_ms'], 3)} | {fmt_num(r['conv_mm_ms'], 3)} "
            f"| {fmt_num(r['dense_mm_ms'], 3)} | **{fmt_num(r['mm_time_share_pct'], 1)}%** "
            f"| {fmt_num(r['unfold_time_share_pct'], 1)}% | {fmt_num(r['other_time_share_pct'], 1)}% |"
        )

    export_report('exp1_matmul_bottleneck', dict(cpu=cpu_summary, gpu=gpu_summary), lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
