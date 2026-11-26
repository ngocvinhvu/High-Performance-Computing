# Thực nghiệm 1 (Bài toán 1): Phép nhân ma trận (GEMM) là bottleneck trong CNN

Đo thực tế 3 lần (lấy trung vị) cho một bước huấn luyện hoàn chỉnh (Forward + Backward + Momentum SGD) của mạng CNN 54.314 tham số trên dữ liệu Fashion-MNIST.

## 1. Tỉ trọng FLOP và phân rã thời gian trên CPU (`cnn_cpu.exe`, C `-O3 -ffast-math`)

| Batch | Tỉ trọng GEMM FLOPs | Conv trực tiếp (ms) | `im2col + GEMM` (ms) | Tăng tốc nhờ GEMM | Thời gian GEMM (ms) | Tỉ trọng GEMM (%) | `im2col/col2im` (%) | Pool/ReLU/SGD (%) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 98,96% | 15,51 | 4,97 | **3,12×** | 3,36 | **68,9%** | 18,2% | 12,9% |
| 64 | 99,18% | 63,05 | 17,93 | **3,52×** | 12,31 | **68,8%** | 18,3% | 12,9% |
| 128 | 99,21% | 123,44 | 37,78 | **3,27×** | 26,22 | **69,4%** | 17,8% | 12,8% |
| 256 | 99,23% | 251,79 | 72,73 | **3,46×** | 50,13 | **69,0%** | 18,2% | 12,8% |

## 2. Phân rã thời gian kernel trên GPU (`cnn_gpu.exe`, NVIDIA Quadro M620, Tiled GEMM 16×16)

| Batch | Tổng kernel (ms) | Conv GEMM (ms) | Dense GEMM (ms) | Tỉ trọng GEMM (%) | `im2col` (%) | Pool/ReLU/Softmax/SGD (%) |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 2,530 | 1,280 | 0,224 | **59,4%** | 27,8% | 12,8% |
| 64 | 8,280 | 4,774 | 0,383 | **62,3%** | 30,4% | 7,3% |
| 128 | 14,461 | 8,508 | 0,539 | **62,6%** | 31,1% | 6,3% |
| 256 | 24,612 | 14,693 | 0,824 | **63,0%** | 31,2% | 5,7% |
