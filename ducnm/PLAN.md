# Kế hoạch thực nghiệm và giả thuyết định lượng (`ducnm/`)

Tài liệu thiết kế 4 nhóm thực nghiệm (`EXP1`–`EXP4`) giải quyết trực tiếp 3 bài toán trọng tâm của học phần Tính toán hiệu năng cao dựa trên bài báo V. Hegde & S. Usmani (*Parallel and Distributed Deep Learning*, Stanford, 2016):

1. **Bài toán 1 (`EXP1`)**: Trong huấn luyện CNN, phép nhân ma trận (`GEMM`) có phải là bottleneck chính không?
2. **Bài toán 2 (`EXP2`)**: GPU mạnh hơn CPU bao nhiêu lần ở từng phép toán cơ bản (`Conv2D`, `Matvec`, `Tiled GEMM`) và trong toàn bộ quá trình hội tụ thực tế? Khi nào GPU không thắng CPU?
3. **Bài toán 3 (`EXP3`, `EXP4`)**: Chi phí truyền thông (`Communication Cost`) trong Gradient Descent phân tán tuân theo quy luật nào theo kích thước thông điệp, thuật toán Allreduce, kích thước batch và độ trễ đạo hàm (`Staleness`)?

---

## 1. Cấu hình phần cứng và môi trường đo thực tế

- **CPU**: Intel Core i7-6820HQ @ 2.70 GHz (`4` nhân vật lý / `8` luồng logic)
- **GPU**: NVIDIA Quadro M620 (`2 GB` GDDR5, kiến trúc Maxwell `sm_50`, `512` nhân CUDA, TDP `30 W`, driver 553.46 / CUDA 12.4)
- **Trình biên dịch**: Zig Clang 22.1.8 (`-std=gnu11 -O3 -ffast-math`), NVRTC 12.4 (`--gpu-architecture=compute_50`)
- **Bộ dữ liệu**: Fashion-MNIST (`10.000` ảnh `28×28` chia thành `7.000` train, `1.000` validation, `2.000` test), `batch = 64`, mạng CNN `54.314` tham số (`217.256` byte FP32).

---

## 2. Ma trận thực nghiệm và dự đoán trước khi chạy

| Mã | Bài toán | Giả thuyết định lượng | Script kiểm chứng & Báo cáo |
|---|---|---|---|
| **H1.1** | Bottleneck GEMM | Trong một bước huấn luyện CNN (Forward + Backward + SGD), các phép nhân ma trận (`Conv GEMM + Dense GEMM`) chiếm **`> 98%` tổng số FLOP** ở mọi kích thước batch | `benchmarks/exp1_matmul_bottleneck.py` → `results/exp1_matmul_bottleneck.md` |
| **H1.2** | Bottleneck GEMM | Biến đổi tích chập sang `im2col + GEMM` (vòng lặp `i-k-j` tự động vector hóa AVX2) giúp bước huấn luyện CPU nhanh hơn tích chập trực tiếp **`3,0–3,6` lần** với đạo hàm trùng khớp (`< 10^{-9}` ở FP64) | `bin/verify_grad.exe` & `results/exp1_matmul_bottleneck.md` |
| **H1.3** | Bottleneck GEMM | Riêng phép nhân ma trận chiếm **`60–75%` thời gian thực thi**, và cụm `im2col + GEMM` chiếm **`85–95%` thời gian** trên cả CPU lẫn GPU (các phép `MaxPool`, `ReLU`, `Softmax`, `SGD` chỉ chiếm `5–15%`) | `results/exp1_matmul_bottleneck.md` |
| **H2.1** | GPU vs CPU | Ở kích thước nhỏ (`batch = 1` với Conv2D hoặc `n ≤ 512` với Matvec 1 vector), GPU **chậm hơn CPU** (`< 1,0×`) do độ trễ khởi chạy kernel trên WDDM (~45–75 µs) lớn hơn thời gian tính toán | `benchmarks/exp2_gpu_vs_cpu.py` → `results/exp2_gpu_vs_cpu.md` |
| **H2.2** | GPU vs CPU | Khi kích thước đủ lớn (`batch ≥ 64`, `n ≥ 2048`), GPU Quadro M620 vượt CPU **`2,4–3,1` lần** ở Conv2D/Matvec (nhờ gộp truy cập bộ nhớ theo warp) và **`4,0–8,6` lần** ở `16×16` Shared-Memory Tiled GEMM | `results/exp2_gpu_vs_cpu.md` |
| **H2.3** | GPU vs CPU | Tới cùng ngưỡng Validation Accuracy `84,0%` trên Fashion-MNIST, GPU Quadro M620 nhanh hơn CPU 1 nhân (Conv trực tiếp) **`6,0–7,5` lần** và nhanh hơn CPU 1 nhân (`im2col + GEMM`) **`1,6–2,0` lần** | `results/exp2_gpu_vs_cpu.md` |
| **H3.1** | Communication Cost | Với thông điệp nhỏ (`4–16 KB`, bị chi phối bởi độ trễ khởi tạo $\alpha$), `rd` (Recursive Doubling, $\log_2 p$ bước) nhanh nhất; với thông điệp lớn (`≥ 64 KB`, gồm vector đạo hàm `212 KB` tại $p \ge 4$), **`ring` nhanh nhất** nhờ truyền băng thông tối ưu $2\frac{p-1}{p} M\beta$ | `benchmarks/exp3_comm_allreduce.py` → `results/exp3_comm_allreduce.md` |
| **H3.2** | Communication Cost | Khi tăng `batch size` từ `16` lên `256` trên 4 worker, số lần gọi Allreduce mỗi epoch ($\lceil N/B \rceil$) giảm `16` lần, làm tỉ trọng thời gian truyền thông/chờ giảm từ **`~15–18%` xuống dưới `10%`** | `results/exp3_comm_allreduce.md` |
| **H4.1** | Scaling & Staleness | Đa luồng `Sync-Ring` trên `4` nhân đạt tăng tốc **`2,5–2,9` lần** so với 1 nhân `im2col+GEMM` (**`8,5–9,0` lần** so với 1 nhân Conv trực tiếp), với chi phí chờ/truyền thông khoảng **`10%`** ở `batch = 64` | `benchmarks/exp4_dist_sgd_staleness.py` → `results/exp4_dist_sgd_staleness.md` |
| **H4.2** | Scaling & Staleness | Trong huấn luyện có đạo hàm trễ $d$ bước, khi tích $d \times \eta \ge 0,60$ ($\eta = \text{lr}/(1-\text{momentum})$), mô hình phân kỳ về mức đoán ngẫu nhiên (`~10%`); giảm $\eta$ xuống `0,02` giúp mạng hội tụ ổn định ngay cả ở $d = 7$ | `results/exp4_dist_sgd_staleness.md` |
