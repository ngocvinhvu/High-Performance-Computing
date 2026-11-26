# Thực nghiệm Tính toán Hiệu năng cao với CNN: Bottleneck GEMM, GPU so với CPU và Chi phí Truyền thông (`ducnm/`)

Thư mục `ducnm/` xây dựng bộ engine huấn luyện và benchmark mạng nơ-ron tích chập (CNN) từ đầu bằng **C11, Đa luồng Message-Passing và CUDA Runtime Compilation (NVRTC)** trên bộ dữ liệu thực **Fashion-MNIST**, tập trung chứng minh bằng số đo thực tế 3 bài toán trọng tâm của học phần:

1. **Bài toán 1 (`EXP1`) — Trong CNN, phép nhân ma trận (`GEMM`) là bottleneck**: Cài đặt song song hai cơ chế tích chập (vòng lặp trực tiếp và `im2col + GEMM` tự động vector hóa SIMD trên CPU / `16×16` shared-memory tiled GEMM trên GPU), đo chính xác tỉ trọng FLOP và phân rã thời gian từng pha.
2. **Bài toán 2 (`EXP2`) — GPU mạnh hơn CPU trong huấn luyện (và điều kiện để GPU thắng)**: Đo vi chuẩn (microbenchmark) từng lớp mạng (`Conv2D`, `Warp-coalesced Matvec`, `Tiled GEMM`) tách biệt giữa thời gian **chỉ chạy kernel** và thời gian **gồm truyền PCIe H2D/D2H**, kết hợp đo thời gian hội tụ thực tế tới cùng ngưỡng accuracy `84,0%`.
3. **Bài toán 3 (`EXP3`, `EXP4`) — Phân tích Communication Cost trong Gradient Descent phân tán**: Cài đặt **4 thuật toán Allreduce** (`ring`, `tree`, `rd` recursive doubling, `naive` star parameter-server) và **3 chiến lược SGD phân tán** (`sync`, `local_avg`, `ps_async`), khớp mô hình **Hockney $\alpha$–$\beta$**, quét `batch size` (`16..256`), đo **Strong Scaling** (`p = 1, 2, 4`) và khảo sát ngưỡng phân kỳ do **độ trễ đạo hàm ($d \times \eta$)** trên GPU.

---

## 1. Cấu trúc thư mục

```text
ducnm/
├── README.md                              Tài liệu phương pháp, hướng dẫn biên dịch và tổng hợp kết quả
├── PLAN.md                                Kế hoạch thực nghiệm và các giả thuyết định lượng (H1.1–H4.2)
├── build_engine.py                        Trình biên dịch tự động 4 binary vào bin/ (GCC hoặc Zig Clang + NVRTC)
├── .gitignore
├── data/                                  Dữ liệu Fashion-MNIST (t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte)
├── bin/                                   Các tệp thực thi sau khi build (cnn_cpu.exe, cnn_dist.exe, cnn_gpu.exe, verify_grad.exe)
├── src/
│   ├── core/
│   │   ├── sys_timer.h                    Đồng hồ độ phân giải cao, cấp phát bộ nhớ, PRNG Xorshift64*
│   │   ├── lenet_gemm.h                   Định nghĩa kiến trúc CNN 54.314 tham số & cấu trúc GemmProfile
│   │   ├── lenet_gemm.c                   Lan truyền thuận/ngược: tích chập trực tiếp & im2col + SIMD GEMM
│   │   └── fmnist_loader.h                Bộ đọc định dạng IDX Fashion-MNIST, đánh giá accuracy và parser CLI
│   ├── cpu_runner.c                       Huấn luyện tuần tự trên CPU & profiler phân rã pha GEMM (--profile-gemm)
│   ├── dist_worker.c                      Huấn luyện đa worker (sync / local_avg / async) & benchmark 4 thuật toán Allreduce
│   ├── gpu_driver.h                       Cơ chế nạp động Win32 cho nvcuda.dll và nvrtc64_120_0.dll
│   ├── gpu_kernels.cu                     14 kernel CUDA: 16x16 tiled GEMM, im2col, warp-shuffle Matvec, Conv/Pool/SGD
│   ├── gpu_runner.c                       Huấn luyện trên GPU, microbenchmark từng lớp (--bench), profiler & mô phỏng trễ
│   └── verify_grad.c                      Kiểm thử đạo hàm sai phân hữu hạn FP64 & kiểm chứng tương đương im2col+GEMM
├── benchmarks/
│   ├── runner_utils.py                    Gán affinity lõi CPU vật lý, điều phối tiến trình và xuất báo cáo Markdown/JSON
│   ├── exp1_matmul_bottleneck.py          Thực nghiệm 1: Phân tích bottleneck nhân ma trận trên CPU & GPU
│   ├── exp2_gpu_vs_cpu.py                 Thực nghiệm 2: So sánh GPU vs CPU theo từng lớp và thời gian hội tụ tới 84,0%
│   ├── exp3_comm_allreduce.py             Thực nghiệm 3: Đo 4 thuật toán Allreduce, khớp mô hình α–β và quét Batch Size
│   └── exp4_dist_sgd_staleness.py         Thực nghiệm 4: Strong Scaling đa nhân và lưới độ trễ đạo hàm d × η trên GPU
└── results/
    ├── exp1_matmul_bottleneck.{md,json}
    ├── exp2_gpu_vs_cpu.{md,json}
    ├── exp3_comm_allreduce.{md,json}
    └── exp4_dist_sgd_staleness.{md,json}
```

---

## 2. Hướng dẫn biên dịch và chạy lại

**Môi trường thực nghiệm:** Windows 10, CPU **Intel Core i7-6820HQ @ 2.70 GHz** (`4` nhân vật lý / `8` luồng), GPU **NVIDIA Quadro M620 2 GB** (`sm_50`, driver CUDA 12.4), Python 3.12, `ziglang` (Clang 22.1.8) và `nvidia-cuda-nvrtc-cu12==12.4.127`.

### Bước 1: Chuẩn bị công cụ biên dịch

```powershell
pip install ziglang nvidia-cuda-nvrtc-cu12==12.4.127
```

### Bước 2: Biên dịch và kiểm tra tính đúng đắn toán học

```powershell
cd ducnm
python build_engine.py

# 1. Kiểm tra đạo hàm bằng sai phân hữu hạn (FP64) & đối chiếu im2col+GEMM với tích chập trực tiếp
.\bin\verify_grad.exe

# 2. Kiểm tra đạo hàm tính trên GPU Quadro M620 so với CPU
.\bin\cnn_gpu.exe --verify
```

**Kết quả kiểm tra trên máy:**
- `verify_grad.exe`: Sai số tương đối lớn nhất ở 4 khối mạng lần lượt là `6,562e-09`, `1,976e-08`, `1,262e-07`, `5,131e-08` (**OK**); sai khác giữa `im2col + GEMM` và tích chập trực tiếp ở FP64 là **`9,955e-17`** (**OK** — trùng khớp tới giới hạn dấu phẩy động).
- `cnn_gpu.exe --verify`: Sai số tương đối giữa vector đạo hàm GPU (`Quadro M620`) và CPU là **`1,069e-07`** (**OK**).

### Bước 3: Chạy các chương trình huấn luyện và bộ thực nghiệm `EXP1`–`EXP4`

```powershell
# Huấn luyện thử trên từng thiết bị / chế độ (3 epoch)
.\bin\cnn_cpu.exe --epochs 3
.\bin\cnn_cpu.exe --naive-conv --epochs 3
.\bin\cnn_dist.exe --workers 4 --strategy sync --collective ring --epochs 3
.\bin\cnn_dist.exe --workers 4 --strategy avg --collective ring --epochs 3
.\bin\cnn_dist.exe --workers 4 --strategy async --epochs 3
.\bin\cnn_gpu.exe --epochs 3

# Tái lập toàn bộ kết quả trong thư mục results/
python benchmarks\exp1_matmul_bottleneck.py
python benchmarks\exp2_gpu_vs_cpu.py
python benchmarks\exp3_comm_allreduce.py
python benchmarks\exp4_dist_sgd_staleness.py
```

---

## 3. Kết quả thực nghiệm chi tiết

### 3.1 Thực nghiệm 1 (`EXP1`): Trong CNN, phép nhân ma trận (`GEMM`) là bottleneck

Báo cáo gốc: [results/exp1_matmul_bottleneck.md](results/exp1_matmul_bottleneck.md) · Dữ liệu JSON: [results/exp1_matmul_bottleneck.json](results/exp1_matmul_bottleneck.json).

**Bảng 1a — Tỉ trọng FLOP và phân rã thời gian một bước huấn luyện trên CPU (`cnn_cpu.exe --profile-gemm`):**

| Batch | Tỉ trọng GEMM FLOPs | Conv trực tiếp (ms) | `im2col + GEMM` (ms) | Tăng tốc nhờ GEMM | Thời gian GEMM (ms) | Tỉ trọng GEMM (%) | `im2col/col2im` (%) | Pool/ReLU/SGD (%) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | **98,96%** | 15,51 | 4,97 | **3,12×** | 3,36 | **68,9%** | 18,2% | 12,9% |
| 64 | **99,18%** | 63,05 | 17,93 | **3,52×** | 12,31 | **68,8%** | 18,3% | 12,9% |
| 128 | **99,21%** | 123,44 | 37,78 | **3,27×** | 26,22 | **69,4%** | 17,8% | 12,8% |
| 256 | **99,23%** | 251,79 | 72,73 | **3,46×** | 50,13 | **69,0%** | 18,2% | 12,8% |

**Bảng 1b — Phân rã thời gian kernel trên GPU (`cnn_gpu.exe --profile-gemm`, Quadro M620, Tiled GEMM `16×16`):**

| Batch | Tổng kernel (ms) | Conv GEMM (ms) | Dense GEMM (ms) | Tỉ trọng GEMM (%) | `im2col` (%) | Pool/ReLU/Softmax/SGD (%) |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 2,530 | 1,280 | 0,224 | **59,4%** | 27,8% | 12,8% |
| 64 | 8,280 | 4,774 | 0,383 | **62,3%** | 30,4% | 7,3% |
| 128 | 14,461 | 8,508 | 0,539 | **62,6%** | 31,1% | 6,3% |
| 256 | 24,612 | 14,693 | 0,824 | **63,0%** | 31,2% | 5,7% |

> **Kết luận Bài toán 1**:
> - Về mặt số phép tính, nhân ma trận (`Conv GEMM + Dense GEMM`) chiếm **`98,96%–99,23%` tổng số FLOP** của toàn mạng CNN.
> - Chuyển tích chập từ 7 vòng lặp lồng nhau sang `im2col + GEMM` giúp tận dụng tính cục bộ bộ nhớ và vector hóa SIMD, làm bước huấn luyện CPU **nhanh hơn `3,12–3,52` lần**.
> - Về thời gian thực thi, riêng `GEMM` chiếm **`68,8%–69,4%` trên CPU** và **`59,4%–63,0%` trên GPU**; nếu tính cả bước trải ma trận `im2col/col2im` phục vụ GEMM thì cụm này chiếm **`87,1%–87,2%` trên CPU** và **`87,2%–94,2%` trên GPU** (toàn bộ `MaxPool`, `ReLU`, `Softmax`, `SGD` chỉ chiếm `5,7%–12,8%` thời gian GPU).

---

### 3.2 Thực nghiệm 2 (`EXP2`): GPU mạnh hơn CPU trong huấn luyện và theo từng lớp

Báo cáo gốc: [results/exp2_gpu_vs_cpu.md](results/exp2_gpu_vs_cpu.md) · Dữ liệu JSON: [results/exp2_gpu_vs_cpu.json](results/exp2_gpu_vs_cpu.json).

**Bảng 2a — So sánh CPU 1 nhân và GPU Quadro M620 theo từng lớp mạng (`cnn_gpu.exe --bench`, trung vị 3 lần):**

| Phép toán / Lớp | Kích thước nhỏ | Kích thước trung bình | Kích thước lớn |
|---|---|---|---|
| **Conv2D `8@5×5` (`28×28`)** | `batch = 1`: **0,56×** kernel (`0,35×` +PCIe) — *GPU thua* | `batch = 64`: **2,39×** kernel (`1,73×` +PCIe) | `batch = 1024`: **2,84×** kernel (`2,19×` +PCIe) |
| **Matvec `n × n` (1 vector)** | `n = 256`: **0,16×** kernel (`0,07×` +PCIe) — *GPU thua* | `n = 1024`: **1,04×** kernel (`0,74×` +PCIe) | `n = 4096`: **3,14×** kernel (`2,97×` +PCIe) |
| **Tiled GEMM `n × n` (`16×16`)** | `n = 128`: **1,44×** | `n = 512`: **4,32×** | `n = 2048`: **8,56×** (`1952,1 ms` CPU vs `234,4 ms` GPU) |

**Bảng 2b — Thời gian huấn luyện tới cùng ngưỡng Validation Accuracy `84,0%` trên Fashion-MNIST (`3` seed, tối đa `6` epoch):**

| Phương pháp | Worker | lr | Seed đạt | Epoch cần | Thời gian tới `84,0%` (s) | Tăng tốc tới ngưỡng | Tăng tốc mỗi epoch | Test Accuracy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CPU 1 nhân (Conv trực tiếp) | 1 | 0,02 | 3/3 | 3 | 19,72 ± 6,87 | 1,00× | 1,00× | 85,25 ± 1,03% |
| CPU 1 nhân (`im2col + GEMM`) | 1 | 0,02 | 3/3 | 3 | 5,08 ± 1,20 | **3,88×** | 3,49× | 85,37 ± 1,07% |
| **GPU CUDA (`Quadro M620`)** | **1** | **0,02** | **3/3** | **3** | **3,02 ± 0,65** | **6,52×** | **5,70×** | **85,00 ± 0,77%** |
| **Đa luồng `Sync-Ring` (4 worker)** | **4** | **0,02** | **3/3** | **3** | **2,32 ± 0,90** | **8,48×** | **8,96×** | **85,37 ± 1,10%** |
| Đa luồng `Local-Avg` (4 worker) | 4 | 0,02 | 2/3 | 6 | 4,40 (2 seed) | – | 8,87× | 83,77 ± 0,14% |
| Bất đồng bộ `Parameter Server` (4 worker) | 4 | 0,10 | 0/3 | – | – | – | 8,15× | 81,97 ± 1,98% |

> **Kết luận Bài toán 2**:
> - GPU **không tự động nhanh hơn CPU ở mọi kích thước**: khi bài toán quá nhỏ (`batch = 1` hoặc `Matvec n ≤ 512`), thời gian chạy CPU chỉ `1,4–39 µs` trong khi chi phí khởi chạy kernel và đồng bộ WDDM trên GPU mất `~42–76 µs`, làm GPU chậm hơn CPU (`0,03×–0,56×`).
> - Khi kích thước batch và ma trận đủ lớn để che độ trễ khởi tạo và khai thác tái sử dụng dữ liệu trong shared memory (`Tiled GEMM 2048×2048` hoặc huấn luyện toàn mạng), GPU Quadro M620 nhanh hơn CPU 1 nhân **`8,56×`** ở GEMM và **`6,52×`** về thời gian đạt ngưỡng accuracy `84,0%`.

---

### 3.3 Thực nghiệm 3 & 4 (`EXP3`, `EXP4`): Phân tích Communication Cost, Strong Scaling và Gradient Staleness

Báo cáo gốc: [results/exp3_comm_allreduce.md](results/exp3_comm_allreduce.md) và [results/exp4_dist_sgd_staleness.md](results/exp4_dist_sgd_staleness.md).

#### a) So sánh 4 thuật toán Allreduce & mô hình Hockney $\alpha$–$\beta$ (`EXP3`)

Khớp từ số đo thực tế: độ trễ khởi tạo $\alpha \approx \mathbf{15,88\text{ }\mu\text{s}}$, băng thông hiệu dụng $B = 1/\beta \approx \mathbf{0,90\text{ GB/s}}$.

| Kích thước thông điệp | $p=4$: `ring` (µs) | $p=4$: `tree` (µs) | $p=4$: `rd` (µs) | $p=4$: `naive` (µs) | $p=8$: `ring` (µs) | $p=8$: `tree` (µs) | $p=8$: `rd` (µs) | $p=8$: `naive` (µs) | Thuật toán thắng ($p=4 / p=8$) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `4 KB` | 76,0 | 45,7 | **10,9** | 19,8 | 197,5 | 66,9 | **31,5** | 42,5 | **`rd` / `rd`** |
| `16 KB` | 72,7 | 84,5 | **34,7** | 62,6 | 198,1 | 102,1 | **53,6** | 87,1 | **`rd` / `rd`** |
| `64 KB` | **122,1** | 250,2 | 147,1 | 344,4 | **194,4** | 232,5 | 196,9 | 425,8 | **`ring` / `ring`** |
| **`212 KB` (CNN)** | **189,7** | 596,2 | 537,1 | 643,7 | **248,4** | 532,0 | 672,3 | 807,2 | **`ring` / `ring`** |
| `1 MB` | **2460,0** | 2831,5 | 3773,0 | 2573,1 | **2851,7** | 5801,7 | 11416,9 | 7023,7 | **`ring` / `ring`** |
| `16 MB` | **41392,4** | 53736,9 | 71464,8 | 49372,1 | **91117,1** | 111532,6 | 188505,2 | 117002,2 | **`ring` / `ring`** |

- Ở thông điệp nhỏ (`4–16 KB`), độ trễ khởi tạo chiếm ưu thế nên `rd` ($\log_2 p$ bước) nhanh nhất.
- Từ `64 KB` đến `16 MB` (bao gồm vector đạo hàm `212 KB` của CNN), `ring` nhanh nhất ở mọi cấu hình $p = 4, 8$ nhờ chia nhỏ gói tin và chỉ truyền $2\frac{p-1}{p} M$ byte mỗi worker.

#### b) Quét Batch Size (`EXP3` mục 2) và Strong Scaling (`EXP4` mục 1)

- **Khi quét Global Batch Size từ `16` lên `256` ($p = 4$ worker, 2 epoch):** Số lần gọi Allreduce mỗi epoch ($\lceil N/B \rceil$) giảm từ `438` xuống `28` lần, giúp **tỉ trọng truyền thông/chờ đồng bộ giảm từ `15,3%` (`batch = 16`) xuống `9,0%–9,2%` (`batch = 64–256`)**.
- **Strong Scaling trên `7.000` ảnh Fashion-MNIST (`batch = 64`, 3 epoch, 3 lần lặp — `EXP4`):**

| Phương pháp | Worker ($p$) | Thời gian trung vị (s) | Khoảng (min–max) | Tăng tốc (vs trực tiếp) | Tăng tốc (vs GEMM) | Hiệu suất (vs GEMM) | Tỉ trọng chờ/truyền |
|---|--:|--:|--:|--:|--:|--:|--:|
| CPU 1 nhân (Conv trực tiếp) | 1 | 19,090 | 18,999–19,777 | 1,00× | 0,29× | – | – |
| CPU 1 nhân (`im2col + GEMM`) | 1 | 5,502 | 5,464–5,658 | **3,47×** | 1,00× | – | – |
| Đa luồng `Sync-Ring` ($p=1$) | 1 | 5,648 | 5,470–6,009 | 3,38× | 0,97× | 97,4% | 0,0% |
| Đa luồng `Sync-Ring` ($p=2$) | 2 | 3,400 | 3,271–3,452 | **5,61×** | **1,62×** | **80,9%** | **2,8%** |
| Đa luồng `Sync-Ring` ($p=4$) | 4 | 2,146 | 2,137–2,243 | **8,89×** | **2,56×** | **64,1%** | **10,2%** |
| **GPU CUDA (`Quadro M620`)** | **1** | **3,405** | **3,400–3,446** | **5,61×** | **1,62×** | – | – |

#### c) Ảnh hưởng của độ trễ đạo hàm (Gradient Staleness $d$) theo bước hiệu dụng $\eta$ trên GPU (`EXP4` mục 2)

Đo sau `4` epoch trên GPU Quadro M620 (`2` seed):

| Momentum | Bước hiệu dụng $\eta$ | lr thực | $d = 0$ | $d = 3$ | $d = 7$ | $d = 15$ |
|--:|--:|--:|--:|--:|--:|--:|
| 0,0 | 0,02 | 0,020 | 77,9 ± 0,0% | 78,2 ± 0,1% | **74,8 ± 2,5%** | **41,0 ± 10,3%** |
| 0,0 | 0,05 | 0,050 | 80,0 ± 1,7% | 77,3 ± 1,7% | 41,6 ± 15,5% | 25,1 ± 21,6% |
| 0,0 | 0,10 | 0,100 | 82,4 ± 1,2% | 69,0 ± 14,1% | **9,8 ± 0,0%** *(phân kỳ)* | **9,8 ± 0,0%** *(phân kỳ)* |
| 0,0 | 0,20 | 0,200 | 81,4 ± 0,2% | **8,9 ± 1,2%** *(phân kỳ)* | **8,9 ± 1,2%** *(phân kỳ)* | **9,8 ± 0,0%** *(phân kỳ)* |
| 0,9 | 0,02 | 0,002 | 81,5 ± 0,8% | 77,1 ± 1,6% | **70,8 ± 4,2%** | 20,6 ± 14,6% |
| 0,9 | 0,05 | 0,005 | 83,9 ± 0,8% | 80,0 ± 0,6% | 19,4 ± 15,3% | **10,8 ± 1,4%** *(phân kỳ)* |
| 0,9 | 0,10 | 0,010 | 85,2 ± 1,6% | 65,4 ± 9,3% | 22,0 ± 17,3% | **10,1 ± 0,4%** *(phân kỳ)* |
| 0,9 | 0,20 | 0,020 | 85,7 ± 0,9% | **8,9 ± 1,2%** *(phân kỳ)* | **9,9 ± 0,1%** *(phân kỳ)* | **9,8 ± 0,0%** *(phân kỳ)* |

> **Kết luận Bài toán 3**:
> - Chi phí truyền thông tăng theo số worker $p$ nhưng giảm tỉ trọng khi tăng kích thước `batch`. Với vector đạo hàm `212 KB` của CNN, `ring` Allreduce luôn đạt thời gian thấp nhất ở $p \ge 4$.
> - Trong các phương pháp giảm truyền thông hoặc bỏ đồng bộ: `Local-Avg` (trung bình mỗi epoch) và `Parameter Server` bất đồng bộ tuy chạy mỗi epoch nhanh nhưng hội tụ chậm hơn `Sync-Ring`; khi độ trễ $d$ lớn ($d \times \eta \ge 0,60$), đạo hàm cũ làm mô hình phân kỳ hoàn toàn về mức đoán ngẫu nhiên (`~9,8%`), buộc phải giảm learning rate xuống $\eta = 0,02$.
