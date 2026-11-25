# HPC 20261B, Nhóm 10: Parallel and Distributed Deep Learning (MPI + CUDA)

Đề tài của Nhóm 10, môn Tính toán hiệu năng cao (IT5408, 20261B). Nhóm tái lập và kiểm chứng bằng số đo các nhận định trong bài
V. Hegde, S. Usmani, *Parallel and Distributed Deep Learning* (Stanford, 2016; [hgpu.org/?p=16046](https://hgpu.org/?p=16046)).
Bài báo không nằm trong repo (bản quyền của tác giả).

Mỗi thành viên làm một phần của bài báo trong **một thư mục riêng**. Mỗi phần có mã C tuần tự, mã song song (MPI và/hoặc CUDA) tự
cài đặt, kiểm tra tính đúng so với bản tuần tự và kết quả đo kèm log gốc. Tài liệu này là cửa vào chung: bài toán, phần nào của
bài báo do ai làm, cách biên dịch và chạy từng phần, và kết quả chính.

Mục lục: [1. Bài toán](#1-bài-toán) · [2. Phân công](#2-phân-công-theo-bài-báo) · [3. Biên dịch và chạy](#3-biên-dịch-và-chạy) ·
[4. Kết quả chính](#4-kết-quả-chính) · [5. Tính đúng](#5-kiểm-tra-tính-đúng) · [6. Giới hạn](#6-giới-hạn-chung) ·
[7. Cấu trúc repo](#7-cấu-trúc-repo) · [8. Đóng góp](#8-cách-đóng-góp)

## 1. Bài toán

Huấn luyện mạng nơ-ron bằng SGD theo mini-batch:

```
L_i = -f_{y_i} + log(Σ_j exp(f_j))                 (loss softmax của ảnh i)
w  <- w - α · (1/m) Σ_{i ∈ M} ∇_w L_i              (cập nhật trên mini-batch M, |M| = m)
```

Câu hỏi của bài báo và của nhóm: làm sao chia việc này ra nhiều nhân CPU, nhiều GPU hoặc nhiều máy, và phải trả giá gì.

- **Tính toán (§3 của bài).** Conv được đưa về nhân ma trận bằng im2col, lớp kết nối đầy đủ là nhân ma trận–vector. GPU nhanh hơn
  CPU bao nhiêu, và tỉ lệ đó thay đổi theo batch và kích thước ma trận ra sao?
- **Song song dữ liệu (§5).** Mỗi tiến trình hoặc GPU giữ một bản sao mô hình và một phần dữ liệu, gradient hoặc tham số được gộp lại
  bằng allreduce. Bài phân tích chi phí truyền thông của cách gom về một máy rồi phát lại là `O(L·k) + O(k·p/B)` (k máy, p tham số,
  độ trễ L, băng thông B). Nhóm đo chi phí này với các thuật toán allreduce tự viết (tree, ring, recursive doubling).
- **Đồng bộ và bất đồng bộ (§5.1–5.2).** Sync SGD, Parallel SGD (trung bình tham số, có ShuffleSGD), ADMM.SGD, Downpour SGD
  (parameter server, gradient trễ).
- **Song song mô hình (§2).** Bài nói chỉ nên chia mô hình khi mô hình không vừa một máy, không phải để tăng tốc. Nhóm kiểm chứng
  bằng pipeline parallelism và tensor parallelism trên nhiều GPU.

Theo yêu cầu của đề, **thuật toán song song được tự thiết kế và cài đặt**: GEMM, im2col, lan truyền ngược, các thuật toán allreduce,
pipeline/tensor parallel và các phương pháp tối ưu phân tán đều được viết lại. Các ngoại lệ được ghi rõ ở [mục 6](#6-giới-hạn-chung).

## 2. Phân công theo bài báo

| Mục của bài báo | Nhận định của bài | Ai kiểm chứng | Mã song song |
|---|---|---|---|
| §2 Model parallelism | Chỉ dùng khi mô hình không vừa một máy | [`vinhvn/`](vinhvn/) | CUDA, nhiều GPU |
| §3 CPU so với GPU: conv | Thời gian tuyến tính theo batch, GPU nhanh hơn khoảng 40 lần | [`hieunv/`](hieunv/), [`tuannn/`](tuannn/) (E2), [`thinhnd/`](thinhnd/) (T6) | CUDA, OpenMP |
| §3 CPU so với GPU: FC | Ma trận–vector O(n²), speedup tăng theo n | [`hieunv/`](hieunv/), [`thinhnd/`](thinhnd/) (T6) | CUDA |
| §5.1 Sync SGD, strong scaling | Đồng bộ cho cùng kết quả như tuần tự | [`tuannn/`](tuannn/) (E4), [`thinhnd/`](thinhnd/) (T7) | MPI, CUDA |
| §5.1.1 Parallel SGD, ShuffleSGD | Cần trộn dữ liệu vì hàm loss không lồi | [`tuannn/`](tuannn/) (E5) | MPI |
| §5.1.2 ADMM.SGD | Đồng bộ qua parameter server | [`tuannn/`](tuannn/) (E5) | MPI |
| §5.2 Downpour SGD, bất đồng bộ | Bỏ chờ đồng bộ để tăng tốc | [`tuannn/`](tuannn/) (E5), [`thinhnd/`](thinhnd/) (T4, T8) | MPI |
| §5 Chi phí truyền thông | `O(L·k) + O(k·p/B)` cho gom-rồi-phát | [`ltnaphixay/`](ltnaphixay/), [`tuannn/`](tuannn/) (E3) | CUDA (đa GPU), MPI |

### Thành viên

| Thư mục | Thành viên | Tài khoản | Phần việc |
|---|---|---|---|
| [`vinhvn/`](vinhvn/) | (điền) | (điền) | Model parallelism: pipeline (GPipe) và tensor parallelism (kiểu Megatron) trên 4× H100, GEMM và all-reduce tự viết |
| [`hieunv/`](hieunv/) | (điền) | (điền) | Đo lại §3 bằng C tuần tự và CUDA tự viết trên Kaggle T4: conv kiểu Caffe theo batch, matvec theo n, GEMM, băng thông bộ nhớ; bản PyTorch giữ làm mốc tham khảo |
| [`tuannn/`](tuannn/) | Tuan Nguyen | [tuannt2906](https://github.com/tuannt2906) | CNN tự cài bằng C/MPI/OpenMP/CUDA: Sync SGD, Parallel SGD, ADMM, Downpour, 6 thuật toán allreduce |
| [`thinhnd/`](thinhnd/) | Nguyen Duc Thinh | [ducthinhh1002](https://github.com/ducthinhh1002) | CNN tự viết bằng C/MPI/CUDA trên RTX 4060: thời gian tới cùng accuracy, ảnh hưởng của gradient trễ |
| [`ltnaphixay/`](ltnaphixay/) | LATTANA APHIXAY | [infinityrbx](https://github.com/infinityrbx) | SGD song song dữ liệu trên CUDA với allreduce tự viết: 2× T4 (PCIe), 4× A100 (NVLink so với qua host) |

## 3. Biên dịch và chạy

Mỗi phần tự đứng được: vào thư mục của phần đó rồi chạy các lệnh dưới đây. Tham số đầy đủ, cách tải dữ liệu và cách tái lập toàn bộ
thí nghiệm nằm trong `README.md` của từng thư mục.

### Môi trường

| Phần | Bản tuần tự cần | Bản song song cần | Đã chạy trên |
|---|---|---|---|
| `vinhvn` | gcc/clang, make | CUDA Toolkit (nvcc), ≥ 2 GPU NVIDIA cho chế độ nhiều GPU | 4× H100 80GB, CUDA 12.4 |
| `tuannn` | gcc, make | CMake, MPI (OpenMPI/MPICH), OpenMP; CUDA tuỳ chọn | i5-12600K + RTX 2060, WSL2, OpenMPI 4.1.6, CUDA 12.0 |
| `thinhnd` | gcc | MPI (MS-MPI), CUDA Toolkit (NVRTC) | Windows 11, i7-13620H + RTX 4060, CUDA 12.8 |
| `ltnaphixay` | gcc, make | CUDA Toolkit, ≥ 2 GPU NVIDIA | Kaggle 2× T4, RunPod 4× A100 SXM |
| `hieunv` | gcc, make | CUDA Toolkit (nvcc), 1 GPU NVIDIA | Kaggle T4 + Xeon 2,0 GHz, gcc 13.3, nvcc 13.0 |

### `vinhvn/`: model parallelism (C và CUDA)

```bash
cd vinhvn
make seq                                   # bản tuần tự  -> bin/mlp_seq
make cuda ARCH=sm_90                       # bản song song -> bin/mlp_cuda   (A100: sm_80, T4: sm_75)

bin/mlp_seq --H 512 --L 4 --B 128 --steps 2                                  # tuần tự, CPU một luồng
bin/mlp_cuda --mode pipeline --gpus 4 --micro 8 --H 4096 --L 8 --B 2048      # pipeline 4 stage, 8 micro-batch
bin/mlp_cuda --mode tensor   --gpus 4          --H 4096 --L 8 --B 2048      # tensor parallelism trên 4 GPU
scripts/check_correctness.sh                                                 # so bản song song với bản tuần tự (cần 4 GPU)
```

### `tuannn/`: CNN với Sync SGD, Parallel SGD, ADMM, Downpour (C, MPI, OpenMP, CUDA)

```bash
# Tuần tự (C thuần, không MPI/OpenMP/CUDA)
cd tuannn/code/sequential
make                                       # -> ./seq_train
make run-synthetic                         # chạy thử không cần tải dữ liệu
./seq_train --epochs 3 --output seq.json   # LeNet trên MNIST (cần ../data/mnist-*.pdlds, xem tuannn/README.md)

# Song song
cd tuannn/code
scripts/build.sh                           # CMake Release, tự bật CUDA nếu có nvcc
ctest --test-dir build --output-on-failure # 13 bài kiểm tra tính đúng
mpirun -np 4 build/bin/pdl_train --strategy sync --global-batch 256 --lr 0.05 --allreduce ring
mpirun -np 4 build/bin/pdl_train --strategy psgd --avg-period 10 --reshuffle 1
mpirun -np 4 build/bin/pdl_train --strategy admm --admm-local-steps 20
mpirun -np 5 build/bin/pdl_train --strategy downpour --push-every 2 --fetch-every 2
build/bin/pdl_train --strategy sequential --device cuda
mpirun -np 4 build/bin/pdl_bench_comm --algos all              # 6 thuật toán allreduce
```

### `thinhnd/`: CNN trên Fashion-MNIST (C, MPI, CUDA)

Script build hiện viết cho Windows (WinLibs GCC, MS-MPI, CUDA Toolkit); chưa thử trên Linux. Dữ liệu: hai file `t10k-*` của
Fashion-MNIST giải nén vào `thinhnd/cnn/data/` (đường dẫn tải trong [`thinhnd/README.md`](thinhnd/README.md)).

```powershell
cd thinhnd/cnn
python build.py                                   # biên dịch mọi chương trình vào bin/
.\bin\gradient_check.exe                          # kiểm tra lan truyền ngược bằng sai phân hữu hạn
.\bin\train_seq.exe --epochs 3                    # tuần tự
mpiexec -n 4 .\bin\train_mpi.exe --mode sync  --epochs 3
mpiexec -n 4 .\bin\train_mpi.exe --mode avg   --epochs 3
mpiexec -n 4 .\bin\train_mpi.exe --mode async --epochs 3
.\bin\train_cuda.exe --check                      # so CUDA với C
.\bin\train_cuda.exe --epochs 3
```

### `ltnaphixay/`: SGD song song dữ liệu với allreduce tự viết (C và CUDA)

```bash
cd ltnaphixay
make data                                          # tải MNIST vào ./data
make seq                                           # bản tuần tự (chỉ cần gcc) -> bin/mlp_seq
make ARCH=sm_75                                    # bản song song (sm_75 = T4, sm_80 = A100)
make simcheck                                      # kiểm tra tree/ring/rd bằng CUDA giả lập, không cần GPU

./bin/mlp_seq                                      # tuần tự (CPU)
./bin/train_dp --gpus 1                            # 1 GPU
./bin/train_dp --gpus 2 --method ring [--overlap]  # song song dữ liệu, method: host_ref | tree | ring | rd
./bin/bench_allreduce --method ring --gpus 2 [--check-only]
```

### `hieunv/`: đo lại §3, CPU so với GPU (C và CUDA)

Mã nằm trong [`hieunv/c_cuda/`](hieunv/c_cuda/): `bench_seq.c` (C, một luồng, không BLAS/OpenMP) và `bench_cuda.cu` (kernel CUDA tự
viết, không cuBLAS/cuDNN). Mỗi chương trình kiểm tra mọi kernel so với float64 trước khi đo, sai thì thoát với mã 1.

```bash
cd hieunv/c_cuda
make seq                                   # bản tuần tự  -> ./bench_seq
make cuda ARCH=sm_75                       # bản song song -> ./bench_cuda   (T4: sm_75)
make check                                 # chạy phần kiểm tra tính đúng

./bench_seq  --check-only
./bench_seq  --machine laptop --outdir ../results/laptop/seq_quick --quick
./bench_cuda --check-only
./bench_cuda --machine kaggle --outdir ../results/kaggle/c_cuda1 --order-seed 1 --repeat 30
```

Trên Kaggle: import [`hieunv/kaggle/hpc_c_cuda_kaggle.ipynb`](hieunv/kaggle/hpc_c_cuda_kaggle.ipynb), chọn GPU T4, Run All
(biên dịch, in `ldd` để thấy không liên kết thư viện, kiểm tra, rồi 3 lần đo). Bản PyTorch đo trước đó nằm trong `hieunv/code/`,
chỉ để tham khảo.

## 4. Kết quả chính

Mọi số dưới đây lấy từ README và log gốc của từng phần; mỗi dòng có liên kết tới nơi chứa dữ liệu. Phần cứng khác nhau giữa các
phần, nên chỉ so tỉ lệ (tăng tốc, xu hướng), không so thời gian tuyệt đối giữa các thư mục.

### 4.1 §3: CPU so với GPU

| Phép đo | Kết quả | Phần cứng | Nguồn |
|---|---|---|---|
| Conv 96×11×11, stride 4, ảnh 224×224: thời gian theo batch | Tuyến tính: R² ≥ 0,999 (CPU) và ≥ 0,9998 (GPU) với C/CUDA tự viết | T4 + Xeon (Kaggle) | [hieunv ANALYSIS §8](hieunv/ANALYSIS.md), [tuannn E2](tuannn/experiments/results/SUMMARY_dot2.md) |
| Conv: GPU / CPU 1 luồng (bài báo: khoảng 40×) | **23,3×** (22,3–23,3; kiểu Caffe); speedup theo batch 18,6× (batch 1) → 23,1× (batch 256) | T4 + Xeon, C/CUDA tự viết | [hieunv](hieunv/README.md) |
| | **54,5×** ở batch 256 (13,2× so với 12 luồng; 11× nếu tính cả chép dữ liệu) | RTX 2060, i5-12600K | [tuannn E2](tuannn/experiments/results/SUMMARY_dot2.md) |
| | **40–56×** ở batch 64–1024; 2,95× ở batch 1 | RTX 4060, i7-13620H | [thinhnd T6](thinhnd/results/t6_layers.md) |
| | Tham khảo PyTorch: 27,8× (kiểu Caffe), khoảng 14× nếu dùng `nn.Conv2d` | T4 + Xeon | [hieunv](hieunv/README.md) |
| FC (ma trận–vector) theo n | O(n²); speedup tăng đều theo n: **1,6× (n = 256) → 23,1× (n = 16384)** với C/CUDA tự viết, chạm khoảng 93–100% băng thông bộ nhớ đo độc lập | T4 + Xeon | [hieunv ANALYSIS §8](hieunv/ANALYSIS.md) |
| | Ở n = 256 GPU **thua** CPU khi chi phí khởi chạy lớn: 0,51× (thinhnd), 0,43× (PyTorch); thắng từ n ≥ 512–1024 | RTX 4060; T4 | [thinhnd T6](thinhnd/results/t6_layers.md), [hieunv](hieunv/README.md) |
| Cả mô hình: C tuần tự so với 1 GPU | 33× (CNN), 41× (MLP), 371× (MLP lớn, H = 512) | RTX 4060; T4; H100 | [thinhnd](thinhnd/README.md), [ltnaphixay](ltnaphixay/README.md), [vinhvn](vinhvn/README.md) |

**Kết luận:** xu hướng của bài đúng (tuyến tính theo batch, speedup tăng theo batch và theo n). Con số "40×" không phải hằng số: nó
thay đổi từ dưới 1× đến hơn 50× tuỳ loại lớp, kích thước, phần cứng và cách cài đặt.

### 4.2 §5: song song dữ liệu trên CPU (MPI)

| Phép đo | Kết quả | Nguồn |
|---|---|---|
| Strong scaling Sync SGD, LeNet/MNIST, batch 256 | 1,79 / 2,91 / 3,12 / 3,16 / 3,45× ở k = 2 / 4 / 6 / 8 / 10; bão hoà vì phần tính toán dùng chung tài nguyên, không phải vì truyền thông | [tuannn E4](tuannn/experiments/results/SUMMARY_dot2.md) |
| Strong scaling Sync SGD, CNN/Fashion-MNIST | 1,88 / 3,27 / 3,57× ở p = 2 / 4 / 6; phần chờ/truyền tăng từ 2,2% lên 27,8% | [thinhnd T7](thinhnd/results/t7_scaling.md) |
| Parallel SGD, dữ liệu không IID | Accuracy rơi còn 77,7% (k = 4, 3 epoch); ShuffleSGD kéo lên 97,9%; trung bình mỗi 10 bước cũng gỡ được | [tuannn E5](tuannn/experiments/results/SUMMARY_dot1.md) |
| ADMM.SGD | 98,1% (IID), 88,5% (không IID) ở k = 4; xấu đi khi k tăng | [tuannn E5 đợt 1](tuannn/experiments/results/SUMMARY_dot1.md), [đợt 2](tuannn/experiments/results/SUMMARY_dot2.md) |
| Đồng bộ so với bất đồng bộ, thời gian tới cùng accuracy 85% | Đồng bộ nhanh hơn bất đồng bộ 1,46× (không momentum) và 3,31× (momentum 0,9), 5/5 seed | [thinhnd T8](thinhnd/results/t8_fair_comparison.md) |
| Gradient trễ d bước, learning rate η | Tích d·η càng lớn càng dễ không học được (chỉ 0–1/10 seed học được ở d·η ≥ 0,6); trễ lớn vẫn học được nếu bước nhỏ và đủ epoch | [thinhnd T4](thinhnd/results/t4c_control.md) |

### 4.3 §5: chi phí truyền thông trên nhiều GPU

| Phép đo | Kết quả | Nguồn |
|---|---|---|
| 2× T4, MLP 1,63 MB gradient, batch 256 | 2 GPU **chậm hơn** 1 GPU (0,87–0,89×): khoảng 50% mỗi bước là truyền thông không che được | [ltnaphixay §4](ltnaphixay/README.md) |
| Quét batch trên 2× T4 | Hoà vốn giữa batch 256 và 1024; đạt tới 1,60× ở batch 4096 | [ltnaphixay §6](ltnaphixay/README.md) |
| Allreduce theo kích thước, k = 2 | Nhỏ (≤ 8 KB): qua host nhanh nhất; vừa (16 KB–2 MB): recursive doubling; lớn (≥ 4 MB): ring | [ltnaphixay §5](ltnaphixay/README.md) |
| 4× A100, NVLink so với qua host, 256 MB | Ring 157 GB/s (NVLink) so với 5,8 GB/s (qua host); tree giảm từ 93 xuống 51 GB/s khi k = 2 → 4, đúng phân tích gom-rồi-phát của bài | [ltnaphixay §7](ltnaphixay/README.md) |
| Số hạng độ trễ O(L·k) | Thông điệp nhỏ: ring cần 2(k−1) = 6 bước ở k = 4 (khoảng 0,44 ms), rd chỉ log₂k = 2 bước (khoảng 0,15 ms) | [ltnaphixay §7](ltnaphixay/README.md) |
| 6 thuật toán allreduce trên CPU, 16 MiB, k = 2–10 | Ring nhanh nhất từ k = 4 (33 ms so với 47 ms của `MPI_Allreduce` ở k = 10); recursive doubling sụp ở k ≥ 8 | [tuannn E3](tuannn/experiments/results/SUMMARY_dot2.md) |

### 4.4 §2: model parallelism trên 4× H100

| Cấu hình | Kết quả | Nguồn |
|---|---|---|
| Chia mô hình, không chia micro-batch (M = 1), 2 và 4 GPU | **1,00×**: không tăng tốc, đúng nhận định của bài | [vinhvn §5.2](vinhvn/README.md) |
| Pipeline, 4 GPU, 8 micro-batch | 2,64× (trần lý thuyết 2,91×); M = 16 lại chậm hơn M = 8 | [vinhvn §5.2](vinhvn/README.md) |
| Tensor parallelism, 2 và 4 GPU | 1,94× và 3,73× (hiệu suất 93–97%) | [vinhvn §5.2](vinhvn/README.md) |
| Mô hình 12,9 tỉ tham số (≈ 103 GB) | 1 GPU: hết bộ nhớ; 2 và 4 GPU: chạy được (52,6 và 26,6 GB mỗi GPU) | [vinhvn §5.3](vinhvn/README.md) |

**Kết luận:** chia mô hình thuần tuý chỉ giúp mô hình chạy được khi không vừa một GPU. Muốn tăng tốc phải thêm micro-batch
(pipeline) hoặc chia từng phép nhân ma trận (tensor parallelism).

## 5. Kiểm tra tính đúng

Mỗi phần so bản song song với bản tuần tự trước khi đo hiệu năng:

| Phần | Kiểm tra | Kết quả |
|---|---|---|
| `vinhvn` | 12 cấu hình pipeline/tensor so với C tuần tự; C tuần tự so với PyTorch autograd (fp64) | Sai số tương đối của loss ≤ 8·10⁻⁹, của chuẩn gradient ≤ 1,4·10⁻⁸ ([`check.log`](vinhvn/results/check.log)) |
| `tuannn` | CTest, gradient check, CUDA so với CPU, sync SGD so với tuần tự | 105 kiểm tra, 0 lỗi; sync k = 2/4 lệch tuần tự 5,7·10⁻⁸ / 2,9·10⁻⁸ |
| `thinhnd` | Sai phân hữu hạn; CUDA so với C | Lệch tối đa 8,9·10⁻⁷; CUDA lệch C 1,8·10⁻⁷ |
| `hieunv` | Mọi kernel C và CUDA so với tham chiếu float64 trên CPU (cạnh ô, K lớn hơn một khối, `im2col` so từng bit, đuôi của phép rút gọn) trước khi đo | Kiểm tra chạy trước mỗi lần đo và dừng chương trình (mã 1) nếu sai; cả 3 lần đo chính thức đều chạy xong |
| `ltnaphixay` | `simcheck` 5.670 lần chạy với thứ tự stream ngẫu nhiên; so 2 GPU với 1 GPU qua 10 epoch | Các bản sao tham số giống nhau từng bit; loss lệch tối đa 2,3·10⁻⁵ |

## 6. Giới hạn chung

- **Một máy cho mỗi lần đo.** MPI chạy trên một node qua shared memory; đa GPU nằm trong một máy. Chưa đo qua mạng giữa nhiều máy.
- **Mô hình và dữ liệu nhỏ** (LeNet, CNN nhỏ, MLP) ở các phần song song dữ liệu: truyền thông tương đối rẻ, kết luận có thể khác với mô hình lớn.
- **Số lần lặp ít** (1–5 lần mỗi cấu hình tuỳ phần); chênh lệch vài phần trăm chưa đủ để kết luận.
- **Phần dùng thư viện:**
  - `hieunv/code/` (PyTorch) chỉ là mốc tham khảo; số chính thức của `hieunv/` lấy từ bản C/CUDA tự viết trong `hieunv/c_cuda/`.
  - `tuannn/` có thêm các lựa chọn cuBLAS (`--impls cublas`) và `MPI_Allreduce` (`--allreduce mpi`) để so sánh với bản tự viết. Phép
    đo strong scaling E4 dùng `--allreduce mpi`. Trung bình tham số khi đánh giá Parallel SGD (`strategy_psgd.c`) cũng gọi thẳng
    `MPI_Allreduce`. Các thuật toán allreduce tự viết (ring, tree, recursive doubling, naive) được đo riêng ở E3.
  - `reference_pytorch/` trong `vinhvn/` chỉ dùng để đối chứng, không thuộc mã nộp.
- **Chưa kiểm chứng:** nhận định của §5.2 rằng Downpour SGD không còn tăng tốc khi số worker vượt `T/τ`.

Giới hạn riêng của từng phần nằm ở mục "Giới hạn" trong README của thư mục đó.

## 7. Cấu trúc repo

```
.
├── README.md        tài liệu này
├── vinhvn/          §2 model parallelism: src/mlp_seq.c, src/mlp_cuda.cu, results/
├── tuannn/          §3 + §5: code/ (sequential/, src/, apps/), experiments/ (PLAN.md, results/), report/, slides/
├── thinhnd/         §3 + §5: cnn/src/{seq,mpi,cuda}/, results/ (t1–t8)
├── ltnaphixay/      §5 truyền thông đa GPU: src/{seq,cuda,allreduce}/, bench/, tests/, runs/, docs/report/
└── hieunv/          §3 CPU so với GPU: c_cuda/ (bench_seq.c, bench_cuda.cu), code/ (PyTorch tham khảo), kaggle/, results/, ANALYSIS.md
```

Mỗi thư mục có `README.md` riêng gồm mục tiêu, cách chạy, kết quả và giới hạn. Log và dữ liệu gốc (CSV, JSON, log) nằm cạnh bản tổng hợp.

## 8. Cách đóng góp

**Chỉ sửa trong thư mục của mình.** Muốn sửa thư mục của người khác hoặc `README.md` gốc, mở Merge Request và nhờ chủ thư mục xem trước.

```bash
git clone https://gitlab.nqtuan0192.me/vu/hpc_20261b_group10_paralleldistributeddeeplearning_mpi-cuda.git
cd hpc_20261b_group10_paralleldistributeddeeplearning_mpi-cuda

git checkout main && git pull
git checkout -b <thư-mục-của-bạn>/<việc-đang-làm>
git add <thư-mục-của-bạn>
git commit -m "Mô tả ngắn: thay đổi gì và vì sao"
git pull --rebase origin main
git push -u origin <thư-mục-của-bạn>/<việc-đang-làm>
```

Commit thường xuyên trong lúc làm, mỗi commit một việc. Không đưa lên repo: thông tin đăng nhập (kubeconfig, khoá, token, `.env`; nếu
lỡ commit thì thu hồi ngay), tệp lớn hoặc dataset (ghi cách tải trong README), tệp sinh ra (`bin/`, `build/`, `__pycache__/`,
`.venv/`), bài báo có bản quyền và thông tin cá nhân không cần thiết. Gặp xung đột khi `git pull --rebase` thì dừng lại hỏi nhóm, không `--force`.
