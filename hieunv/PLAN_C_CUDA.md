# Kế hoạch đo lại bằng C tuần tự và CUDA tự viết

Viết ngày 25/11/2026, **trước** khi biên dịch hay chạy bất kỳ lần đo nào của bản C/CUDA. Laptop chưa có trình
biên dịch C, và bản CUDA chưa từng được biên dịch. Cột "Kết quả" ở mục 6 sẽ điền sau khi chạy; các cột dự đoán
giữ nguyên.

## 1. Vì sao làm lại

Bài tập lớn không cho dùng thư viện song song có sẵn. Bản đã đo trong `code/` dùng PyTorch. Trên CPU, phép tính
chạy bằng MKL/oneDNN; trên GPU, bằng cuBLAS/cuDNN. Các con số ở `ANALYSIS.md` vì vậy là hiệu năng của thư viện,
không phải của code tự viết. Bản PyTorch được **giữ nguyên, không xóa**: nó cho biết thư viện tối ưu đạt được
bao nhiêu trên cùng phần cứng. Phần mới nằm ở [`c_cuda/`](c_cuda/):

- `bench_seq.c`: C tuần tự, 1 luồng, không BLAS, không OpenMP.
- `bench_cuda.cu`: CUDA, mọi kernel tự viết, không cuBLAS/cuDNN/Thrust/CUB.

**Ranh giới "không dùng thư viện".** Phần được tính giờ chỉ dùng thư viện chuẩn C (libc, libm) và CUDA runtime
(cấp phát bộ nhớ, chạy kernel, đồng bộ). Notebook in `ldd` của hai chương trình để kiểm chứng. Các công cụ nằm
ngoài phần tính giờ vẫn dùng Python: sinh notebook, ghi xung nhịp GPU qua NVML trong một tiến trình riêng, vẽ
hình và phân tích bằng pandas/matplotlib. **Giả định cần nhóm hoặc giảng viên xác nhận:** những công cụ này được
phép, vì chúng không làm phần tính toán. Trình biên dịch tự vector hóa (SIMD) với `-O3 -march=native`, nhưng
chương trình vẫn chạy trên 1 lõi, 1 luồng. Nếu yêu cầu là tuần tự vô hướng, có thể build thêm một bản với
`-fno-tree-vectorize`.

## 2. Các cài đặt

Chi tiết ở [`c_cuda/README.md`](c_cuda/README.md). Dòng **đậm** là chuỗi chính, dùng để so với bài báo và so CPU
với GPU. Các dòng còn lại là bậc thang "đơn giản → tối ưu" trên từng thiết bị, làm tư liệu cho báo cáo môn HPC.

| Thí nghiệm | CPU (C tuần tự) | GPU (CUDA) |
|---|---|---|
| Conv kiểu Caffe theo batch (Hình 1–2) | **`caffe`**, `direct` | **`caffe`**, `batched`, `direct` |
| Matvec theo n (Hình 3–4) | `matvec_naive`, **`matvec`** | `matvec_naive`, **`matvec`** |
| GEMM n×n (chú thích Hình 3) | `gemm_naive`, `gemm_ikj`, **`gemm`** | `gemm_naive`, `gemm_tiled`, **`gemm`** |
| Băng thông (kiểm chứng giới hạn bộ nhớ) | `sum`, `dot`, `copy`, `triad`, `matvec` | như CPU |
| FC với B vector (giả thuyết về bài báo) | `mm1` … `mm128` | như CPU |
| Tách conv kiểu Caffe | `im2col`, `gemm`, `caffe` | như CPU |

Lớp conv vẫn là lớp của bài báo: ảnh xám 224×224, 96 bộ lọc 11×11, stride 4. Mỗi ảnh được trải thành ma trận
121×2916.

## 3. Kiểm tra đúng đắn trước khi đo

Mỗi chương trình so mọi kernel với kết quả float64 tính trên CPU **trước** khi tính giờ. Chỉ cần một kiểm tra sai
là chương trình dừng với mã lỗi 1, nên notebook dừng theo. Ngưỡng sai số tương đối (so với giá trị tham chiếu lớn
nhất) là 1e-4 cho GEMM, conv và matvec. Với `sum` và `dot`, ngưỡng là 1e-6 so với tổng độ lớn. `copy` và `im2col`
phải khớp từng bit. Lỗi chỉ số cho sai số cỡ 1, nên các ngưỡng này phát hiện được.

Các hình dạng kiểm tra:

- kích thước lẻ nằm ở mép ô (37×53×29, 5×17×3);
- K lớn hơn một khối (300);
- cả hai dạng ô thanh ghi trên GPU (M = 96 dùng ô 48 hàng, M = 64 dùng ô 64 hàng);
- GEMM theo lô với 3 ảnh;
- matvec trên đường vô hướng (n = 37) và đường float4;
- reduction có phần đuôi (N = 1 000 003).

## 4. Cách đo

Giữ quy tắc của bản PyTorch để hai bản so được với nhau:

- 3 lần khởi động. Số lần tính giờ: GPU 30, CPU 10. CPU chậm hơn nhiều, và độ lệch giữa các lần trên CPU nhỏ.
- Thời gian là đồng hồ tường của host. GPU được đồng bộ trước và sau mỗi lần; `issue_ms` là thời gian đến khi các
  lệnh launch trả về.
- Dữ liệu đã nằm sẵn trên thiết bị, không tính thời gian chép host↔GPU.
- Trước mỗi cài đặt, GPU được giữ bận 5 s bằng GEMM tự viết.
- Thứ tự kích thước xáo trộn theo hạt giống bằng số thứ tự lần chạy.
- Xung nhịp GPU được ghi mỗi 20 ms qua NVML. `t_start`, `t_end` dùng giờ Unix để ghép với nhật ký xung nhịp.
- Kaggle: 3 lần chạy trong cùng một phiên, mỗi lần chạy GPU trước rồi CPU. CPU và GPU đo trên **cùng một máy**,
  như bài báo đo trên một máy AWS.

| Lưới | CPU | GPU |
|---|---|---|
| Batch conv `caffe` (và `batched` trên GPU) | 1, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256 | như CPU |
| Batch conv `direct` | 1, 16, 64, 256 (CPU chậm) | như `caffe` |
| n của matvec | 256, 512, 1024, 2048, 4096, 8192, 12288, 16384 | như CPU |
| n của GEMM | `gemm_naive`: 256, 512. Hai bản còn lại: 256–2048 | cả ba bản: 256–4096 |
| n của băng thông (mảng n²) | 1024, 4096, 8192, 16384 | như CPU |
| FC với B vector | n = 2048, 8192; B = 1, 16, 64, 128 | như CPU |
| Tách conv | batch 16, 64, 256 | như CPU |

**Laptop** (tùy chọn, sau khi cài gcc trong WSL Ubuntu): chạy `bench_seq` 3 lần, mỗi lần 1 luồng. Laptop không có
GPU NVIDIA nên không chạy được CUDA. WSL2 là máy ảo, giới hạn 4 vCPU và 6 GB RAM, nên số liệu laptop của bản C
không so trực tiếp với các lần chạy PyTorch trên Windows.

## 5. Phân tích sẽ làm

Phân tích đánh giá như bài báo, và như `ANALYSIS.md` đã làm cho bản PyTorch:

1. **Hình 1–2:**
   - thời gian conv `caffe` theo batch, khớp t(B) = a + b·B trên batch 16–256 (khoảng batch của Hình 1);
   - tỷ lệ độ dốc CPU/GPU, so với khoảng 40× của bài báo;
   - speedup theo từng batch.
2. **Hình 3–4:**
   - thời gian matvec theo n;
   - số mũ k của t ∝ nᵏ, báo cả hai khoảng n ≥ 4096 và n ≥ 1024 như trước;
   - speedup GPU/CPU theo n.
3. **Giới hạn phần cứng:**
   - matvec so với băng thông đọc đo độc lập (`sum`, `dot`) trên từng thiết bị;
   - GEMM tính theo % trần FP32 ở xung nhịp đo được (T4: 5,12 TFLOP/s mỗi GHz);
   - conv tính theo chu kỳ mỗi ảnh (µs × MHz), vì T4 bị giới hạn công suất.
4. **Bậc thang tối ưu:** mỗi bước nhanh hơn bao nhiêu lần, và vì sao (truy cập bộ nhớ, thanh ghi, gộp truy cập).
5. **So với thư viện trên cùng loại máy Kaggle** (C so với MKL, CUDA so với cuBLAS/cuDNN), dùng số trong
   `ANALYSIS.md`. Phiên Kaggle khác nên máy có thể khác: so `lscpu` và tên GPU trước khi so.

Lấy trung vị qua 3 lần chạy và báo cả khoảng min–max của từng lần. Code phân tích (`code/analyze_c_cuda.py`) sẽ
viết sau khi có dữ liệu, theo đúng các khoảng khớp ghi ở trên. Nếu đổi khoảng khớp sau khi thấy số, phải ghi rõ.

## 6. Dự đoán (viết trước khi chạy)

Các mốc so sánh lấy từ `ANALYSIS.md` và `results/analysis/mem_tables.md` (bản PyTorch, Kaggle, 22/11/2026).

| Ký hiệu | Dự đoán | Nếu sai thì nghĩa là | Kết quả |
|---|---|---|---|
| E1 | Kaggle, conv `caffe`: khớp t(B) = a + b·B trên batch 16–256 đạt R² ≥ 0,99 ở cả CPU và GPU, ở mọi lần chạy. Matvec: số mũ k (n ≥ 4096) nằm trong 1,9–2,2 trên cả hai thiết bị | R² < 0,99 hoặc k ngoài khoảng: tính tuyến tính và O(n²) của bài báo phụ thuộc vào cài đặt, không chỉ vào phép tính | Chưa chạy |
| E2 | CPU C, conv kiểu Caffe ở batch 256 (bảng tách bước): `im2col` ≤ 0,40 ms/ảnh (PyTorch 0,80); GEMM tự viết đạt 15–60 GFLOP/s (MKL 88); `caffe` 1,3–5,0 ms/ảnh (PyTorch 1,76) | GEMM < 15 GFLOP/s: gcc không giữ được ô 4×16 trong thanh ghi, cần xem mã máy. `im2col` > 0,40: im2col C cũng chậm như của PyTorch | Chưa chạy |
| E3 | GPU CUDA, conv kiểu Caffe ở batch 256: `im2col` 25–60 nghìn chu kỳ/ảnh (PyTorch 43,6k); GEMM tự viết đạt 10–35% trần FP32 ở xung đo (cuBLAS 26%); `caffe` 95–250 nghìn chu kỳ/ảnh (PyTorch 95,5k) | GEMM > 35%: kernel tự viết hợp với ma trận "gầy" 96×121 hơn cuBLAS. < 10%: kernel có vấn đề hiệu năng (occupancy, bank conflict) | Chưa chạy |
| E4 | Tỷ lệ độ dốc conv CPU (C, 1 luồng) / GPU (CUDA), batch 16–256: 15–60× (bài báo khoảng 40×, PyTorch 27,8×). Speedup tăng từ batch 1 đến 32, sau đó thay đổi không quá 10% từ batch 64 đến 256 | Ngoài khoảng: hệ quả của E2 hoặc E3 sai, ghi rõ là cái nào. Không phẳng: phần chi phí cố định trên GPU lớn hơn mô hình a + b·B | Chưa chạy |
| E5 | CPU C, n ≥ 4096: băng thông đọc tốt nhất (max của `sum`, `dot`) 12–16 GB/s (PyTorch 13,4–14,3, cùng loại máy); `matvec` (32 tổng riêng) ≥ 85% con số đó; `matvec_naive` (1 tổng) ≤ 40% của `matvec` | `matvec` < 85%: matvec C không chạm trần bộ nhớ. `matvec_naive` > 40%: độ trễ phép cộng không phải giới hạn chính | Chưa chạy |
| E6 | GPU CUDA, n ≥ 8192: `sum`/`dot` đạt 240–300 GB/s (PyTorch 253–272); `matvec` (1 warp mỗi hàng) ≥ 85% tốt nhất của `sum`/`dot`; `matvec_naive` (1 luồng mỗi hàng) ≤ 60% của `matvec` | `sum`/`dot` < 240: kernel băng thông tự viết kém hơn của PyTorch, nên tỷ lệ phải dùng số tự đo. `matvec_naive` > 60%: cache bù được việc đọc không gộp | Chưa chạy |
| E7 | Tỷ lệ matvec GPU/CPU ở n = 8192 và 16384 lệch không quá 15% so với tỷ lệ băng thông đọc tốt nhất GPU/CPU đo trong cùng lần chạy (dự kiến khoảng 17–23×) | Lệch > 15%: ít nhất một bên không bị giới hạn bởi bộ nhớ | Chưa chạy |
| E8 | Bậc thang GEMM. CPU: `gemm` ≥ 2× `gemm_ikj` ở n = 1024 và 2048, và `gemm_ikj` ≥ 3× `gemm_naive` ở n = 512. GPU ở n = 4096: `gemm` ≥ 2× `gemm_tiled`, `gemm_tiled` ≥ 1,2× `gemm_naive`, và `gemm` đạt 25–60% trần FP32 ở xung đo (cuBLAS 84–97%) | Bậc nào không đạt: tối ưu đó không có tác dụng trên máy này. Ghi lại như đo được; không sửa code sau khi thấy số rồi đo lại mà không ghi | Chưa chạy |
| E9 | Conv trực tiếp. CPU: `direct` chậm hơn `caffe` ít nhất 2× mỗi ảnh. GPU: `direct` nhanh hơn `caffe` ở batch ≥ 64 (vì không phải ghi rồi đọc lại ma trận cột 1,4 MB mỗi ảnh) | GPU `direct` chậm hơn: đọc lại ảnh 121 lần qua cache tốn hơn im2col + GEMM | Chưa chạy |
| E10 | Laptop (nếu chạy), WSL, C 1 luồng, n ≥ 4096: băng thông đọc tốt nhất 20–35 GB/s; `matvec` ≥ 85% con số đó | `matvec` < 85%: như E5 | Chưa chạy |

Không dự đoán: con số tuyệt đối của bài báo, vì phần cứng khác. Cũng không dự đoán `batched` trên GPU: lần đo
PyTorch cho thấy nó phụ thuộc mạnh vào xung nhịp.

**Ghi chú minh bạch:** lần trước, dự đoán băng thông không được commit trước khi chạy (xem `PREDICTIONS.md`). Lần
này nên commit file này cùng code **trước** khi chạy Kaggle, để git chứng minh thứ tự. Nếu sửa code sau lần chạy thử
trên laptop, ghi lại sửa gì và vì sao.

**Đã làm sau khi viết bảng dự đoán (25/11/2026), các cột dự đoán không sửa:**

- Cài gcc 13.3 trong WSL. `bench_seq --check-only` qua cả 28 kiểm tra.
- Chạy thử một lần lưới `--quick` trên laptop (batch 1–3, n ≤ 1000) và đã xem các số đó. Ở n ≤ 1000, dữ liệu
  nằm gọn trong cache nên các số này không nói gì về E5–E7. Số đáng kể duy nhất là conv kiểu Caffe trên laptop:
  khoảng 1 ms/ảnh, GEMM khoảng 75 GFLOP/s. Đây là Ryzen, không phải CPU Kaggle mà E2 dự đoán.
- Sửa sau lần chạy thử:
  - (1) thêm cờ `-ffp-contract=fast`. Mã máy cho thấy gcc ở chế độ `-std=c11` không sinh lệnh FMA nào. Trên CPU
    Intel của Kaggle, điều đó giới hạn bản C ở nửa trần, trong khi MKL và kernel CUDA đều dùng FMA. Trên Ryzen
    tốc độ gần như không đổi;
  - (2) `CC = gcc` trong Makefile, vì `CC ?=` không có tác dụng;
  - (3) `--quick` không còn ghi đè các tùy chọn ghi rõ trên dòng lệnh;
  - (4) hằng số của lớp conv chuyển từ `enum` sang `#define` (cảnh báo của C++20);
  - (5) `die` đánh dấu `noreturn`.
- Bản CUDA chưa qua nvcc:
  - g++ kiểm tra kiểu toàn bộ file, dùng một header thay thế cho CUDA;
  - một bộ giả lập CUDA trên CPU (mỗi luồng CUDA là một luồng hệ điều hành, có `__syncthreads` và warp shuffle)
    chạy qua cả 42 kiểm tra đúng đắn và lưới `--quick`;
  - bộ giả lập chỉ dùng để kiểm tra, không nằm trong repo; thời gian đo trên nó không có ý nghĩa.

## 7. Các bước và chi phí

| Bước | Ai làm | Ước tính |
|---|---|---|
| 1. Cài gcc trong WSL (`sudo apt-get install build-essential`), chạy `--check-only` và `--quick` của `bench_seq` trên laptop, sửa lỗi biên dịch | Bạn cài; Claude chạy và sửa | 10–20 phút |
| 2. Commit code và kế hoạch này trước khi đo | Claude, khi bạn đồng ý | – |
| 3. Kaggle, GPU T4: import `kaggle/hpc_c_cuda_kaggle.ipynb`, Run All, tải `results_c_cuda.zip` | Bạn | 30–45 phút GPU, ước tính chưa đo. Bản CUDA lần đầu được biên dịch ở đây; nếu lỗi biên dịch, notebook dừng sau vài phút |
| 4. Laptop: 3 lần `bench_seq` trong WSL | Claude | khoảng 15–25 phút, ước tính |
| 5. Phân tích: `code/analyze_c_cuda.py`, điền cột Kết quả, cập nhật `README.md`/`ANALYSIS.md` | Claude | – |
| 6. Tùy chọn: thêm một phiên Kaggle khác để xem kết quả có lặp lại trên máy khác không | Bạn | 30–45 phút GPU |

## 8. Rủi ro và giới hạn đã biết

- **Bản CUDA chưa được biên dịch lần nào:** laptop không có nvcc. Lỗi biên dịch hoặc lỗi kiểm tra đúng đắn sẽ lộ ở
  ô đầu notebook, trước khi đo.
- **Tốc độ GEMM trên CPU phụ thuộc vào việc gcc vector hóa micro-kernel 4×16.** Nếu E2 sai theo hướng chậm, cần
  xem mã máy (`-fopt-info-vec`, `objdump`) trước khi kết luận về thuật toán.
- **Máy Kaggle có thể khác lần đo PyTorch** (CPU ảo, có thể khác thế hệ). So `lscpu` và tên GPU trước khi đặt hai
  bản cạnh nhau.
- **GEMM tự viết chậm hơn cuBLAS nhiều ở ma trận vuông**, nên tỷ lệ GPU/CPU của bản tự viết đo cả chất lượng code
  của mình, không chỉ phần cứng. Báo cáo cần nói rõ điều này khi so với bài báo, vốn dùng Caffe với BLAS và cuBLAS.
