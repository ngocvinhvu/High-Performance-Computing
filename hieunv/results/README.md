# Kết quả đo lại mục 3 của "Parallel and Distributed Deep Learning"

Thời gian forward của lớp convolution theo batch size (Hình 1–2 của paper) và nhân ma trận–vector theo
kích thước n (Hình 3–4), trên CPU và GPU. Code ở [`../code`](../code), notebook Kaggle ở [`../kaggle`](../kaggle),
tài liệu tham khảo ở [`../REFERENCES.md`](../REFERENCES.md).

## Nên dùng thư mục nào

| Thư mục | Trạng thái | Nội dung |
|---|---|---|
| `kaggle/agg_20261122_074407` | **Chính** | Gộp 3 lần chạy Kaggle (Tesla T4 + Xeon 2.0 GHz, cùng một máy): CSV, hình và `summary.md` (trong `figures/`), `gpu_diag.md` |
| `kaggle/run{1,2,3}_20261122_074407` | Dữ liệu thô của bản chính | Mỗi lần: 30 lần đo/điểm, thứ tự xáo trộn, GPU trước; có `gpu_clocks.csv` |
| `kaggle/diag_cold_20261122_074407` | Chẩn đoán | GPU nguội, batch tăng dần; dùng để tìm nguyên nhân bậc nhảy thời gian GPU |
| `laptop/agg_20261122_152118` | **Chính** | Gộp 3 lần chạy trên laptop (Ryzen 7 5800H, không có GPU CUDA) |
| `laptop/run{1,2,3}_20261122_152118` | Dữ liệu thô của bản chính | 30 lần đo/điểm; có `cpu_perf.csv` (xung nhịp và tải nền) |
| `analysis` | Phân tích | Bảng số (`tables.md`, `mem_tables.md`) và Hình 7–10, sinh bởi `../code/analyze.py` và `../code/analyze_mem.py`; xem `../ANALYSIS.md` |
| `kaggle/bw{1,2,3}_20261122_095300` | Kiểm chứng | Đo băng thông độc lập, FC nhiều vector, tách im2col/GEMM (`bw.csv`, `fcb.csv`, `step.csv`); dự đoán ở `../PREDICTIONS.md` |
| `kaggle/lscpu_20261122_095300.txt` | Kiểm chứng | Thông tin CPU của máy ảo Kaggle (2 nhân × 2 luồng, L3 38,5 MB) |
| `kaggle/c_cuda{1,2,3}_20261125_050209` | **Chính thức (bản không thư viện)** | C tuần tự (CPU) và CUDA tự viết (GPU T4) trên cùng máy Kaggle, 3 lần chạy (GPU 30 lần tính giờ, CPU 10). Mỗi thư mục có `conv.csv`, `fc.csv`, `bw.csv`, `fcb.csv`, `step.csv`, `env_seq.json`, `env_cuda.json`, log, `build_*.txt`, `gpu_clocks.csv`. Bảng số: `analysis/c_cuda_tables.md` (sinh bởi `../code/analyze_c_cuda.py`); đối chiếu dự đoán: `../PLAN_C_CUDA.md` mục 6 |
| `kaggle/lscpu_c_cuda_20261125_050209.txt` | Kiểm chứng | Thông tin CPU của máy Kaggle trong phiên chạy bản C/CUDA (Xeon 2,0 GHz, 4 vCPU, có AVX-512) |
| `paper_digitized.csv` | Số liệu bài báo | Điểm đọc từ Hình 1–4 của bài báo, kèm sai số đọc |
| `kaggle/run_kaggle_20261121_074742` | **Cũ** | 1 lần chạy, 10 lần đo. Cột CPU của `batched` và `conv2d` sai (tính cả chi phí cấp phát bộ nhớ); thời gian GPU bị ảnh hưởng bởi xung nhịp |
| `laptop/run_trial_balanced` | **Cũ** | Chạy thử, có cùng lỗi cấp phát, matvec bị nhiễu (52,5 ms ở n=16384, bản chính là 38,4 ms) |

Các thư mục cũ được giữ lại để đối chiếu, không nên trích số từ đó (trừ khi ghi rõ giới hạn trên).

## Kết quả chính

Mọi số là trung bình 3 lần chạy; mỗi điểm đo 3 lần khởi động và 30 lần tính giờ. Conv dùng cách cài đặt kiểu
Caffe (im2col rồi nhân ma trận cho từng ảnh), ảnh xám 224×224, 96 filter 11×11, stride 4.

| | Kaggle | Laptop | Paper |
|---|---|---|---|
| Conv, CPU 1 luồng (ms/ảnh) | 1,88 | 2,00 | 6,45 |
| Conv, CPU nhiều luồng (ms/ảnh) | 1,26 (2 luồng) | 0,526 (8 luồng) | không nêu |
| Conv, GPU (ms/ảnh) | 0,068 (Tesla T4) | không có | 0,16 (GRID K520) |
| Tỷ lệ CPU 1 luồng / GPU (từng lần chạy: 28,6 / 27,4 / 27,3) | 27,8 | – | ≈ 40 |
| Matvec n = 16384, CPU 1 luồng | 87,3 ms | 38,4 ms | không nêu |
| Matvec n = 16384, GPU | 4,0 ms | – | không nêu |
| Matvec, số mũ k khớp từ n ≥ 4096 (CPU 1 luồng) | 2,07 | 2,01 | O(n²) |
| Matvec, số mũ k khớp từ n ≥ 1024 (CPU 1 luồng) | 2,21 | 2,13 | O(n²) |

Đơn vị: trục Hình 1 của bài báo là thời gian (ms) theo batch size nên độ dốc là ms/ảnh. Số mũ matvec phụ thuộc khoảng khớp, khoảng
n ≥ 4096 được chọn sau khi thấy kết quả với khoảng n ≥ 1024, nên ghi cả hai.

## Những gì đã chứng minh được

- Thời gian conv tăng tuyến tính theo batch size trên cả CPU và GPU (R² ≈ 1,000).
- Matvec tăng xấp xỉ theo n² trên CPU (cả hai máy, số mũ 2,0–2,2 tùy khoảng khớp). Trên GPU số mũ phụ thuộc khoảng
  khớp: 1,94 (n ≥ 4096) hoặc 1,75 (n ≥ 1024).
- **Bậc nhảy thời gian mỗi ảnh trên GPU không phải do batch size mà do xung nhịp GPU.** Ở lần chẩn đoán và 3 lần chạy
  chính, thời gian × xung nhịp mỗi ảnh (batch ≥ 16) nằm trong 93.500–100.000 chu kỳ, trong khi xung nhịp đi từ 1005 đến
  1510 MHz; khi xáo trộn thứ tự đo, bậc nhảy biến mất. Xem `kaggle/agg_20261122_074407/gpu_diag.md`. Riêng lần chạy đầu
  (109 → 69 µs) không ghi xung nhịp nên ở lần đó đây là suy luận.

## Giới hạn

- Ba lần chạy của mỗi máy nằm trong cùng một phiên: sai số giữa các lần là dao động trên một máy, không phải
  giữa các máy. Trên Kaggle, GPU nóng dần (khoảng 54 → 80 °C) nên xung nhịp giảm khoảng 7% từ lần 1 đến lần 3.
- Bước làm nóng GPU khiến GPU bị hạ xung ngay sau đó; điểm conv batch 8 của lần 3 (GPU 14,2× CPU) bị ảnh hưởng.
  Điểm này chưa bị loại, nên thanh sai số ở batch 8 rất dài.
- GEMM và cuDNN trên T4 bị giới hạn công suất (khoảng 70 W) nên số TFLOP/s là trạng thái nóng, không phải đỉnh.
- Khi so các cách cài đặt trên GPU, mỗi khối chạy ở xung nhịp khác nhau.
- Trên Windows, `conv2d` (oneDNN) của CPU vẫn bao gồm chi phí cấp phát bộ nhớ đầu ra.
- Nhiều điểm rất ngắn (nhất là `conv2d`) có dao động lớn giữa các lần đo.
- Tỷ lệ CPU/GPU chỉ có nghĩa khi CPU và GPU trên cùng một máy (Kaggle), không kết hợp với số của laptop.

## Định dạng

`conv.csv`, `fc.csv`: mỗi dòng một điểm đo (trong thư mục `agg_*` là trung bình 3 lần; `runs_*.csv` là dữ liệu gốc
từng lần). Cột `issue_ms` là thời gian CPU phát lệnh (chỉ có ý nghĩa trên GPU). `cpu_perf.csv` (laptop) chứa xung nhịp
theo phần trăm tần số cơ sở và tải CPU, mỗi giây một dòng; tên máy đã được thay bằng `LAPTOP`.
Tái tạo hình: `python ../code/plot.py <thư mục>`; gộp các lần chạy: `python ../code/aggregate.py <run1> <run2> <run3> --out <dir>`.
