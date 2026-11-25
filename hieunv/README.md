# Đo lại CPU so với GPU trong "Parallel and Distributed Deep Learning"

Phần việc của thư mục `hieunv/`: chạy lại thực nghiệm mục 3 của bài báo (Hegde & Usmani, xem [REFERENCES.md](REFERENCES.md)).
Bài báo gốc không nằm trong repo.

## Mục tiêu

Bài báo đo thời gian forward của lớp convolution theo batch size (Hình 1–2) và nhân ma trận–vector theo kích thước n
(Hình 3–4), trên CPU và GPU. Câu hỏi: các kết luận của bài (convolution tuyến tính theo batch, matvec O(n²), speedup
tăng theo batch và theo n, GPU nhanh hơn khoảng 40 lần) có lặp lại trên phần cứng hiện tại không? Nếu con số khác
thì đo được nguyên nhân nào? Thư mục này **chỉ đo lại mục 3**; phần SGD phân tán của bài không được chạy ở đây.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| [`code/`](code/) | Code đo (PyTorch): `bench_conv.py`, `bench_fc.py`, `run_all.py` (một lần chạy), `run_session.py` (ba lần chạy trên laptop), `aggregate.py`, `plot.py`, `gpu_diag.py`. Chi tiết ở [code/README.md](code/README.md) |
| [`kaggle/hpc_replication_kaggle.ipynb`](kaggle/hpc_replication_kaggle.ipynb) | Notebook tự chứa để chạy trên Kaggle, sinh từ `code/make_kaggle_notebook.py` |
| [`c_cuda/`](c_cuda/) | Bản không dùng thư viện song song (theo yêu cầu bài tập lớn): `bench_seq.c` (C tuần tự, 1 luồng) và `bench_cuda.cu` (kernel CUDA tự viết, không cuBLAS/cuDNN). Chưa chạy. Chi tiết ở [c_cuda/README.md](c_cuda/README.md) |
| [`PLAN_C_CUDA.md`](PLAN_C_CUDA.md) | Kế hoạch đo lại bằng bản C/CUDA và dự đoán E1–E10, viết trước khi chạy |
| [`kaggle/hpc_c_cuda_kaggle.ipynb`](kaggle/hpc_c_cuda_kaggle.ipynb) | Notebook Kaggle cho bản C/CUDA, sinh từ `code/make_c_cuda_notebook.py` |
| [`results/`](results/) | Dữ liệu gốc (CSV, `env.json`, log) và hình. Nên dùng thư mục nào: [results/README.md](results/README.md) |
| [`ANALYSIS.md`](ANALYSIS.md) | Phân tích: đối chiếu từng kết luận của bài báo, mô hình, giới hạn, bước kiểm chứng tiếp |
| [`PREDICTIONS.md`](PREDICTIONS.md) | Dự đoán ghi trước khi đo băng thông (kiểm chứng ANALYSIS mục 7); cột kết quả điền sau khi chạy |
| [`results/analysis/`](results/analysis/) | Bảng số (`tables.md`) và 3 hình phân tích (Hình 7–9), sinh từ `code/analyze.py` |
| [`results/paper_digitized.csv`](results/paper_digitized.csv) | Điểm đọc từ Hình 1–4 của bài báo, kèm sai số đọc |
| [`REFERENCES.md`](REFERENCES.md) | Trích dẫn và đường dẫn tài liệu |

Dữ liệu chính thức: `results/kaggle/agg_20261122_074407` và `results/laptop/agg_20261122_152118`.

## Cách chạy lại

Môi trường đã dùng:

| | Laptop (CPU) | Kaggle (CPU + GPU) |
|---|---|---|
| CPU | AMD Ryzen 7 5800H, torch dùng tối đa 8 luồng | Intel Xeon 2.00 GHz, 4 vCPU, torch dùng tối đa 2 luồng |
| GPU | Không có GPU CUDA | Tesla T4 (chỉ dùng một GPU) |
| Phần mềm | Windows 10, Python 3.12.10, PyTorch 2.15.0+cpu | Linux, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 |

**Laptop.** Cắm sạc, chọn chế độ "Best performance", đóng các ứng dụng khác. Phiên chính gồm 3 lần chạy, mỗi lần khoảng
6 phút cộng 60 giây nghỉ trước mỗi lần, tổng khoảng 21 phút. Chạy từ thư mục gốc của repo:

```powershell
py -V:3.12 -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```powershell
pip install -r hieunv\code\requirements.txt
```

```powershell
cd hieunv\code
```

```powershell
python run_all.py --machine laptop --quick
```

```powershell
python run_session.py --machine laptop
```

Lệnh `--quick` chỉ để thử trong vài giây. Trên Windows, `run_session.py` ghi `cpu_perf.csv`, trong đó dòng tiêu đề có
**tên máy**; hãy thay tên máy trước khi commit (bản trong repo đã thay bằng `LAPTOP`).

**Kaggle.** Mở notebook bằng File → Import Notebook, chọn Accelerator **GPU T4 x2**, bấm Run All. Notebook chạy một lần
chẩn đoán GPU nguội rồi 3 lần đo (mỗi lần khoảng 7 phút). Tải `results_kaggle.zip` ở panel Output về `results/kaggle/`.

**Tạo lại bảng và hình phân tích** (giai đoạn 5; không cần đo lại, chạy trong `hieunv\code`):

```powershell
python analyze.py
```

**Chỉ vẽ lại hình và bảng từ dữ liệu có sẵn** (chạy trong `hieunv\code`, không cần đo lại; `summary.md` sinh ra giống hệt bản đã commit):

```powershell
python plot.py ..\results\kaggle\agg_20261122_074407 --figdir $env:TEMP\figures_moi
```

## Kết quả

Điều kiện đo: ảnh xám 224×224, 96 filter 11×11, stride 4 (đầu ra 54×54); bản chính là kiểu Caffe (im2col rồi nhân ma
trận cho **từng ảnh**). Không tính thời gian copy dữ liệu giữa CPU và GPU. Mỗi điểm: 3 lần khởi động và 30 lần tính giờ;
**3 lần chạy** trên mỗi máy (cùng một phiên), thứ tự kích thước xáo trộn. Số dưới đây là trung bình 3 lần; ngoặc đơn là
khoảng của từng lần chạy.

### Dự đoán trước khi chạy và kết quả

Dự đoán ghi trong kế hoạch ngày 21/11/2026, trước khi viết code đo. Kế hoạch đó **không nằm trong git** nên không kiểm
chứng được thứ tự thời gian từ lịch sử repo.

| Dự đoán | Kết quả |
|---|---|
| Thời gian convolution tuyến tính theo batch | Đúng: với kiểu Caffe, R² ≥ 0,998 ở mọi lần chạy chính thức, CPU (1, 2, 8 luồng) và GPU |
| Matvec tăng theo O(n²) | Đúng trên CPU: số mũ 2,0–2,2 tùy khoảng khớp. Trên GPU chỉ xấp xỉ: 1,94 (n ≥ 4096) hoặc 1,75 (n ≥ 1024) |
| Speedup tăng theo batch và theo n | Convolution: tăng tới batch khoảng 32 rồi phẳng ở 26–29× (từng lần). Matvec: tăng đến n lớn nhất đã đo (0,43× → 21,7×) |
| Tỷ lệ GPU/CPU khác 40 lần nhưng xu hướng giống | Đúng: 27,8× (conv) và 21,7× (matvec, n = 16384). Kế hoạch không dự đoán tỷ lệ sẽ lớn hơn hay nhỏ hơn 40 |

Các hướng giải thích khác trong kế hoạch (vòng lặp từng ảnh, chi phí khởi chạy kernel, giới hạn tính toán/băng thông) và kết quả, kể cả phần
không đúng: [ANALYSIS.md](ANALYSIS.md) mục 5. Dự đoán của phép đo băng thông sau đó (D1–D6, ghi trước khi đo) ở [PREDICTIONS.md](PREDICTIONS.md):
D1, D2 (gần như hoàn toàn), D3, D5 đúng; **D4 sai** (đoán 22–28×, đo được 35× ở B = 64).

### Các con số chính

**Convolution, kiểu Caffe (độ dốc của đường thời gian theo batch, ms/ảnh):**

| | Kaggle | Laptop | Bài báo |
|---|---|---|---|
| CPU 1 luồng | 1,88 (1,86–1,90) | 2,00 (1,99–2,02) | 6,45 |
| CPU nhiều luồng | 1,26 (1,24–1,30) với 2 luồng | 0,526 (0,522–0,531) với 8 luồng, tức 3,8×, hiệu suất khoảng 48% | không nêu |
| GPU | 0,0677 (0,0648–0,0694) | không có | 0,16 |
| Tỷ lệ CPU 1 luồng / GPU | **27,8×** (28,6 / 27,4 / 27,3) | – | khoảng 40× |

Đơn vị ms/ảnh: Hình 1 của bài có trục "Time (ms)" theo batch size (đọc hình ra độ dốc CPU 6,32, sát 6,45 trong phần chữ).

**Speedup conv theo batch, CPU 1 luồng / GPU, Kaggle** (tỷ lệ trung bình 3 lần chạy): batch 1 là 18,6×, batch 16 là
26,1×, từ batch 32 trở đi 27–28× (từng lần 26–29×). Batch 8 là 21,4× nhưng **lẫn một lần đo hỏng**: lần 3 chỉ 14,2× vì
GPU đang bị giảm xung nhịp; hai lần còn lại là 25,2× và 24,9×. Điểm này chưa bị loại khỏi số liệu tổng hợp.

**Matvec** (n = 16384, dòng CPU là 1 luồng):

| | Kaggle | Laptop |
|---|---|---|
| CPU 1 luồng | 87,3 ms (12,3 GB/s) | 38,4 ms (28,1 GB/s) |
| GPU T4 | 4,02 ms (267 GB/s), tức 21,7× so với CPU 1 luồng | – |
| CPU 8 luồng / 1 luồng | – | 1,03× trung bình, nhưng từng lần là 1,12 / 1,00 / 0,96: không khác 1 |
| Số mũ k (n ≥ 4096 / n ≥ 1024), CPU 1 luồng | 2,07 / 2,21 | 2,01 / 2,13 |

Khoảng n ≥ 4096 được chọn **sau khi đã thấy** kết quả với n ≥ 1024, nên ghi cả hai. Ở n = 256, GPU **chậm hơn** CPU
(0,43×); từ n = 512 GPU mới nhanh hơn. Bài báo cũng ghi hệ số chênh của matvec là "khoảng 40 lần"; chúng tôi đo được
21,7× trên Kaggle (có thể không cùng phép, xem [ANALYSIS.md](ANALYSIS.md) mục 3).

**Cách cài đặt thay đổi tỷ lệ rất nhiều.** Với `nn.Conv2d` trên cả hai thiết bị, tỷ lệ CPU 1 luồng / GPU trên Kaggle chỉ
còn khoảng 14× (15,8 / 13,4 / 13,0 từng lần): so với kiểu Caffe, CPU (oneDNN) nhanh hơn 2,4× còn GPU (cuDNN, đo khi bị hạ xung) chỉ nhanh
hơn 1,2×. Con số 40× hay 27,8×
vì vậy gắn với cách cài đặt kiểu Caffe, không phải là tỷ lệ của "convolution nói chung".

**GEMM n = 4096 (phần thêm, vì chú thích Hình 3 của bài ghi "matrix-matrix", dù tiêu đề trong hình ghi "matrix vector"):** Kaggle CPU 1 luồng 129 GFLOP/s, GPU 2,97–3,49 TFLOP/s;
laptop 106 GFLOP/s (1 luồng) và 475 GFLOP/s (8 luồng). Đây là trạng thái T4 bị giới hạn công suất, không phải đỉnh.

Hình và bảng đầy đủ: [`results/kaggle/agg_20261122_074407/figures/`](results/kaggle/agg_20261122_074407/figures/) và
[`results/laptop/agg_20261122_152118/figures/`](results/laptop/agg_20261122_152118/figures/).

### Đã chứng minh (trong phạm vi các lần đo này)

- Thời gian convolution tuyến tính theo batch trên cả CPU và GPU; matvec xấp xỉ bậc hai theo n trên CPU ở cả hai máy.
- Tỷ lệ CPU 1 luồng / GPU cho convolution kiểu Caffe là 27–29× trên máy Kaggle, ổn định giữa 3 lần chạy. Speedup tăng theo batch rồi phẳng.
- Bậc nhảy thời gian GPU theo batch (thấy ở lần chạy đầu: 109 µs/ảnh ở batch 64, 69 µs/ảnh từ batch 128) **không phải do batch size**: khi xáo trộn thứ tự
  đo thì bậc nhảy biến mất, và thời gian × xung nhịp mỗi ảnh (batch ≥ 16) luôn nằm trong 93.500–100.000 chu kỳ trong khi xung nhịp
  đổi từ 1005 đến 1510 MHz (tương quan giữa thời gian và 1/xung nhịp: 0,994). Nghĩa là thời gian GPU chủ yếu do xung nhịp quyết định.
  Riêng lần chạy đầu **không ghi xung nhịp**, nên với lần đó đây là suy luận.

- **Matvec (FC) bị giới hạn bởi băng thông bộ nhớ trên Kaggle** (PREDICTIONS.md D1–D3, ghi trước khi đo): kernel đọc độc lập (`sum`, `dot`) đạt 13–14 GB/s ở CPU 1 luồng và 253–272 GB/s ở GPU; matvec đạt 92–100% và 97–99% của mức đó. Tỷ lệ băng thông GPU/CPU (18–20×) khớp tỷ lệ matvec (17,6–21,4×) trong 7%. Chỉ một máy, 3 lần chạy.

### Chưa chứng minh

- **Nguyên nhân** của khác biệt so với con số 40× của bài báo (27,8× ở đây): phần cứng, phần mềm và cách cài đặt đều khác
  nhau, chúng tôi không tách được từng yếu tố.
- **Speedup FC bằng tỷ lệ băng thông bộ nhớ** mới được kiểm chứng trên **một** máy Kaggle (khớp trong 7%, xem bên trên); chưa biết có lặp lại ở máy khác.
- Vì sao matvec 8 luồng trên laptop không nhanh hơn 1 luồng.
- **Vì sao GEMM của conv (ma trận 96×121 nhỏ) chỉ đạt 26% trần FP32** (đã đo riêng: im2col chiếm 46% chu kỳ; phần còn lại chưa rõ nguyên nhân).
- Vì sao số tuyệt đối FC của bài báo không khớp matvec thường (CPU khoảng 0,28 GB/s, GPU khoảng 14–16 GB/s): giả thuyết là nhân với nhiều vector (khoảng 64–78), **chưa kiểm chứng**; xem ANALYSIS.md mục 3.

## Giới hạn

- **Mỗi máy chỉ có 3 lần chạy trong cùng một phiên.** Dao động giữa các lần là trôi trên một máy, không phải giữa các máy. Kế hoạch
  định chạy 2–3 phiên khác nhau, chưa làm. Trên Kaggle, GPU nóng dần (khoảng 54 → 80 °C) nên xung nhịp khi đo conv giảm khoảng 7%
  từ lần 1 đến lần 3.
- **T4 bị giới hạn ở khoảng 70 W**, nên GEMM và cuDNN chạy khi GPU bị hạ xung. Số của các cách cài đặt khác nhau trên GPU không so
  trực tiếp được nếu không quy về chu kỳ.
- Bước làm nóng GPU của chính phép đo khiến điểm đo đầu bị hạ xung; batch 8 của lần 3 là hệ quả.
- Laptop không có GPU CUDA nên không có tỷ lệ GPU/CPU trên laptop. **Không nên ghép CPU laptop với GPU Kaggle.**
- Matvec 1 luồng trên laptop ở n = 16384: lần 1 là 41,6 ms, lần 2 và 3 là 36,7–36,8 ms, nên giá trị 38,4 ms có sai số khoảng 7%.
- Trên Windows, `conv2d` (oneDNN) của CPU vẫn bao gồm chi phí cấp phát bộ nhớ đầu ra. Nhiều điểm `conv2d` ngắn và dao động lớn.
- Batched (gộp cả batch vào một GEMM) so với từng ảnh **ngược dấu giữa hai máy**: trên Kaggle CPU nhanh hơn 8–9% (1 luồng) và
  20–24% (2 luồng), trên laptop chậm hơn 7–9% (1 luồng) và 28–30% (8 luồng), theo từng lần chạy. Chưa biết vì sao.
- Các lần chạy cũ (`run_kaggle_20261121_074742`, `run_trial_balanced`) có lỗi đo ở `batched` và `conv2d` trên CPU; không trích số từ đó
  (xem [results/README.md](results/README.md)).
- CPU Kaggle là máy ảo: `lscpu` báo 2 nhân × 2 luồng, L3 38,5 MB (đã ghi ở `results/kaggle/lscpu_20261122_095300.txt`). Tên và thông số phần cứng của bài báo lấy từ phần chữ của bài.
- Phần phân tích đối chiếu từng kết luận của bài báo ở [ANALYSIS.md](ANALYSIS.md); **báo cáo và slide chưa làm**.
