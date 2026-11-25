# Bài nộp: HPC 20261B, Nhóm 10, Parallel and Distributed Deep Learning (MPI + CUDA)

**Repo:** https://gitlab.nqtuan0192.me/vu/hpc_20261b_group10_paralleldistributeddeeplearning_mpi-cuda

| Thành viên | MSSV |
|---|---|
| Vũ Ngọc Vinh | 20262094M |
| Nguyễn Văn Hiếu | 20262227M |
| Nguyễn Ngọc Tuấn | 20262434M |
| Nguyễn Đức Thịnh | 20262442M |
| Lattana Aphixay | 20262696M |
| Nguyễn Mạnh Đức | 20262139M |

Thành phần theo yêu cầu của đề:

| # | Thành phần | Vị trí |
|---|---|---|
| 1 | Báo cáo | `report/BAO_CAO.md` (bản PDF `report/BAO_CAO.pdf` có trong file zip nộp) |
| 2 | Slide | `slides/HPC_Nhom10_ParallelDistributedDL.pptx` |
| 3 | Mã C (tuần tự) | bảng dưới, cột "Bản tuần tự" |
| 4 | Mã song song (MPI, CUDA) | bảng dưới, cột "Bản song song" |
| 5 | README: cách biên dịch và chạy bản tuần tự và bản song song | `README.md` mục 3; chi tiết trong `README.md` của từng thư mục |

| Cấu hình | Nội dung | Bản tuần tự | Bản song song |
|---|---|---|---|
| A | Song song mô hình (§2), 4× H100 | `vinhvn/src/mlp_seq.c` | `vinhvn/src/mlp_cuda.cu` (CUDA) |
| B | CPU so với GPU (§3), T4 | `hieunv/c_cuda/bench_seq.c` | `hieunv/c_cuda/bench_cuda.cu` (CUDA) |
| C | CNN: Sync SGD, Parallel SGD, ADMM, Downpour, 6 allreduce (§5) | `tuannn/code/sequential/seq_train.c` | `tuannn/code/src/` (MPI, OpenMP, CUDA) |
| D | CNN: tới cùng độ chính xác, gradient trễ (§5.2) | `thinhnd/cnn/src/seq/train_seq.c` | `thinhnd/cnn/src/mpi/train_mpi.c` (MPI), `thinhnd/cnn/src/cuda/` (CUDA) |
| E | Allreduce tự viết trên nhiều GPU (§5) | `ltnaphixay/src/seq/mlp_seq.c` | `ltnaphixay/src/cuda/`, `ltnaphixay/src/allreduce/` (CUDA) |

Kết quả đo gốc (log, CSV, JSON) nằm trong thư mục `results/`, `runs/` hoặc `experiments/results/` của từng phần; nguồn của
từng con số trong báo cáo ghi ở Phụ lục B của báo cáo.
