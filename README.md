# Nhóm 10: repo làm việc chung

Repo chung của **Nhóm 10**. Mỗi thành viên có **một thư mục riêng** và đóng góp phần nghiên cứu, mã nguồn và kết quả của mình vào đó.
Tài liệu này giúp bạn (thành viên mới hoặc người đọc) hiểu repo được tổ chức thế nào và cách đóng góp mà không đụng vào phần của người khác.

## 1. Cấu trúc repo

```
.
├── README.md            # tài liệu này (chỉ sửa qua Pull Request, xem mục 4)
├── <tên-thành-viên>/    # mỗi người một thư mục, ví dụ: vinhvn/
│   ├── README.md        # BẮT BUỘC: giới thiệu phần việc của bạn (mẫu ở mục 3)
│   └── ...              # mã nguồn, dữ liệu nhỏ, kết quả, ghi chú
└── <tên-thành-viên-khác>/
```

Quy ước đặt tên thư mục: chữ thường, không dấu, không khoảng trắng, dùng một tên cố định (ví dụ tên tài khoản hoặc mã sinh viên của bạn) và **không đổi tên sau khi đã push** để liên kết và lịch sử không bị gãy.

## 2. Danh sách thành viên và thư mục

| Thư mục | Thành viên | GitHub | Chủ đề / phần việc |
|---|---|---|---|
| [`vinhvn/`](vinhvn/) | (điền) | (điền) | Model parallelism (§2): pipeline và tensor parallelism trên nhiều GPU, tự cài đặt bằng C (tuần tự) và CUDA (song song), không dùng thư viện; đối chứng PyTorch trong `reference_pytorch/` |
| [`hieunv/`](hieunv/) | (điền) | (điền) | Đo lại mục 3 của Parallel and Distributed Deep Learning: CPU so với GPU (convolution, matvec), đối chiếu với bài báo, kiểm chứng băng thông bộ nhớ |
| [`tuannn/`](tuannn/) | Tuan Nguyen | [tuannt2906](https://github.com/tuannt2906) | Huấn luyện CNN song song/phân tán tự cài đặt bằng C/MPI/OpenMP/CUDA: bản tuần tự và bản song song, Sync SGD, Parallel SGD, ADMM, Downpour, 6 thuật toán allreduce |
| [`thinhnd/`](thinhnd/) | Nguyen Duc Thinh | [ducthinhh1002](https://github.com/ducthinhh1002) | CNN tự viết bằng C/MPI/CUDA chạy trên RTX 4060: so ở cùng accuracy, ảnh hưởng của gradient trễ theo learning rate, CPU so với GPU trên cùng laptop |
| [`ltnaphixay/`](ltnaphixay/) | LATTANA APHIXAY | [infinityrbx](https://github.com/infinityrbx) | SGD song song dữ liệu trên CUDA với allreduce tự viết (tree, ring, recursive doubling): đo chi phí truyền thông trên 2× T4 (PCIe) và 4× A100 (NVLink so với qua host), so với mô hình α–β của bài |
| [`ducnm/`](ducnm/) | (điền) | (điền) | Bài nền: huấn luyện hồi quy synthetic, bottleneck GEMM, CPU/GPU và mô hình chi phí communication trong gradient descent phân tán |
| (thư mục của bạn) | | | |

Khi thêm thư mục mới, thêm **một dòng** vào bảng này trong cùng Pull Request.

## 3. Mỗi thư mục cần có gì

Bắt buộc có `README.md` ở gốc thư mục của bạn. Mẫu gợi ý:

```markdown
# <Tên đề tài / phần việc>

## Mục tiêu
Câu hỏi bạn muốn trả lời, 2–4 câu.

## Cấu trúc thư mục
Mỗi thư mục con/tệp quan trọng làm gì.

## Cách chạy lại
Môi trường (phiên bản Python, thư viện, phần cứng nếu có) và các lệnh để tái lập kết quả.

## Kết quả
Con số chính, kèm điều kiện đo (dữ liệu, cấu hình, số lần chạy). Nói rõ điều gì đã chứng minh được và điều gì chưa.

## Giới hạn
Điều kết quả không chứng minh được, hoặc điều bạn chưa kiểm tra.
```

Gợi ý để người khác đọc và tái lập được phần của bạn:
- Ghi **dự đoán/giả thuyết trước khi chạy** nếu có thể, rồi đối chiếu với kết quả, kể cả khi dự đoán sai.
- Giữ **log và dữ liệu gốc** (CSV, JSON) cùng với bản tổng hợp, để người khác kiểm tra lại số liệu.
- Ghi số lần chạy lặp. Một lần chạy khác với năm lần chạy.
- Tên tệp kết quả nên có ngày hoặc mã đợt chạy.

## 4. Cách đóng góp

**Quy tắc chính: chỉ sửa trong thư mục của mình.** Việc này gần như loại bỏ xung đột khi nhiều người push cùng lúc.
Muốn sửa thư mục của người khác hoặc `README.md` gốc, hãy mở Pull Request và nhờ chủ thư mục xem trước.

Quy trình đề xuất (nhóm có thể điều chỉnh):

```bash
# 1. Lần đầu: lấy repo
git clone git@github.com:ngocvinhvu/High-Performance-Computing.git
cd High-Performance-Computing

# 2. Mỗi lần làm việc: cập nhật main, tạo nhánh riêng
git checkout main && git pull
git checkout -b <thư-mục-của-bạn>/<việc-đang-làm>     # ví dụ: vinhvn/them-ket-qua-dot-10

# 3. Làm việc trong thư mục của bạn rồi commit
git add <thư-mục-của-bạn>
git commit -m "Mô tả ngắn, nói rõ thay đổi gì và vì sao"

# 4. Đồng bộ lại trước khi push, rồi push nhánh của bạn
git pull --rebase origin main
git push -u origin <thư-mục-của-bạn>/<việc-đang-làm>
```

Sau đó mở Pull Request trên GitHub vào `main`. Nếu nhóm thống nhất cho phép push thẳng lên `main` đối với thư mục của chính mình thì
dùng `git pull --rebase` trước mỗi lần push là đủ.

Quy ước commit: một commit nói về một việc; mô tả bằng tiếng Việt hoặc tiếng Anh nhưng thống nhất trong cùng nhánh; không gộp mã và kết quả lớn không liên quan vào một commit.

## 5. Không được đưa lên repo

- **Thông tin đăng nhập:** `kubeconfig`, khoá SSH/API, token, mật khẩu, tệp `.env`. Nếu lỡ commit, **thu hồi và đổi thông tin đó ngay**, vì xoá tệp ở commit sau không xoá khỏi lịch sử.
- **Tệp lớn:** GitHub từ chối tệp trên 100 MB. Với mô hình, dataset hay checkpoint, ghi cách tải về trong `README.md` thay vì commit.
- **Tệp sinh ra:** `__pycache__/`, `*.pyc`, `.DS_Store`, thư mục môi trường ảo (`venv/`, `.venv/`), tệp tạm.
- **Tài liệu có bản quyền của người khác** (bài báo, sách PDF): dẫn nguồn và đường liên kết thay vì đưa tệp vào repo.
- **Thông tin cá nhân** không cần thiết: email, số điện thoại, địa chỉ.

Mỗi thư mục nên có `.gitignore` riêng nếu có tệp sinh ra hoặc thông tin nhạy cảm. Thư mục [`vinhvn/`](vinhvn/) có một `.gitignore` làm ví dụ.

## 6. Ví dụ để tham khảo

Thư mục [`vinhvn/`](vinhvn/) là một ví dụ đầy đủ: có `README.md` mô tả cấu trúc, `experiments/PLAN.md` ghi dự đoán trước khi chạy,
các bản tổng hợp kết quả (`experiments/results/SUMMARY_*.md`) và log gốc cạnh bản tổng hợp.

## 7. Cần giúp?

- Gặp xung đột khi `git pull --rebase`: dừng lại, đừng ép push (`--force`). Hỏi cả nhóm trước, vì ép push có thể xoá commit của người khác.
- Không chắc nên để gì trong thư mục của mình: bắt đầu từ `README.md` theo mẫu ở mục 3 rồi bổ sung dần.
