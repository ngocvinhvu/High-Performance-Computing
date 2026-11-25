# Build the sequential C and the CUDA benchmarks (Linux: Kaggle, WSL Ubuntu).
#   make seq                 -> bench_seq   (needs gcc)
#   make cuda ARCH=sm_75     -> bench_cuda  (needs nvcc; sm_75 = Tesla T4, sm_60 = P100)
#   make check               -> correctness checks of both programs
# No -fopenmp, no BLAS, no cuBLAS/cuDNN: only libc, libm and the CUDA runtime are linked.

# -ffp-contract=fast lets gcc fuse a*b + c into FMA instructions. In ISO C mode (-std=c11) gcc turns this off by
# default; on Intel cores mul, add and FMA share the same ports, so without it the C code would be capped at half the
# FMA peak, while MKL and the CUDA code (fmaf) use FMA. It is not -ffast-math: no reordering of sums.
CC      = gcc
CFLAGS  = -O3 -march=native -std=c11 -ffp-contract=fast -Wall -Wextra -fno-tree-loop-distribute-patterns
NVCC    = nvcc
ARCH    ?= sm_75
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

.PHONY: all seq cuda check clean
all: seq cuda
seq: bench_seq
cuda: bench_cuda

bench_seq: bench_seq.c common.h
	$(CC) $(CFLAGS) -o $@ bench_seq.c -lm
	@echo "$(CC) $(CFLAGS) -o $@ bench_seq.c -lm" > build_seq.txt
	@$(CC) --version | head -1 >> build_seq.txt

bench_cuda: bench_cuda.cu common.h
	$(NVCC) $(NVFLAGS) -o $@ bench_cuda.cu
	@echo "$(NVCC) $(NVFLAGS) -o $@ bench_cuda.cu" > build_cuda.txt
	@$(NVCC) --version | tail -2 >> build_cuda.txt

check: all
	./bench_seq --check-only
	./bench_cuda --check-only

clean:
	rm -f bench_seq bench_cuda build_seq.txt build_cuda.txt
