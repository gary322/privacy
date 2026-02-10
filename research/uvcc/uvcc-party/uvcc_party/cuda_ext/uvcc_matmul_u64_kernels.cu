#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

// u64 ring matmul (mod 2^64) implemented over int64 tensors that carry u64 bit-patterns.
// This is a simple tiled kernel intended for correctness and small/medium matrices.

static constexpr int UVCC_MATMUL_TILE = 16;

__global__ void uvcc_matmul_u64_kernel(
    const int64_t* __restrict__ A_i64,
    const int64_t* __restrict__ B_i64,
    int64_t* __restrict__ C_i64,
    int m,
    int k,
    int n) {
  const int row = (int)blockIdx.y * UVCC_MATMUL_TILE + (int)threadIdx.y;
  const int col = (int)blockIdx.x * UVCC_MATMUL_TILE + (int)threadIdx.x;

  __shared__ unsigned long long As[UVCC_MATMUL_TILE][UVCC_MATMUL_TILE];
  __shared__ unsigned long long Bs[UVCC_MATMUL_TILE][UVCC_MATMUL_TILE];

  unsigned long long acc = 0ULL;

  for (int t = 0; t < k; t += UVCC_MATMUL_TILE) {
    const int a_col = t + (int)threadIdx.x;
    const int b_row = t + (int)threadIdx.y;

    unsigned long long a = 0ULL;
    unsigned long long b = 0ULL;

    if (row < m && a_col < k) {
      // reinterpret int64 bits as u64
      a = (unsigned long long)A_i64[row * k + a_col];
    }
    if (b_row < k && col < n) {
      b = (unsigned long long)B_i64[b_row * n + col];
    }
    As[(int)threadIdx.y][(int)threadIdx.x] = a;
    Bs[(int)threadIdx.y][(int)threadIdx.x] = b;
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < UVCC_MATMUL_TILE; kk++) {
      acc += As[(int)threadIdx.y][kk] * Bs[kk][(int)threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    C_i64[row * n + col] = (int64_t)acc;
  }
}

torch::Tensor uvcc_matmul_u64_cuda(torch::Tensor A_i64, torch::Tensor B_i64) {
  TORCH_CHECK(A_i64.is_cuda(), "A_i64 must be CUDA");
  TORCH_CHECK(B_i64.is_cuda(), "B_i64 must be CUDA");
  TORCH_CHECK(A_i64.scalar_type() == torch::kInt64, "A_i64 must be int64");
  TORCH_CHECK(B_i64.scalar_type() == torch::kInt64, "B_i64 must be int64");
  TORCH_CHECK(A_i64.dim() == 2, "A_i64 must be 2D");
  TORCH_CHECK(B_i64.dim() == 2, "B_i64 must be 2D");
  TORCH_CHECK(A_i64.size(1) == B_i64.size(0), "matmul shape mismatch (A.cols != B.rows)");

  auto A = A_i64.contiguous();
  auto B = B_i64.contiguous();
  const int64_t m64 = A.size(0);
  const int64_t k64 = A.size(1);
  const int64_t n64 = B.size(1);
  TORCH_CHECK(m64 >= 0 && k64 >= 0 && n64 >= 0, "invalid matmul sizes");
  TORCH_CHECK(m64 <= INT32_MAX && k64 <= INT32_MAX && n64 <= INT32_MAX, "matmul sizes too large for kernel");

  auto C = torch::empty({m64, n64}, A.options());

  const dim3 block(UVCC_MATMUL_TILE, UVCC_MATMUL_TILE);
  const dim3 grid((unsigned int)((n64 + UVCC_MATMUL_TILE - 1) / UVCC_MATMUL_TILE), (unsigned int)((m64 + UVCC_MATMUL_TILE - 1) / UVCC_MATMUL_TILE));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  uvcc_matmul_u64_kernel<<<grid, block, 0, stream>>>(
      (const int64_t*)A.data_ptr<int64_t>(),
      (const int64_t*)B.data_ptr<int64_t>(),
      (int64_t*)C.data_ptr<int64_t>(),
      (int)m64,
      (int)k64,
      (int)n64);
  return C;
}


