// gemv_q4.cu — Q4_K dequant-fused GEMV for Orin SM 8.7
//
// Matches llama.cpp's exact Q4_K block layout and dequant formula.
// Reference: ggml-cuda/convert.cu dequantize_block_q4_K
//
// Block layout (256 elements, 144 bytes):
//   dm:        half2 (4 bytes) — dm.x = d (super-block scale), dm.y = dmin
//   scales[12]: uint8 — packed 6-bit sub-block scales and mins
//   qs[128]:    uint8 — 256 × 4-bit quants (lower nibble = offset 0, upper = offset +32)
//
// Dequant: val = d * sc * (q & 0xF) - dmin * m       (lower nibble, first 32 of 64-group)
//          val = d * sc2 * (q >> 4) - dmin * m2       (upper nibble, second 32 of 64-group)

#include "jllm_kernels.h"
#include <cuda_fp16.h>

namespace jllm {

static constexpr int QK_K = 256;

// Matches ggml's block_q4_K exactly
struct block_q4_K {
    half2   dm;           // dm.x = d, dm.y = dmin
    uint8_t scales[12];   // packed 6-bit scales and mins
    uint8_t qs[QK_K/2];  // 4-bit quants (128 bytes)
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");

// Extract 6-bit scale and min — matches llama.cpp get_scale_min_k4
__device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d, uint8_t& m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// GEMV: y[M] = W[M × K] (Q4_K) × x[K] (FP16)
// One warp per output row, lanes stride across Q4_K blocks
__global__ void gemv_q4k_kernel(
    half*              __restrict__ y,
    const block_q4_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K)
{
    const int row  = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q4_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc = 0.0f;

    for (int b = lane; b < n_blocks; b += 32) {
        const block_q4_K& blk = row_blocks[b];

        const float dall = __low2float(blk.dm);
        const float dmin = __high2float(blk.dm);

        int k_base = b * QK_K;

        // Process 4 groups of 64 elements (il=0..3)
        for (int il = 0; il < 4; il++) {
            int is = 2 * il;  // scale index pair

            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            float d1 = dall * sc1;
            float dm1 = dmin * m1;
            float d2 = dall * sc2;
            float dm2 = dmin * m2;

            // Each group of 64: first 32 from lower nibble, second 32 from upper nibble
            const uint8_t* q = blk.qs + 32 * il;

            for (int l = 0; l < 32; l++) {
                int k_lo = k_base + 64 * il + l;       // lower nibble position
                int k_hi = k_base + 64 * il + l + 32;  // upper nibble position

                float w_lo = d1 * (q[l] & 0xF) - dm1;
                float w_hi = d2 * (q[l] >> 4)  - dm2;

                acc += w_lo * __half2float(x[k_lo]);
                acc += w_hi * __half2float(x[k_hi]);
            }
        }
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        y[row] = __float2half(acc);
}

void gemv_q4(half* y, const void* W_q4, const half* scales, const half* x,
             int M, int K, int group_size, cudaStream_t stream) {
    dim3 grid((M + 3) / 4);
    dim3 block(128);
    gemv_q4k_kernel<<<grid, block, 0, stream>>>(
        y, (const block_q4_K*)W_q4, x, M, K);
}

}  // namespace jllm
