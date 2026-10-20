#include <armadillo>
#include <algorithm>
#include <cmath>
#include "statcore/matmult.h"

namespace matmult{

arma::mat ijk_mult(const arma::mat& A, const arma::mat& B) {
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t i = 0; i < A.n_rows; i++) {
        for(std::size_t j = 0; j < B.n_cols; j++) {
            for(std::size_t k = 0; k < A.n_cols; k++) {
                C.at(i,j) += A.at(i,k) * B.at(k,j);     //.at() for efficient access (no bound check)
            }
        }
    }
    return C;
}

arma::mat ikj_mult(const arma::mat& A, const arma::mat& B) {
    // allows for compiler optimization since innermost loop
    // multiplies by a constnat and reads B in order (column-major)
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t i = 0; i < A.n_rows; i++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            for(std::size_t j = 0; j < B.n_cols; j++) {
                C.at(i,j) += A.at(i,k) * B.at(k,j);
            }
        }
    }
    return C;
}

arma::mat jki_mult(const arma::mat& A, const arma::mat& B) {
    // like ikj but geared towards column-major layouts (armadillo)
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t j = 0; j < B.n_cols; j++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            for(std::size_t i = 0; i < A.n_rows; i++) {
                C.at(i,j) += A.at(i,k) * B.at(k,j);
            }
        }
    }
    return C;
}

inline arma::mat simd_jki_mult(const arma::mat& A, const arma::mat& B) {
    // Like jki but using simd operations
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t j = 0; j < B.n_cols; j++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            const double Bkj = B.at(k,j);
            const double* A_k_col = A.colptr(k);
            double* C_j_col = C.colptr(j);
            #pragma omp simd
            for(std::size_t i = 0; i < A.n_rows; i++) {
                // C.at(i,j) += A.at(i,k) * Bkj;
                C_j_col[i] += A_k_col[i] * Bkj;
            }
        }
    }
    return C;
}

arma::mat manopt_mult(const arma::mat& A, const arma::mat& B) {
    /*
        Do the compiler optimization for jki manually, 
        to see what happens.
    */
   arma::mat C(A.n_rows, B.n_cols);
   for(std::size_t j = 0; j < B.n_cols; j++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            const double Bkj = B.at(k,j);
                for(std::size_t i = 0; i < A.n_rows; i++) {
                C.at(i,j) += A.at(i,k) * Bkj; // we loop over column k of i -> cache-friendly
            }
        }
    }
    return C;
}

arma::mat sq_mat_block_mult(
    const arma::mat& A, 
    const arma::mat& B, 
    std::size_t block_size = 128 // (double: 8 bytes) * 128 * 128 = 128 KB (Apple M2) 
) { 
    std::size_t N = A.n_cols;
    arma::mat C = arma::mat(N,N, arma::fill::zeros);
    //
    std::size_t n_blocks = std::ceil((float) N / block_size);
    for (std::size_t I = 0; I < n_blocks; I++) {
        for (std::size_t K = 0; K < n_blocks; K++){
            std::size_t block_I_end = std::min((I + 1) * block_size, N) - 1; // submat is inclusive on both ends
            std::size_t block_K_end = std::min((K + 1) * block_size, N) - 1;
            const arma::mat A_block = A.submat(I * block_size, K * block_size, block_I_end, block_K_end);
            //
            for (std::size_t J = 0; J < n_blocks; J++) {
                std::size_t block_J_end = std::min((J + 1) * block_size, N) - 1;
                arma::mat B_block = B.submat(K * block_size, J * block_size, block_K_end, block_J_end);
                //
                C.submat(I * block_size, J * block_size, block_I_end, block_J_end) += jki_mult(A_block, B_block);
            }
        }
    }
    return C;
}

void jki_mult(
    const double* A_block, 
    const double* B_block, 
    double* C_block,            //writes directly on C_block
    std::size_t N,              // full matrix size
    std::size_t target_n_rows,
    std::size_t target_n_cols,
    std::size_t contracted_dim
) {
    // A [target_n_rows x contracted_dim]
    // B [contracted_dim x target_n_cols]
    // C = A * B [target_n_rows x target_n_cols]
    for(std::size_t j = 0; j < target_n_cols; j++) {
        for(std::size_t k = 0; k < contracted_dim; k++) {
            for(std::size_t i = 0; i < target_n_rows; i++) {
                C_block[i + j * N] += A_block[i + k * N] * B_block[k + j * N];
                // compare to:
                // C.at(i,j) += A.at(i,k) * B.at(k,j);
                // (i,j) -> j-th col; i-th elem -> i + (j * N) displacement = j full-mat columns + i elements in col 
            }
        }
    }
}

void simd_jki_mult(
    const double* A_block, 
    const double* B_block, 
    double* C_block,            //writes directly on C_block
    std::size_t N,              // full matrix size
    std::size_t target_n_rows,
    std::size_t target_n_cols,
    std::size_t contracted_dim
) {
    // A [target_n_rows x contracted_dim]
    // B [contracted_dim x target_n_cols]
    // C = A * B [target_n_rows x target_n_cols]
    for(std::size_t j = 0; j < target_n_cols; j++) {
        for(std::size_t k = 0; k < contracted_dim; k++) {
            const double b = B_block[k + j * N];
            #pragma omp simd
            for(std::size_t i = 0; i < target_n_rows; i++) {
                C_block[i + j * N] += A_block[i + k * N] * b;
                // compare to:
                // C.at(i,j) += A.at(i,k) * B.at(k,j);
                // (i,j) -> j-th col; i-th elem -> i + (j * N) displacement = j full-mat columns + i elements in col 
            }
        }
    }
}

arma::mat tiled_sq_mult(
    const arma::mat& A, 
    const arma::mat& B, 
    std::size_t C_block_size = 64,
    std::size_t contracted_dim = 64,
    bool simd = true
) { 
    std::size_t N = A.n_cols;
    arma::mat C = arma::mat(N,N, arma::fill::zeros);
    //
    std::size_t n_IJ_blocks = std::ceil((float) N / C_block_size);
    std::size_t n_K_blocks = std::ceil((float) N / contracted_dim);
    for (std::size_t I = 0; I < n_IJ_blocks; I++) {
        
        // edge cases:
        std::size_t I_size = std::min(C_block_size,   N - I * C_block_size);

        for (std::size_t K = 0; K < n_K_blocks; K++){
            // edge cases
            std::size_t K_size = std::min(contracted_dim, N - K * contracted_dim);

            // block
            const double* A_block_ptr = A.memptr() + (I * C_block_size + K * N * contracted_dim);

            // std::size_t block_I_end = std::min((I + 1) * block_size, N) - 1; // submat is inclusive on both ends
            // std::size_t block_K_end = std::min((K + 1) * block_size, N) - 1;
            // const arma::mat A_block = A.submat(I * block_size, K * block_size, block_I_end, block_K_end);
            //
            for (std::size_t J = 0; J < n_IJ_blocks; J++) {
                // edge cases
                std::size_t J_size = std::min(C_block_size,   N - J * C_block_size);

                // blocks
                const double* B_block_ptr = B.memptr() + (K * contracted_dim + J * N * C_block_size);
                double* C_block_ptr = C.memptr() + (I * C_block_size + J * N * C_block_size);
                if (simd) {
                    simd_jki_mult(
                        A_block_ptr,
                        B_block_ptr,
                        C_block_ptr,
                        N,
                        I_size,
                        J_size,
                        K_size
                    );
                } else {
                    jki_mult(
                        A_block_ptr,
                        B_block_ptr,
                        C_block_ptr,
                        N,
                        I_size,
                        J_size,
                        K_size
                    );
                }
            }
        }
    }
    return C;
}

arma::mat sq_mat_simd_block_mult(
    const arma::mat& A, 
    const arma::mat& B, 
    std::size_t block_size = 128 // (double: 8 bytes) * 128 * 128 = 128 KB (Apple M2) 
) { 
    std::size_t N = A.n_cols;
    arma::mat C = arma::mat(N,N, arma::fill::zeros);
    //
    std::size_t n_blocks = std::ceil((float) N / block_size);
    for (std::size_t I = 0; I < n_blocks; I++) {
        for (std::size_t K = 0; K < n_blocks; K++){
            //
            std::size_t block_I_end = std::min((I + 1) * block_size, N) - 1; // submat is inclusive on both ends
            std::size_t block_K_end = std::min((K + 1) * block_size, N) - 1;
            const arma::mat A_block = A.submat(I * block_size, K * block_size, block_I_end, block_K_end);
            //
            // #pragma omp simd
            {
                for (std::size_t J = 0; J < n_blocks; J++) {
                    std::size_t block_J_end = std::min((J + 1) * block_size, N) - 1;
                    arma::mat B_block = B.submat(K * block_size, J * block_size, block_K_end, block_J_end);
                    //
                    C.submat(I * block_size, J * block_size, block_I_end, block_J_end) += simd_jki_mult(A_block, B_block);
                }
            }
        }
    }
    return C;
}

// Claude Code's functions (pure vibe-coding, I wanna see how these compare with the rest):

// --- the micro-kernel --------------------------------------------------------
// C[0:4, 0:4] += A[0:4, 0:kd] * B[0:kd, 0:4]
//
// The 4x4 block of C lives in the local array c for the whole k loop. Because
// its size is a compile-time constant, the compiler unrolls the i/j loops and
// keeps all 16 values in registers (8 NEON/SSE registers of 2 doubles each).
// C in memory is touched only once at the start and once at the end.
//
// Per k: 4 loads of A (contiguous: 2 vector loads) + 4 loads of B (one per
// column), then 16 multiply-adds that use only registers.
inline void micro_4x4(const double* A, const double* B, double* C,
                      std::size_t N, std::size_t kd) {
    double c[4][4] = {};                       // c[j][i] = C(i, j), in registers
    for (std::size_t k = 0; k < kd; k++) {
        const double* a = A + k * N;           // column k of A: a[0..3]
        for (int j = 0; j < 4; j++) {
            const double b = B[k + j * N];     // B(k, j), reused 4 times
            for (int i = 0; i < 4; i++)
                c[j][i] += a[i] * b;           // a[i] reused 4 times (once per j)
        }
    }
    for (int j = 0; j < 4; j++)                // write the block back once
        for (int i = 0; i < 4; i++)
            C[i + j * N] += c[j][i];
}

// --- tile kernel built from micro-kernels -----------------------------------
// Same signature as jki_mult, so tiled_sq_mult can call either one.
// Covers the tile with 4x4 blocks; rows/cols left over when m or n is not a
// multiple of 4 go through plain jki_mult.
void reg_blocked_mult(const double* A, const double* B, double* C, std::size_t N,
                      std::size_t m, std::size_t n, std::size_t kd) {
    const std::size_t m4 = m - m % 4, n4 = n - n % 4;
    for (std::size_t j = 0; j < n4; j += 4)
        for (std::size_t i = 0; i < m4; i += 4)
            micro_4x4(A + i, B + j * N, C + i + j * N, N, kd);
    if (m4 < m)  // bottom strip: rows m4..m, columns 0..n4
        jki_mult(A + m4, B, C + m4, N, m - m4, n4, kd);
    if (n4 < n)  // right strip: all rows, columns n4..n
        jki_mult(A, B + n4 * N, C + n4 * N, N, m, n - n4, kd);
}

} // namespace matmult