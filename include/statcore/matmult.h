#ifndef MATMULT
#define MATMULT

#include <algorithm>
#include <armadillo>

namespace matmult {
    arma::mat ijk_mult(const arma::mat& A, const arma::mat& B);
    arma::mat ikj_mult(const arma::mat& A, const arma::mat& B);
    inline arma::mat jki_mult(const arma::mat& A, const arma::mat& B);
    inline arma::mat simd_jki_mult(const arma::mat& A, const arma::mat& B);
    arma::mat manopt_mult(const arma::mat& A, const arma::mat& B);
    arma::mat sq_mat_block_mult(const arma::mat& A, const arma::mat& B, std::size_t block_size);
    arma::mat sq_mat_simd_block_mult(const arma::mat& A, const arma::mat& B, std::size_t block_size);
    //
    // void versions for optimal performance
    inline void simd_jki_mult(
        const double* A_block, 
        const double* B_block, 
        double* C_block,            //writes directly on C_block
        std::size_t N,              // full matrix size
        std::size_t target_n_rows,
        std::size_t target_n_cols,
        std::size_t contracted_dim
    );
    inline void jki_mult(
        const double* A_block, 
        const double* B_block, 
        double* C_block,            //writes directly on C_block
        std::size_t N,              // full matrix size
        std::size_t target_n_rows,
        std::size_t target_n_cols,
        std::size_t contracted_dim
    );
    arma::mat tiled_sq_mult(
        const arma::mat& A, 
        const arma::mat& B, 
        std::size_t C_block_size,
        std::size_t contracted_dim,
        bool simd
    );

    // Claude's suggestions:
    inline void micro_4x4(const double* A, const double* B, double* C,
                      std::size_t N, std::size_t kd);

    void reg_blocked_mult(const double* A, const double* B, double* C, std::size_t N,
                      std::size_t m, std::size_t n, std::size_t kd);

    // --- driver (your tiled_sq_mult, with the kernel as a template parameter) ----
    template <auto Kernel>
    arma::mat tiled_sq_mult_claude(const arma::mat& A, const arma::mat& B,
                            std::size_t bs = 64, std::size_t kb = 64) {
        const std::size_t N = A.n_cols;
        arma::mat C(N, N, arma::fill::zeros);
        for (std::size_t i0 = 0; i0 < N; i0 += bs) {
            const std::size_t m = std::min(bs, N - i0);
            for (std::size_t k0 = 0; k0 < N; k0 += kb) {
                const std::size_t kd = std::min(kb, N - k0);
                for (std::size_t j0 = 0; j0 < N; j0 += bs) {
                    const std::size_t n = std::min(bs, N - j0);
                    Kernel(A.memptr() + i0 + k0 * N, B.memptr() + k0 + j0 * N,
                        C.memptr() + i0 + j0 * N, N, m, n, kd);
                }
            }
        }
        return C;
    }

}

#endif