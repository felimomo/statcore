#ifndef MATMULT
#define MATMULT

#include <armadillo>

namespace matmult {
    arma::mat ijk_mult(const arma::mat& A, const arma::mat& B);
    arma::mat ikj_mult(const arma::mat& A, const arma::mat& B);
    arma::mat jki_mult(const arma::mat& A, const arma::mat& B);
    arma::mat manopt_mult(const arma::mat& A, const arma::mat& B);
    arma::mat sq_mat_block_mult(const arma::mat& A, const arma::mat& B, std::size_t block_size);
    arma::mat sq_mat_simd_block_mult(const arma::mat& A, const arma::mat& B, std::size_t block_size);
}

#endif