#include <armadillo>
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

arma::mat manopt_mult(const arma::mat& A, const arma::mat& B) {
    /*
        Do the compiler optimization for jki manually, 
        to see what happens.
    */
   arma::mat C(A.n_rows, B.n_cols);
   for(std::size_t j = 0; j < B.n_cols; j++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            double Bkj = B.at(k,j);
                for(std::size_t i = 0; i < A.n_rows; i++) {
                C(i,j) += A.at(i,k) * Bkj; // we loop over column k of i -> cache-friendly
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
    std::size_t n_blocks = std::round((float) N / block_size);
    for (std::size_t I = 0; I < n_blocks; I++) {
        for (std::size_t K = 0; K < n_blocks; K++){
            std::size_t block_I_end = std::min((I + 1) * block_size, N) - 1; // submat is inclusive on both ends
            std::size_t block_K_end = std::min((K + 1) * block_size, N) - 1;
            arma::mat A_block = A.submat(I * block_size, K * block_size, block_I_end, block_K_end);
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

} // namespace matmult