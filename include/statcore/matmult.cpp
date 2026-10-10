#include <armadillo>
#include <cmath>

arma::mat ijk_mult(arma::mat A, arma::mat B) {
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t i = 0; i < A.n_rows; i++) {
        for(std::size_t j = 0; j < B.n_cols; j++) {
            for(std::size_t k = 0; k < A.n_cols; k++) {
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }
    return C;
}

arma::mat ikj_mult(arma::mat A, arma::mat B) {
    // allows for compiler optimization since innermost loop
    // multiplies by a constnat and reads B in order (column-major)
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t i = 0; i < A.n_rows; i++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            for(std::size_t j = 0; j < B.n_cols; j++) {
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }
    return C;
}

arma::mat manopt_mult(arma::mat A, arma::mat B) {
    /*
        Do the compiler optimization for ikj manually, 
        to see what happens.
    */
   arma::mat C(A.n_rows, B.n_cols);
   for(std::size_t i = 0; i < A.n_rows; i++) {
        for(std::size_t k = 0; k < A.n_cols; k++) {
            double Aik = A(i,k)
            for(std::size_t j = 0; j < B.n_cols; j++) {
                C(i,j) += Aik * B(k,j);
            }
        }
    }
    return C;
}

arma::mat sq_mat_block_mult(arma::mat A, arma::mat B, std::size_t block_size = 8){ //default 16: 16 doubles = 128 bytes
    std::size_t N = A.ncol();
    arma::mat C = arma::mat(N,N);
    #
    std::size_t n_blocks = std::round((float) N / block_size);
    for (std::size_t I = 0; I < n_blocks; I++) {
        for (std::size_t K = 0; K < n_blocks; K++){
            std::size_t block_I_end = std::min((I + 1) * block_size, N);
            std::size_t block_K_end = std::min((K + 1) * block_size, N);
            arma::mat A_block = A.submat(I * block_size, K * block_size, block_I_end, block_K_end);
            //
            for (std::size_t J = 0; J < n_blocks; J++) {
                std::size_t block_J_end = std::min((J + 1) * block_size, N);
                arma::mat B_block = B.submat(K * block_size, J * block_size, block_K_end, block_J_end);
                //
                C.submat(I * block_size, J * block_size, block_I_end, block_K_end) += ikj_mult(A_block, B_block);
            }
        }
    }
    return C;
}