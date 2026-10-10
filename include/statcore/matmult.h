#ifndef MATMULT
#define MATMULT

#include <armadillo>

namespace matmult {
    arma::mat ijk_mult(arma::mat A, arma::mat B);
    arma::mat ikj_mult(arma::mat A, arma::mat B);
    arma::mat manopt_mult(arma::mat A, arma::mat B);
    arma::mat sq_mat_block_mult(arma::mat A, arma::mat B, std::size_t block_size);
}

#endif