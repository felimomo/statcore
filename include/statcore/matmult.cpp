#include <armadillo>

arma::mat ijk_mult(arma::mat A, arma::mat B) {
    arma::mat C(A.n_rows, B.n_cols);
    for(std::size_t i = 0; i < A.n_rows; i++) {
        for(std::size_t j = 0; j < A.n_cols; j++) {
            for(std::size_t k = 0; k < B.n_cols; k++) {
                C(i,j) += A(i,j) * B(j,k);
            }
        }
    }
}