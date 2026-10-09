#include <armadillo>

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

