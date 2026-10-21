// #define EIGEN_DEBUG_ASSIGN // print out assignment decisions

#include <iostream>
#include <vector>

#include <armadillo>
#include <Eigen/Dense>

#include "statcore/agg.h"
#include "statcore/matmult.h"

void basic_test_sums() {
    arma::vec rv = arma::randn<arma::vec>(1024);
    double arma_total = arma::accu(rv);

    std::vector<double> std_rv = arma::conv_to<std::vector<double>>::from(rv);
    double kahn_total = agg::KahanSum(std_rv);
    double pair_total = agg::pairwiseSum(std_rv);

    std::cout << "Armadillo total: " << arma_total 
              << "\nKahn total: " << kahn_total 
              << "\nPairwise total: " << pair_total;
}

void mat_test() {
    std::size_t d = 1024;
    arma::mat A = arma::randn<arma::mat>(d, d);
    arma::mat B = arma::randn<arma::mat>(d, d);
    std::size_t block_size = 8;
    arma::mat C = matmult::sq_mat_simd_block_mult(A,B,block_size);
    arma::mat diff = A * B - C;
    diff.submat(0,0,8,8).print("A x B = ");
}

void eigen_test() {
    using Eigen::MatrixXd;
    MatrixXd A = MatrixXd::Random(5,2);
    MatrixXd B = MatrixXd::Random(2,5);
    MatrixXd C = MatrixXd::Random(5,5);
    // auto expr = A * B + C;
    std::cout << "A * B + C = \n" << A * B + C << "\n";
}

inline double binom_coeff(std::size_t N, std::size_t m) {
    std::size_t n = std::max(m, N-m);
    std::size_t n_rest = N - n;
    double result = 1.0;
    for (std::size_t i = 0; i < n_rest; i++) {
        result *= ((double) (N-i)) / (n_rest-i);
    }
    return result;
}

void binom_coeff_test() {
    for (int N = 3; N < 10; N++) {
        for (int m = 1; m <= std::ceil((float) N / 2); m++) {
            double result = binom_coeff(N, m);
            std::cout << "Bin(N=" << N << ", m=" << m << ") = " << result << "\n";
        }
    }
}

void arma_test() {
    arma::mat A = arma::mat(5,5, arma::fill::ones);
    arma::vec v = arma::randn(5);
    std::cout << "v = \n" << v << "\n";
    // for(arma::uword i=0; i<A.n_cols; i++){
    //     A.col(i) *= v[i];
    // }
    //equivalent:
    arma::mat B = v % A.each_col(); // % is entrywise product
    std::cout << "diag(v) * A = \n" << B << "\n";
}

int main() {
    // basic_test_sums();
    // mat_test();
    // eigen_test();
    // binom_coeff_test();
    arma_test();
}