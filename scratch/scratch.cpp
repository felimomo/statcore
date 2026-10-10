#include <armadillo>
#include <iostream>
#include <vector>
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

int main() {
    // basic_test_sums();
    mat_test();
}