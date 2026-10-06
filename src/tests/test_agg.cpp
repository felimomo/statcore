#include <armadillo>
#include <iostream>
#include <vector>
#include "../agg.h"

void test_sums() {
    arma::vec rv = arma::randn<arma::vec>(1024);
    arma_total = arma::accu(rv);

    std::vector<double> std_rv = arma::conv_to<std::vector<double>>::from(rv);
    kahn_total = agg::KahanSum(std_rv);
    pair_total = agg::pairwiseSum(std_rv);

    std::cout << "Armadillo total: " << arma_total 
              << "\nKahn total: " << kahn_total 
              << "\nPairwise total: " << pair_total;
}