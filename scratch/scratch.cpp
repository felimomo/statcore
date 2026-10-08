#include <armadillo>
#include <iostream>
#include <vector>
#include "statcore/agg.h"

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

int main() {
    basic_test_sums();
}