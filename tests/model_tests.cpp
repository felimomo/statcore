#include <cmath>
#include <random>
#include <armadillo>

#include "statcore/models.h"
#include "statcore/logger.h"

void test_gauss(){
    // constants
    std::size_t dim = 40; // dim of predictor space
    std::size_t N = 1000;  // n data rows
    arma::vec beta_targ = arma::ones(dim);
    double beta0_targ = 0.0;
    double sigma = 0.05;

    // data generation
    arma::mat X(N, dim, arma::fill::randn);
    arma::vec y(N);
    for (std::size_t i = 0; i<N; i++) {
        double mu = arma::as_scalar(X.row(i) * beta_targ) + beta0_targ;
        y[i] = mu + arma::randn<double>() * sigma;
    }

    // logging
    logger = Logger::ModelLogger();

    // fitting
    Models::Gaussian GM(dim, sigma);
    GM.fit(X, y, &logger);
    

    // printing
    std::cout << "dim = " << dim << ", N = " << N <<"\n";
    std::cout << "TARGETS:\n";
    std::cout << "beta  = " << beta_targ
              << "beta0 = " << beta0_targ << "\n";
    std::cout << "INFERRED:\n";
    std::cout << "beta  = " << GM.get_beta()
              << "beta0 = " << GM.get_beta0() << "\n";
}

int main() {
    test_gauss();
}