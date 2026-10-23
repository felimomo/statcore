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
    Logging::ModelLogger logger {}; // initialize logger object with default arguments

    // fitting
    Models::Gaussian GM(dim, sigma);
    arma::vec infer_params = GM.fit(X, y, &logger);

    // log writing
    logger.summary();
    logger.write();
    std::cout << "dim = " << dim << ", N = " << N <<"\n";
    std::cout << "Log locations:\n" 
              << logger.step_csv_ << "\n" 
              << logger.summary_csv_ << "\n";
}

int main() {
    test_gauss();
}