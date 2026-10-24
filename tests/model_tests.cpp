#include <cmath>
#include <random>
#include <armadillo>

#include "statcore/models.h"
#include "statcore/logger.h"

void test_gauss(){
    // constants
    std::size_t dim = 500; // dim of predictor space
    std::size_t N = 20000;  // n data rows
    arma::vec beta_targ = arma::randn(dim);
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
    Logging::ModelLogger logger(".", "gaussian_log");
    Models::FitOptions fit_options;
    fit_options.logger = &logger;
    
    // fitting
    Models::Gaussian GM(dim, sigma);
    arma::vec infer_params = GM.fit(X, y, fit_options);

    // log writing
    logger.summary(infer_params, arma::join_cols(beta_targ, arma::vec{beta0_targ}));
    logger.write();
    std::cout << "dim = " << dim << ", N = " << N <<"\n";
    std::cout << "Log locations:\n" 
              << logger.get_step_csv() << "\n" 
              << logger.get_summary_csv() << "\n";
}

void test_binom() {
    // constants
    std::size_t dim = 500; // dim of predictor space
    std::size_t N = 20000;  // n data rows
    arma::vec beta_targ = arma::randn(dim);
    double beta0_targ = 0.0;

    // data generation
    arma::mat X(N, dim, arma::fill::randn);
    arma::vec p = 1.0 / (1.0 + arma::exp(- X * beta_targ - beta0_targ));
    arma::vec rand_samp = arma::randu<arma::vec>(N);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; i++) {
        if (rand_samp[i] < p[i]) { y[i] = 1.0; } else { y[i] = 0.0; }
    }

    // logging
    Logging::ModelLogger logger(".", "binom_log");
    Models::FitOptions fit_options;
    fit_options.logger = &logger;

    // fitting
    Models::BinomLogit BL(dim);
    arma::vec infer_params = BL.fit(X, y, fit_options);

    // log writing
    logger.summary(infer_params, arma::join_cols(beta_targ, arma::vec{beta0_targ}));
    logger.write();
    std::cout << "dim = " << dim << ", N = " << N <<"\n";
    std::cout << "Log locations:\n" 
              << logger.get_step_csv() << "\n" 
              << logger.get_summary_csv() << "\n";
}

int main() {
    test_gauss();
    test_binom();
}