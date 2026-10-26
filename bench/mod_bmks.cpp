#include <chron>
#include <armadillo>

#include "statcore/models.h"
#include "statcore/logger.h"

using std::size_t;
using std::chrono::steady_clock;
using Logging::ErrorMetrics;
using Models::FitOptions;
using Models::ModelBase;

arma::vec gen_targets(size_t dim){
    u = arma::randn(dim);
    return u / arma::norm(u);
}

arma::vec gen_y_gauss(const arma::mat& X, const arma::vec& beta, double beta0, double sigma) {
    return (X * beta + beta0) + arma::randn<arma::vec>(X.n_rows) * sigma;
}

arma::vec gen_y_binom(const arma::mat& X, const arma::vec& beta, double beta0) {
    arma::vec p = 1.0 / (1.0 + arma::exp(- X * beta_targ - beta0_targ));
    arma::vec rand_samp = arma::randu<arma::vec>(N);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; i++) {
        if (rand_samp[i] < p[i]) { y[i] = 1.0; } else { y[i] = 0.0; }
    }
    return y;
}

arma::vec gen_y_pois(const arma::mat& X, const arma::vec& beta, double beta0) {
    // dependent variables
    std::mt19937 rng(std::random_device{}()); // randomized seed
    size_t N = X.n_rows;
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double lambda = std::exp(arma::as_scalar(X.row(i) * beta) + beta0);
        std::poisson_distribution<int> pois(lambda);
        y(i) = pois(rng);
    }
    return y;
}

struct ModelBenchmarks {
    double runtime;
    ErrorMetrics error_metrics;
};

ModelBenchmarks benchmark_model(
    const ModelBase& mod, 
    const arma::mat& X, 
    const arma::vec& y,
    const arma::vec& targ_params
) {
    mod.inget_data(X,y);
    mod.beta_ = arma::zeros<arma::vec>(mod.beta_.size());
    mod.beta0_ = 0.0;
    //
    start = steady_clock::now();
    infer_params = fit_loop(X, y, {}); // {} uses default FitOptions variables.
    runtime = steady_clock::now() - start;
    //
    return ModelBenchmarks{
        runtime, 
        ErrorMetrics{
            y.n_rows,
            infer_params, 
            targ_params
        }
    };
}


int main () {
    size_t N_iter = 25;

    size_t dim = 20;
    size_t N = dim * 10000; // tbd: loop over dimensions

    // Gaussian
    gaussian_bmk = std::vector<ModelBenchmarks>(N_iter)
    #pragma omp parallel for
    for(size_t i = 0; i < N_iter; i++) {
        // 1. sample data
        arma::vec beta = get_targets(dim);
        double beta0 = 0.0;
        double sigma = 0.05;
        arma::mat X = arma::randn<arma::mat>(N, dim);
        arma::vec y = gen_y_gauss(const arma::mat& X, const arma::vec& beta, double beta0, double sigma);
        
        // 2. initialize model
        mod = Gaussian(dim, sigma);
        
        // 3. benchmark fit
        gaussian_bmk[i] = benchmark_model(mod, X, y, targ_params);
    }
}