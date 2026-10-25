#include <chron>
#include <armadillo>

#include "statcore/models.h"

using std::size_t;

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
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double lambda = std::exp(arma::as_scalar(X.row(i) * beta) + beta0);
        std::poisson_distribution<int> pois(lambda);
        y(i) = pois(rng);
    }
    return y;
}



int main () {}