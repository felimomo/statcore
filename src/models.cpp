#include "statcore/models.h"

#include <armadillo>

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>

#include "statcore/logger.h"

// Lapack ABI
extern "C" {
  void dgesv_(
    const int* n, 
    const int* nrhs, 
    double* a, 
    const int* lda,
    int* ipiv, 
    double* b, 
    const int* ldb, 
    int* info);
}

namespace Models{

    using std::exp;
    using std::numbers::pi; 

    // utility
    inline double binom_coeff(std::size_t N, std::size_t m) {
        std::size_t n = std::max(m, N-m);
        std::size_t n_rest = N - n;
        double result = 1.0;
        for (std::size_t i = 0; i < n_rest; i++) {
            result *= ((double) (N-i)) / (n_rest-i);
        }
        return result;
    }

    // ModelBase implementation
    ModelBase::ModelBase(std::size_t dim): dim_(dim) {
        // initialize random normal:
        beta_ = arma::randn(dim_);
        beta0_ = arma::randn<double>();
    }
    void ModelBase::ingest_data(arma::mat X, arma::vec y) {
        if (y.n_rows != X.n_rows) {
            throw std::invalid_argument("ModelBase::ingest_data: Numer of rows in X and y do not match.");
        }
        if (X.n_cols != dim_) {
            throw std::invalid_argument("ModelBase::ingest_data: Numer of cols in X does not match dim.");
        }
        X_ = X;
        y_ = y;
    }
    void ModelBase::fit(
        arma::mat X, 
        arma::vec y, 
        double beta_thresh, 
        std::size_t max_iter
    ) {
        ingest_data(X, y);
        update_pars_wts(); // initialize weights and response with data

        double dB = 1.0;
        std::size_t i = 0;
        arma::vec past_beta(dim_);
        while (dB > beta_thresh){
            past_beta = beta_;
            irls_update_beta();
            update_pars_wts();
            dB = arma::abs(beta_ - past_beta).max();
            i++;
            if (i > max_iter){
                std::cout << "ModelBase::fit() warn:\n" 
                            << "exceeded max_iter with delta-beta = " 
                            << dB << "\n";
                return;
            }
        }
    
    }
    arma::vec ModelBase::get_beta(){
        return beta_;
    }
    double ModelBase::get_beta0(){
        return beta0_;
    }

    // Gaussian implementation
    Gaussian::Gaussian(
        std::size_t dim,
        double sigma
    ): ModelBase(dim), sigma_(sigma) {
        // initialize random normal:
        beta_ = arma::randn(dim_);
        beta0_ = arma::randn<double>();
    }
    void Gaussian::update_pars_wts(){
        // eta_ = X_ * beta_ + beta0_; // unnecessary
        // tr_mu_ = eta_; // identity link
        w_ = arma::ones<arma::vec>(y_.n_rows) / pow(sigma_, 2); // weights coincidentally independent of mu_ in this Gaussian case.
        z_ = y_; // response
    }
    void Gaussian::irls_update_beta(){
        // LAPACK call to solve:
        //      X.T W X beta = X.T W z
        //      
        //      notice that this equation happens in dim_ -> dim_ space.
        // Solve in-place (LAPACK style) on beta_.
        int nrhs = 1;
        int info = 0;
        int n = static_cast<int>(dim_); // for dgesv_ signature
        std::vector<int> ipiv(dim_);
        arma::mat A = X_.t() * (w_ % X_.each_col());
        arma::vec b = X_.t() * (w_ % z_);

        dgesv_(&n, &nrhs, A.memptr(), &n, ipiv.data(), b.memptr(), &n, &info);

        if (info > 0) {
            std::cout << "dgesv_ info = " << info;
            throw std::runtime_error(
                "Gaussian::irls_update_beta():\n Matrix at dgesv_ solution step singular"
            );
        }
        if (info < 0) {
            std::cout << "dgesv_ info = " << info;
            throw std::runtime_error(
                "Illegal argument."
            );
        }
        beta_ = b;
    }
    double Gaussian::log_likelihood(arma::mat X_obs, arma::vec y_obs) const {
        return (
            arma::accu(
                arma::square(y_obs - X_obs * beta_ - beta0_) 
            ) / (2 * pow(sigma_, 2))
            - 0.5 * y_obs.n_rows * log(2 * pi * pow(sigma_, 2))
        );
    }
    double Gaussian::log_likelihood(arma::rowvec x_obs, double y_obs) const {
        return (
            pow(
                y_obs - arma::as_scalar(x_obs * beta_) - beta0_, 
                2
            ) / (
                2 * pow(sigma_, 2)
            )
            - 0.5 * 1 * log(2 * pi * pow(sigma_, 2))
        );
    }

    // BinomLogit implementation
    BinomLogit::BinomLogit(
        std::size_t dim
    ) : ModelBase(dim) {
        //random init
        beta_ = arma::randn(dim_);
        beta0_ = arma::randn<double>();
    }
    void BinomLogit::update_pars_wts() {
        eta_ = X_ * beta_ + beta0_;
        eta_ = arma::clamp(eta_, -eta_max_, eta_max_);
        p_ = 1.0 / (1.0 + arma::exp(-eta_));
        w_ = p_ % (1.0 - p_);
        z_ = eta_ + (y_ - p_) % (1.0 / (p_ % (1.0 - p_))); // response. denominator is dp / deta
    }
    void BinomLogit::irls_update_beta() {
        int nrhs = 1;
        int info = 0;
        int n = static_cast<int>(dim_); // for dgesv_ signature
        std::vector<int> ipiv(dim_);
        arma::mat A = X_.t() * (w_ % X_.each_col());
        arma::vec b = X_.t() * (w_ % z_);

        dgesv_(&n, &nrhs, A.memptr(), &n, ipiv.data(), b.memptr(), &n, &info);

        if (info > 0) {
            std::cout << "dgesv_ info = " << info;
            throw std::runtime_error(
                "Gaussian::irls_update_beta():\n Matrix at dgesv_ solution step singular"
            );
        }
        if (info < 0) {
            std::cout << "dgesv_ info = " << info;
            throw std::runtime_error(
                "Illegal argument."
            );
        }
        beta_ = b;
    }
    double BinomLogit::log_likelihood(arma::mat X_obs, arma::uvec y_obs) const {
        arma::vec p = 1.0 / (1.0 + arma::exp(- X_obs * beta_ - beta0_));
        std::size_t n = y_obs.n_rows;
        double log_binom = (
            std::lgamma(n + 1) 
            - std::lgamma(arma::accu(y_obs) + 1)
            - std::lgamma(n - arma::accu(y_obs) + 1)
        );
        return arma::accu(
            y_obs % arma::log(p) - (1 - y_obs) % arma::log(1 - p)
        ) + log_binom;
    }
    double BinomLogit::log_likelihood(arma::rowvec x_obs, std::size_t y_obs) const {
        double log_binom = std::lgamma(1 + 1) - std::lgamma(y_obs + 1) - std::lgamma(1 - y_obs + 1);
        double p = 1.0 / (1.0 + exp(
            - arma::as_scalar(x_obs * beta_) - beta0_
        ));
        return y_obs * log(p) - (1 - y_obs) * log(1 - p);
    }

    // PoissonLog implementation
    PoissonLog::PoissonLog(std::size_t dim) : ModelBase(dim) {
        //random init
        beta_ = arma::randn(dim_);
        beta0_ = arma::randn<double>();
    }

    void PoissonLog::update_pars_wts() {
        eta_ = X_ * beta_ + beta0_;
        eta_ = arma::clamp(eta_, -eta_max_, eta_max_);
        lambda_ = arma::exp(eta_);
        w_ = lambda_; // dlamda / deta = lambda. => w = lambda^2 / lambda <-- variance
        z_ = eta_ + (y_ - lambda_) % (1.0 / lambda_); // response, denominator is dlambda / deta
    }
    void PoissonLog::irls_update_beta() {
        int nrhs = 1;
        int info = 0;
        int n = static_cast<int>(dim_); // for dgesv_ signature
        std::vector<int> ipiv(dim_);
        arma::mat A = X_.t() * (w_ % X_.each_col());
        arma::vec b = X_.t() * (w_ % z_);

        dgesv_(&n, &nrhs, A.memptr(), &n, ipiv.data(), b.memptr(), &n, &info);

        if (info > 0) {
            std::cout << "dgesv_ info = " << info;
            throw std::runtime_error(
                "Gaussian::irls_update_beta():\n Matrix at dgesv_ solution step singular"
            );
        }
        if (info < 0) {
            std::cout << "dgesv_ info = " << info;
            throw std::runtime_error(
                "Illegal argument."
            );
        }
        beta_ = b;
    }
    double PoissonLog::log_likelihood(arma::rowvec x_obs, std::size_t y_obs) const {
        double lambda = exp(arma::as_scalar(x_obs * beta_) + beta0_);
        // avoid log(exp( <loglambda> )) and simply use <loglambda>.
        return (
            y_obs * (
                arma::as_scalar(x_obs * beta_) + beta0_
            ) 
            - lambda - std::lgamma(y_obs+1)
        );
    }
    double PoissonLog::log_likelihood(arma::mat X_obs, arma::uvec y_obs) const {
        arma::vec lambda = arma::exp(X_obs * beta_ + beta0_);
        // avoid log(exp( <loglambda> )) and simply use <loglambda>.
        return (
            arma::accu(
                y_obs % (X_obs * beta_ + beta0_) - lambda
            ) 
            - arma::accu(y_obs.transform(
                [] (double val) -> double {return std::lgamma(val + 1); }
            ))
        );
    }
}