#include "statcore/models.h"

#include <Armadillo>

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>

#include "statcore/agg.h"

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

    inline double binom_coeff(std::size_t N, std::size_t m) {
        std::size_t n = std::max(m, N-m);
        std::size_t n_rest = N - n;
        double result = 1.0;
        for (std::size_t i = 0; i < n_rest; i++) {
            result *= ((double) (N-i)) / (n_rest-i);
        }
        return result;
    }

    class ModelBase {
        public:
            explicit ModelBase(std::mt19937_64& rng, std::size_t dim) rng_(rng), dim_(dim) {
                // initialize random normal:
                beta_ = arma::randn(dim_);
                beta0_ = arma::randn<double>();
            }
            virtual ~ModelBase() = default; // destroy resources called from derived class
            virtual double likelihood(VectorXd x) = 0.;

            void ingest_data(arma::mat X, arma::vec y) {
                if (y.n_rows != X.n_rows) {
                    throw std::invalid_argument("ModelBase::ingest_data: Numer of rows in X and y do not match.");
                }
                if (X.n_cols != dim_) {
                    throw std::invalid_argument("ModelBase::ingest_data: Numer of cols in X does not match dim.");
                }
                X_ = X;
                y_ = y;
            }
            void fit(arma::mat X, arma::vec y, double beta_thresh = 0.001, std::size_t max_iter = 100){
                ingest_data(X, y);
                double dB = 1.0;
                std::size_t i = 0;
                arma::vec past_beta(dim_);
                while (dB > beta_thresh){
                    past_beta = beta_;
                    irls_update_beta();
                    update_pars_wts();
                    dB = std::abs(beta_ - past_beta);
                    i++;
                    if (i > max_iter){
                        std::cout << "ModelBase::fit() warn:\n" 
                                  << "exceeded max_iter with delta-beta = " 
                                  << dB << "\n";
                        return;
                    }
                }
            }

        protected: // to be able to access from child classes
            std::mt19937_64& rng_;
            std::size_t dim_;
            arma::mat X_;
            arma::vec y_;
            arma::vec beta_;
            double beta0_;

        private:
            // purely abstract methods to be over-ridden
            virtual void irls_update_beta()=0;
            virtual void update_pars_wts()=0;
    }

    class Guassian : ModelBase {
        public:
            Gaussian (
                std::size_t dim,
                double sigma,
                std::mt19937_64& rng
            ): ModelBase(rng, dim), sigma_(sigma) {
                // initialize random normal:
                beta_ = arma::randn(dim_);
                beta0_ = arma::randn<double>();
            }

            void update_pars_wts() override {
                // eta_ = X_ * beta_ + beta0_; // unnecessary
                // tr_mu_ = eta_; // identity link
                w_ = 1 / pow(sigma_, 2) // weights coincidentally independent of mu_ in this Gaussian case.
                z_ = y_; // response
            }

            void irls_update_beta() override {
                // LAPACK call to solve:
                //      X.T W X beta = X.T W z
                //      
                //      notice that this equation happens in dim_ -> dim_ space.
                // Solve in-place (LAPACK style) on beta_.
                nrhs = 1;
                int info = 0;
                std::vector<int> ipiv(dim_);
                A = w_ * X_.t() * X_;
                b = w_ * X.t() * z_;

                dgesv_(&dim_, &nrhs, A.begin(), &dim_, ipiv.data(), b.begin(), &dim_, &info);

                if (info > 0) {
                    cout << "dgesv_ info = " << info;
                    throw std::runtime_error(
                        "Gaussian::irls_update_beta():\n Matrix at dgesv_ solution step singular"
                    );
                }
                if (info < 0) {
                    cout << "dgesv_ info = " << info;
                    throw std::runtime_error(
                        "Illegal argument."
                    );
                }
                beta_ = b;
            }

            double log_likelihood(arma::mat X_obs, arma::vec y_obs) {
                return (
                    arma::accu(
                        arma::square(y_obs - X_obs * beta_ - beta0_) 
                    ) / (2 * pow(sigma_, 2))
                    - 0.5 * y_obs.n_cols * log(2 * pi * pow(sigma_, 2))
                );
            }
            double log_likelihood(arma::rowvec x_obs, double y_obs) {
                return (
                    arma::square(y_obs - x_obs * beta_ - beta0_)  / (2 * pow(sigma_, 2))
                    - 0.5 * y_obs.n_cols * log(2 * pi * pow(sigma_, 2))
                );
            }

        private:
            // constant parameters
            std::size_t dim_;
            double sigma_;

            // training
            arma::mat X_;
            arma::vec y_;
            arma::vec eta_;
            double w_;
            arma::vec z_;

            // inference
            arma::vec beta_;
            double beta0_;
    };

    class BinomLogit : ModelBase {
        public:
            BinomLogit(
                std::mt19937_64& rng,
                std::size_t dim,
            ) : ModelBase(rng, dim) {
                //random init
                beta_ = arma::randn(dim_);
                beta0_ = arma::randn<double>();
            }

            void update_pars_wts() override {
                eta_ = X_ * beta_ + beta0_;
                eta_ = arma::clamp(eta_, -eta_max_, eta_max_);
                p_ = 1.0 / (1.0 + arma::exp(-eta_));
                w_ = p_ * (1.0 - p_);
                z_ = eta_ + (y_ - p_) % (1.0 / (p_ % (1.0 - p_))); // response. denominator is dp / deta
            }
            void irls_update_beta() override {
                nrhs = 1;
                int info = 0;
                std::vector<int> ipiv(dim_);
                A = X_.t() * (w_ % X_.each_col());
                b = X.t() * (w_ % z_);

                dgesv_(&dim_, &nrhs, A.begin(), &dim_, ipiv.data(), b.begin(), &dim_, &info);

                if (info > 0) {
                    cout << "dgesv_ info = " << info;
                    throw std::runtime_error(
                        "Gaussian::irls_update_beta():\n Matrix at dgesv_ solution step singular"
                    );
                }
                if (info < 0) {
                    cout << "dgesv_ info = " << info;
                    throw std::runtime_error(
                        "Illegal argument."
                    );
                }
                beta_ = b;
            }

            double log_likelihood(arma::mat X_obs, arma::vec y_obs) {
                arma::vec p = 1.0 / (1.0 + arma::exp(- X_obs * beta_ - beta0_));
                std::size_t n = y.n_rows;
                double log_binom = (
                    std::lgamma(n + 1) 
                    - std::lgamma(arma::accu(y_obs) + 1)
                    - std::lgamma(n - arma::accu(y_obs) + 1)
                );
                return arma::accu(
                    y_obs % arma::log(p) - (1 - y_obs) % arma::log(1 - p)
                ) + log_binom;
            }
            double log_likelihood(arma::rowvec x_obs, double y_obs) {
                double log_binom = std::lgamma(1 + 1) - std::lgamma(y_obs + 1) - std::lgamma(1 - y_obs + 1);
                double p = 1.0 / (1.0 + exp(- x_obs * beta_ - beta0_));
                return y_obs * log(p) - (1 - y_obs) * log(1 - p);
            }

        private:
            std::size_t dim_;
            arma::vec beta_;
            double beta0_;
            arma::vec p_;
            arma::vec w_;
            constexpr double eta_max_ = 700.0;
    }

    class PoissonLog : ModelBase {
        public:
            PoissonLog(std::mt19937_64& rng, std::size_t dim) : ModelBase(rng, dim) {
                //random init
                beta_ = arma::randn(dim_);
                beta0_ = arma::randn<double>();
            }

            void update_pars_wts() override {
                eta_ = X_ * beta_ + beta0_;
                eta_ = arma::clamp(eta_, -eta_max_, eta_max_);
                lambda_ = arma::exp(eta_);
                w_ = lambda_; // dlamda / deta = lambda. => w = lambda^2 / lambda <-- variance
                z_ = eta_ + (y_ - lambda_) % (1.0 / lambda_); // response, denominator is dlambda / deta
            }
            void irls_update_beta() override {
                nrhs = 1;
                int info = 0;
                std::vector<int> ipiv(dim_);
                A = X_.t() * (w_ % X_.each_col());
                b = X.t() * (w_ % z_);

                dgesv_(&dim_, &nrhs, A.begin(), &dim_, ipiv.data(), b.begin(), &dim_, &info);

                if (info > 0) {
                    cout << "dgesv_ info = " << info;
                    throw std::runtime_error(
                        "Gaussian::irls_update_beta():\n Matrix at dgesv_ solution step singular"
                    );
                }
                if (info < 0) {
                    cout << "dgesv_ info = " << info;
                    throw std::runtime_error(
                        "Illegal argument."
                    );
                }
                beta_ = b;
            }
            
            double log_likelihood(arma::rowvec x_obs, double y_obs) {
                double lambda = exp(x_obs * beta_ + beta0_);

            }
            
        private:
            double lambda_;
            arma::vec beta_;
            double beta0_;
            arma::vec w_;
            constexpr double eta_max_ = 700.0;
    };

}