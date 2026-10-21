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

    inline double logit(double p) {
        return std::log(p / (1 - p));
    }

    inline double logistic(double a) {
        return 1.0 / (1.0 + exp(-a))
    }

    class ModelBase {
        public:
            explicit ModelBase(std::mt19937_64& rng) rng_(rng) {}
            virtual ~ModelBase() = default; // destroy resources called from derived class
            virtual double likelihood(VectorXd x) = 0.;
        protected: // to be able to access from child classes
            std::mt19937_64& rng_;
    }

    class Guassian : ModelBase {
        public:
            Gaussian (
                std::size_t dim,
                double sigma,
                std::mt19937_64& rng
            ): ModelBase(rng), dim_(dim), sigma_(sigma) {
                // initialize random normal:
                beta_ = arma::randn(dim_);
                beta0_ = arma::randn<double>();
            }

            void ingest_data(arma::mat X, arma::vec y) {
                if (y.n_rows != X.n_rows) {
                    throw std::invalid_argument("Gaussian::ingest_data: Numer of rows in X and y do not match.");
                }
                if (X.n_cols != dim_) {
                    throw std::invalid_argument("Gaussian::ingest_data: Numer of cols in X does not match dim.");
                }
                X_ = X;
                y_ = y;
            }

            void update_pars_wts() {
                eta_ = beta_.t() * X_ + beta0_ * arma::ones<arma::vec>(y.n_rows);
                mu_ = eta_; // identity link
                w_ = 1 / pow(sigma_, 2) // weights coincidentally independent of mu_ in this Gaussian case.
                z_ = y_; response
            }

            void irls_update_beta() {
                // tbd: LAPACK call to solve:
                //      X.T W X beta = X.T W z
                // Solve in-place (LAPACK style) on beta_.
            }

            inline double log_likelihood() {
                return (
                    arma::accu(
                        arma::square(y_ - mu_) 
                    ) / (2 * pow(sigma_, 2))
                    - 0.5 * y_.n_cols * log(2 * pi * pow(sigma_, 2))
                );
            }



        private:
            std::size_t dim_;
            arma::vec beta_;
            double beta0_;
            double sigma_;

            arma::mat X_;
            arma::vec y_;
            arma::vec mu_; 

            arma::vec eta_;
            double w_;
            arma::vec z_;
    };

    class BinomLogit : ModelBase {
        public:
            BinomLogit(
                std::mt19937_64& rng,
                std::size_t dim
            ) : ModelBase(rng), dim_(dim) {}

            inline void random_init() {
                beta_ = VectorXd::random(dim_);
                beta0_ = 
            }
        private:
            std::size_t dim_;
            VectorXd beta_;
            double beta0_;
            VectorXd p_;
            VectorXd w_;
    }

    class BinomLogit {
        public:
            BinomLogit(
                std::size_t N, 
                double p, 
                VectorXd beta,
                double beta0
            ) : ModelBase(rng), beta_(beta), beta0_(beta0) {}

            double likelihood(VectorXd x) {
                return 1.0 / (1.0 + exp(-beta_.dot(x) - beta0_)); //probability of (y = 1 | x)
            }
            double predict(VectorXd x) {
                L = likelihood(x);
                std::uniform_real_distribution<> dis(0.0, 1.0);
                if (dis(rng_) <= L) {
                    return 1;
                } else {
                    return 0;
                }
            }
        private:
            VectorXd beta_;
            double beta0_;
    };

    class PoissonLog {
        public:
            PoissonLog(lambda, std::mt19937_64& rng) : ModelBase(rng), lambda_(lambda) {}
            
            double likelihood(VectorXd)
        private:
            double lambda_;
    };

}