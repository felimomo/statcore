#ifndef MODELS_H
#define MODELS_H

#include  <cmath>
#include <armadillo>

#include "statcore/logger.h"

namespace Models{
    inline double binom_coeff(std::size_t N, std::size_t m);

    class ModelBase {
        public:
            explicit ModelBase(std::size_t dim);
            virtual ~ModelBase() = default;
            void ingest_data(const arma::mat& X, const arma::vec& y);
            arma::vec fit(
                const arma::mat& X, 
                const arma::vec& y, 
                double beta_thresh = 0.001, 
                std::size_t max_iter = 100,
                Logging::ModelLogger* logger = nullptr
            );
            arma::vec get_beta();
            double get_beta0();
            virtual double log_likelihood(const arma::mat& X_obs, const arma::vec& y_obs) const = 0;
        protected: // to be able to access from child classes
            std::size_t dim_;
            arma::mat X_;
            arma::vec y_;
            arma::vec beta_;
            double beta0_;
            arma::vec w_;
        private:
            virtual void irls_update_beta() = 0;
            virtual void update_pars_wts() = 0;
    };

    class Gaussian : public ModelBase { // public keyword to inherit public methods publically by default
        public:
            Gaussian (
                std::size_t dim,
                double sigma
            );
            void update_pars_wts() override;
            void irls_update_beta() override;
            double log_likelihood(const arma::mat& X_obs, const arma::vec& y_obs) const override;
        private:
            double sigma_;
            arma::vec eta_;
            arma::vec z_;
    };

    class BinomLogit : public ModelBase { // public keyword to inherit public methods publically by default
        public:
            BinomLogit(std::size_t dim);
            void update_pars_wts() override;
            void irls_update_beta() override;
            double log_likelihood(const arma::mat& X_obs, const arma::vec& y_obs) const override;
        private:
            arma::vec eta_;
            arma::vec p_;
            arma::vec z_;
            const double eta_max_=700.0;
    };

    class PoissonLog : public ModelBase { // public keyword to inherit public methods publically by default
        public:
            PoissonLog(std::size_t dim);
            void update_pars_wts() override;
            void irls_update_beta() override;
            double log_likelihood(const arma::mat& X_obs, const arma::vec& y_obs) const override;
        private:
            arma::vec lambda_;
            arma::vec eta_;
            arma::vec z_;
            const double eta_max_=700.0;
    };


}

#endif