#ifndef MODELS
#define MODELS

#include  <cmath>
#include <Armadillo>

namespace Models{
    inline double binom_coeff(std::size_t N, std::size_t m);

    class ModelBase {
        public:
            explicit ModelBase(std::size_t dim);
            virtual ~ModelBase() = default;
            void ingest_data(arma::mat X, arma::vec y);
            void fit(arma::mat X, arma::vec y, double beta_thresh, std::size_t max_iter);
            arma::vec get_beta();
            double get_beta0();
        protected: // to be able to access from child classes
            std::size_t dim_;
            arma::mat X_;
            arma::vec y_;
            arma::vec beta_;
            double beta0_;
        private:
            virtual void irls_update_beta() = 0;
            virtual void update_pars_wts() = 0;
    };

    class Gaussian : ModelBase {
        public:
            Gaussian (
                std::size_t dim,
                double sigma
            );
            void update_pars_wts() override;
            void irls_update_beta() override;
            double log_likelihood(arma::mat X_obs, arma::vec y_obs) const;
            double log_likelihood(arma::rowvec x_obs, double y_obs) const;
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
            BinomLogit(std::size_t dim);
            void update_pars_wts() override;
            void irls_update_beta() override;
            double log_likelihood(arma::mat X_obs, arma::uvec y_obs) const;
            double log_likelihood(arma::rowvec x_obs, std::size_t y_obs) const;
        private:
            std::size_t dim_;
            arma::vec beta_;
            arma::vec eta_;
            double beta0_;
            arma::vec p_;
            arma::vec w_;
            arma::vec z_;
            const double eta_max_=700.0;
    };

    class PoissonLog : ModelBase {
        public:
            PoissonLog(std::size_t dim);
            void update_pars_wts() override;
            void irls_update_beta() override;
            double log_likelihood(arma::mat X_obs, arma::uvec y_obs) const;
            double log_likelihood(arma::rowvec x_obs, std::size_t y_obs) const;
        private:
            arma::vec lambda_;
            arma::vec beta_;
            arma::vec eta_;
            double beta0_;
            arma::vec w_;
            arma::vec z_;
            const double eta_max_=700.0;
    };


}

#endif