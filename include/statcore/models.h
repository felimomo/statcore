#ifndef MODELS
#define MODELS

namespace Models{
    inline double binom_coeff(std::size_t N, std::size_t m);

    class ModelBase {
        public:
            explicit ModelBase(std::size_t dim);
            virtual ~ModelBase();
            virtual double likelihood(arma::rowvec x_obs, double y_obs);
            void ingest_data(arma::mat X, arma::vec y);
            void fit(arma::mat X, arma::vec y, double beta_thresh = 0.001, std::size_t max_iter = 100);
        protected: // to be able to access from child classes
            std::size_t dim_;
            arma::mat X_;
            arma::vec y_;
            arma::vec beta_;
            double beta0_;
        private:
            virtual void irls_update_beta()=0;
            virtual void update_pars_wts()=0;
    }

    class Guassian : ModelBase {
        public:
            Gaussian (
                std::size_t dim,
                double sigma,
            );
            void update_pars_wts();
            void irls_update_beta();
            double log_likelihood(arma::mat X_obs, arma::vec y_obs);
            double log_likelihood(arma::rowvec x_obs, double y_obs);
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
    }

    class BinomLogit : ModelBase {
        public:
            BinomLogit(
                std::size_t dim
            );
            void update_pars_wts();
            void irls_update_beta();
            double log_likelihood(arma::mat X_obs, arma::vec y_obs);
            double log_likelihood(arma::rowvec x_obs, double y_obs);
        private:
            std::size_t dim_;
            arma::vec beta_;
            double beta0_;
            arma::vec p_;
            arma::vec w_;
            constexpr double eta_max_;
    }

    class PoissonLog : ModelBase {
        public:
            PoissonLog(
                std::mt19937_64& rng,
                std::size_t dim,
            );
            void update_pars_wts();
            void irls_update_beta();
            double log_likelihood(arma::mat X_obs, arma::vec y_obs);
            double log_likelihood(arma::rowvec x_obs, double y_obs);
        private:
            double lambda_;
            arma::vec beta_;
            double beta0_;
            arma::vec w_;
            constexpr double eta_max_;
    }


}

#endif