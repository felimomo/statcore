#ifndef MODELS
#define MODELS

namespace Models{
    inline double binom_coeff(std::size_t N, std::size_t m);

    class ModelBase {
        public:
            explicit ModelBase(std::mt19937_64& rng, std::size_t dim);
            virtual ~ModelBase();
            void ingest_data(arma::mat X, arma::vec y);
        protected: // to be able to access from child classes
            std::mt19937_64& rng_;
            std::size_t dim_;
            arma::mat X_;
            arma::vec y_;
    }

    class Guassian : ModelBase {
        public:
            Gaussian (
                std::size_t dim,
                double sigma,
                std::mt19937_64& rng
            );
            void update_pars_wts();
            void irls_update_beta();
            void fit(arma::mat X, arma::vec y, double beta_thresh = 0.001, std::size_t max_iter = 100);
            double log_likelihood();
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
    }

    class BinomLogit : ModelBase {
        public:
            BinomLogit(
                std::mt19937_64& rng,
                std::size_t dim,
            );
            void update_pars_wts();
            void irls_update_beta();
            void fit(arma::mat X, arma::vec y, double beta_thresh = 0.001, std::size_t max_iter = 100);
            double log_likelihood();
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
            void fit(arma::mat X, arma::vec y, double beta_thresh = 0.001, std::size_t max_iter = 100);
            double log_likelihood();
        private:
            double lambda_;
            arma::vec beta_;
            double beta0_;
            arma::vec w_;
            constexpr double eta_max_;
    }


}

#endif