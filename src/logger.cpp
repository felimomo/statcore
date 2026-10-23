#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <armadillo>

namespace Logging{

    using std::chrono::steady_clock;

    struct ErrorMetrics {
        // data
        double max_err = 0.0;
        double mean_err = 0.0;
        double median_err = 0.0;
        double err_std = 0.0;
        double frac_wrong = 0.0;

        //config
        double multipl_thresh_ = 0.0;
        double beta_thresh_ = 1.0; // tbd probably have this match the constructor default

        ErrorMetrics() = default; // for header declaration
        ErrorMetrics(
            arma::vec& infer_params, 
            arma::vec& targ_params, 
            double multipl_thresh = 0.05, 
            double beta_thresh = 0.0001
        );
    };

    ErrorMetrics::ErrorMetrics(
        arma::vec& infer_params, 
        arma::vec& targ_params, 
        double multipl_thresh,
        double beta_thresh
    ): 
        multipl_thresh_(multipl_thresh),
        beta_thresh_(beta_thresh)
    {
        arma::vec err = arma::abs(infer_params - targ_params);
        max_err = err.max();
        mean_err = arma::mean(err);
        median_err = arma::median(err);
        err_std = arma::stddev(err);

        arma::uvec healthy_idx = arma::find(infer_params > beta_thresh);

        arma::uword N_wrong = arma::accu(
            arma::abs( 
                err.elem(healthy_idx) % (1 / infer_params.elem(healthy_idx))
            ) 
            > multipl_thresh
        );
        frac_wrong = static_cast<double>(N_wrong) / healthy_idx.size(); 
    }

    class ModelLogger {
        public:
            explicit ModelLogger(const std::filesystem::path& root = ".", const std::string& name = "run");
            void begin_fit(std::size_t max_iter);
            void fit_step(double dB, double log_likelihood);
            void end_fit();
            void summary(std::vector<double> infer_params, std::vector<double> targ_params);
        private:
            std::filesystem::path step_csv_;
            std::filesystem::path summary_csv_;
            std::vector<double> step_dB_;
            std::vector<double> step_loglik_;
            steady_clock::time_point start_;
            steady_clock::time_point fit_time_;
            ErrorMetrics error_metrics_;
            double multipl_thresh_;
    };

    ModelLogger::ModelLogger(
        const std::filesystem::path& root, 
        const std::string& name
    ) : 
        step_csv_(root / (name + "_step.csv")), 
        summary_csv_(root / (name + "_summary.csv")) 
    {
        std::filesystem::create_directories(root);
    }

    void ModelLogger::begin_fit(std::size_t max_iter, double multipl_thresh = 0.05) {
        multipl_thresh_ = multipl_thresh;
        step_dB_.clear();
        step_dB_.reserve(max_iter + 1);   
        step_loglik_.clear();
        step_loglik_.reserve(max_iter + 1); 
        start_ = steady_clock::now();
    }

    void ModelLogger::end_fit() {
        fit_time_ = steady_clock::now() - start_;
    }

    void ModelLogger::fit_step(
        double dB, 
        double log_lik
    ) {
        step_dB_.push_back(dB);
        step_loglik_.push_back(log_lik);
    }

    void ModelLogger::summary(
        arma::vec infer_params, 
        arma::vec targ_params,
    ){
        error_metrics_ = ErrorMetrics(
            arma::vec infer_params, 
            arma::vec targ_params, 
            double multipl_thresh_
        );
    }

}
