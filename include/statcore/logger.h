#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <armadillo>

namespace Logging{

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

    class ModelLogger {
        public:
            explicit ModelLogger(const std::filesystem::path& root = ".", const std::string& name = "run");
            void begin_fit(std::size_t max_iter);
            void fit_step(double dB, double log_likelihood);
            void end_fit();
            void summary(std::vector<double> infer_params, std::vector<double> targ_params);
            void write();
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
}