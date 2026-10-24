#ifndef LOGGER_H
#define LOGGER_H

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
        void print_summary(const std::filesystem::path& file);
    };

    class ModelLogger {
        public:
            explicit ModelLogger(const std::filesystem::path& root = ".", const std::string& name = "run");
            void begin_fit(std::size_t max_iter, double multipl_thresh = 0.05);
            void fit_step(double dB, double log_likelihood);
            void end_fit();
            void summary(
                arma::vec infer_params, 
                arma::vec targ_params
            );
            void write();
            std::filesystem::path get_step_csv() {return step_csv_; }
            std::filesystem::path get_summary_csv() {return summary_csv_; }
        private:
            std::filesystem::path step_csv_;
            std::filesystem::path summary_csv_;
            std::vector<double> step_dB_;
            std::vector<double> step_loglik_;
            steady_clock::time_point start_;
            std::chrono::duration<double> fit_time_;
            ErrorMetrics error_metrics_;
            double multipl_thresh_;
    };
}

#endif