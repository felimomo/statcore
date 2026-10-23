#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <armadillo>

namespace Logging{

    using std::chrono::steady_clock;

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
    ErrorMetrics::print_summary(const std::filesystem::path& file) {
        std::ofstream file_stream;
        file_stream.open(file);
        file_stream << "max_err,mean_err,median_err,err_std,frac_wrong\n"
                    << max_err_    << "," << mean_err_ << ","
                    << median_err_ << "," << err_std_ << ","
                    << frac_wrong  << "\n";
    }

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
        arma::vec targ_params
    ){
        error_metrics_ = ErrorMetrics(
            arma::vec infer_params, 
            arma::vec targ_params, 
            double multipl_thresh_
        );
    }

    void ModelLogger::write() {
        // summary
        error_metrics_.print_summary(summary_csv_);
        // steps
        std::ofstream stp_str;
        stp_str.open(step_csv_);
        stp_str << "step,dB,log_likelihood\n";
        for (std::size_t i = 0; i < dB_.size(); i++){
            stp_str << i << "," << dB_[i] << "," << log_lik[i] << "\n";
        }
    }

}
