#include <chrono>
#include <filesystem>
#include <fstream>
#include <armadillo>

#include "statcore/models.h"
#include "statcore/logger.h"

using std::size_t;
using std::chrono::steady_clock;
using std::filesystem::path;
using Logging::ErrorMetrics;
using Models::FitOptions;
using Models::ModelBase;

arma::vec gen_targets(size_t dim, std::mt19937& rng){
    arma::vec u(dim);
    std::normal_distribution<double> nd{0.0, 1.0};
    std::generate(u.begin(), u.end(), [&] { return nd(rng); });
    return u / arma::norm(u);
}

arma::vec gen_y_gauss(
    const arma::mat& X, 
    const arma::vec& beta, 
    double beta0, 
    double sigma, 
    std::mt19937& rng
) {
    size_t N = X.n_rows;
    std::vector<double> centered_norm(N);
    for (size_t i = 0; i < N; i++){ 
        centered_norm[i] = std::normal_distribution<double>{0.0, sigma}(rng);
    }
    return (X * beta + beta0) + arma::vec(centered_norm);
}

arma::vec gen_y_binom(const arma::mat& X, const arma::vec& beta, double beta0, std::mt19937& rng) {
    size_t N = X.n_rows;
    arma::vec p = 1.0 / (1.0 + arma::exp(- X * beta - beta0));
    arma::vec y(N);
    std::uniform_real_distribution<double> unif{0.0, 1.0};
    for (std::size_t i = 0; i < N; i++) {
        // if (unif(rng) < p[i]) { y[i] = 1.0; } else { y[i] = 0.0; } // equivalent:
        y[i] = unif(rng) < p[i]? 1.0 : 0.0;
    }
    return y;
}

arma::vec gen_y_pois(const arma::mat& X, const arma::vec& beta, double beta0, std::mt19937& rng) {
    // dependent variables
    size_t N = X.n_rows;
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        double lambda = std::exp(arma::as_scalar(X.row(i) * beta) + beta0);
        std::poisson_distribution<int> pois(lambda);
        y(i) = pois(rng);
    }
    return y;
}

struct ModelBenchmarks {
    double runtime;
    ErrorMetrics error_metrics;
};

ModelBenchmarks benchmark_model(
    ModelBase& mod, // will be modified, actually
    const arma::mat& X, 
    const arma::vec& y,
    const arma::vec& targ_params
) {
    mod.ingest_data(X,y);
    //
    steady_clock::time_point start = steady_clock::now();
    arma::vec infer_params = mod.fit_loop({.beta_thresh = 1e-3}); // {} converts to FitOptions.
    // double runtime = ((steady_clock::now() - start).count(); // unnamed defaults to ns. 
    //                                                             fix is assign it first (conversion automatic):
    std::chrono::duration<double> elapsed = steady_clock::now() - start; 
    double runtime = elapsed.count(); 
    //
    return ModelBenchmarks{
        runtime, 
        ErrorMetrics{
            y.n_rows,
            infer_params, 
            targ_params
        }
    };
}

void write_accu_bmks(const std::vector<ModelBenchmarks>& bmk_vec, const path& out_path) {
    bool is_new = !std::filesystem::exists(out_path) || std::filesystem::file_size(out_path) == 0;
    std::ofstream out_file;
    out_file.open(out_path, std::ios::app);
    if(is_new){
        out_file << "N,dim,max_err,mean_err,median_err,err_std,frac_wrong\n";
    }
    for(int i = 0; const auto& bmk: bmk_vec){
        if (!bmk.error_metrics.has_nan_) {
            out_file << bmk.error_metrics.N_ << ","
                    << bmk.error_metrics.dim_ << ","
                    << bmk.error_metrics.max_err << ","
                    << bmk.error_metrics.mean_err << ","
                    << bmk.error_metrics.median_err << ","
                    << bmk.error_metrics.err_std << ","
                    << bmk.error_metrics.frac_wrong << "\n";
        } else {
            // encode nan as blank spaces
            out_file << bmk.error_metrics.N_ << ","
                        << bmk.error_metrics.dim_ << ","
                        << ","
                        << ","
                        << ","
                        << ","
                        << "\n";
        }
        i++;
    } // for i
}

void write_runtime_bmks(const std::vector<ModelBenchmarks>& bmk_vec, const path& out_path) {
    bool is_new = !std::filesystem::exists(out_path) || std::filesystem::file_size(out_path) == 0;
    std::ofstream out_file;
    out_file.open(out_path, std::ios::app);
    if(is_new){
        out_file << "N,dim,runtime,max_err,mean_err,median_err,err_std,frac_wrong\n";     
    }
    for(int i = 0; const auto& bmk: bmk_vec){
        out_file << bmk.error_metrics.N_ << ","
                 << bmk.error_metrics.dim_ << ","
                 << bmk.runtime << ","
                 << bmk.error_metrics.max_err << ","
                 << bmk.error_metrics.mean_err << ","
                 << bmk.error_metrics.median_err << ","
                 << bmk.error_metrics.err_std << ","
                 << bmk.error_metrics.frac_wrong << "\n";
        i++;
    }
}

ModelBenchmarks gauss_bmk_subroutine(size_t dim, size_t N, std::mt19937& rng) {
    // 1. sample 
    // 1.1 model params
    arma::vec beta = gen_targets(dim, rng);
    double beta0 = 0.0;
    arma::vec targ_params = arma::join_cols(beta, arma::vec{beta0});
    double sigma = 0.05;
    // 1.2 data
    // arma::mat X = arma::randn<arma::mat>(N, dim); // use thread-safe rng:
    arma::mat X(N, dim);
    std::normal_distribution<double> nd{0.0, 1.0};
    std::generate(X.begin(), X.end(), [&] { return nd(rng); });
    arma::vec y = gen_y_gauss(X, beta, beta0, sigma, rng);
    
    // 2. initialize model
    Models::Gaussian mod(dim, sigma);
    
    // 3. benchmark fit
    return benchmark_model(mod, X, y, targ_params);
}

ModelBenchmarks binom_bmk_subroutine(size_t dim, size_t N, std::mt19937& rng) {
    // 1. sample 
    // 1.1 model params
    arma::vec beta = gen_targets(dim, rng);
    double beta0 = 0.0;
    arma::vec targ_params = arma::join_cols(beta, arma::vec{beta0});
    // 1.2 data
    // arma::mat X = arma::randn<arma::mat>(N, dim); // use thread-safe rng:
    arma::mat X(N, dim);
    std::normal_distribution<double> nd{0.0, 1.0};
    std::generate(X.begin(), X.end(), [&] { return nd(rng); });
    arma::vec y = gen_y_binom(X, beta, beta0, rng);
    
    // 2. initialize model
    Models::BinomLogit mod(dim);
    
    // 3. benchmark fit
    return benchmark_model(mod, X, y, targ_params);
}

ModelBenchmarks pois_bmk_subroutine(size_t dim, size_t N, std::mt19937& rng) {
    // 1. sample 
    // 1.1 model params
    arma::vec beta = gen_targets(dim, rng);
    double beta0 = 0.0;
    arma::vec targ_params = arma::join_cols(beta, arma::vec{beta0});
    // 1.2 data
    // arma::mat X = arma::randn<arma::mat>(N, dim); // use thread-safe rng:
    arma::mat X(N, dim);
    std::normal_distribution<double> nd{0.0, 1.0};
    std::generate(X.begin(), X.end(), [&] { return nd(rng); });
    arma::vec y = gen_y_pois(X, beta, beta0, rng);
    
    // 2. initialize model
    Models::PoissonLog mod(dim);
    
    // 3. benchmark fit
    return benchmark_model(mod, X, y, targ_params);
}

int main () {
    // globals
    size_t N_iter = 25;
    size_t base_seed = 42;

    // benchmarks definitions
    std::vector<size_t> dims {10, 25, 50, 100, 150};
    std::vector<size_t> dim_to_N {10, 100, 1000, 10000};

    // benchmarks loop
    for(size_t i = 0; auto dim: dims){
        for(size_t j = 0; auto dtn: dim_to_N){
            arma::uword N = dim * dtn;
            j++;

            // Gaussian
            // accuracy benchmarks
            std::cout << "Working on dim = " << dim << ", N = " << N << "\n";
            {
                std::vector<ModelBenchmarks> gaussian_bmk(N_iter);
                #pragma omp parallel for
                for(size_t i = 0; i < N_iter; i++) {
                    // 0. thread-safe random generator
                    std::mt19937 rng(base_seed + i);
                    gaussian_bmk[i] = gauss_bmk_subroutine(dim, N, rng);
                }
                write_accu_bmks(gaussian_bmk, "./bench/data/Gaussian_GLM_accu_bmk.csv");
            }
            // runtime benchmarks (maybe fewer because of no parallelization (competition for resources by threads))
            {
                size_t N_iter_rt = 10;
                std::vector<ModelBenchmarks> gaussian_bmk(N_iter_rt);
                arma::vec runtimes(N_iter_rt);
                for(size_t i = 0; i < N_iter_rt; i++) {
                    // 0. thread-safe random generator
                    std::mt19937 rng(base_seed + i);
                    gaussian_bmk[i] = gauss_bmk_subroutine(dim, N, rng);
                    runtimes[i] = gaussian_bmk[i].runtime;
                }
                write_runtime_bmks(gaussian_bmk, "./bench/data/Gaussian_GLM_runtime_bmk.csv");
            }

            // Binomial
            // accuracy benchmarks
            std::cout << "Working on dim = " << dim << ", N = " << N << "\n";
            {
                std::vector<ModelBenchmarks> binom_bmk(N_iter);
                #pragma omp parallel for
                for(size_t i = 0; i < N_iter; i++) {
                    // 0. thread-safe random generator
                    std::mt19937 rng(base_seed + i);
                    binom_bmk[i] = binom_bmk_subroutine(dim, N, rng);
                }
                write_accu_bmks(binom_bmk, "./bench/data/BinomLogit_GLM_accu_bmk.csv");
            }
            // runtime benchmarks (maybe fewer because of no parallelization (competition for resources by threads))
            {
                size_t N_iter_rt = 10;
                std::vector<ModelBenchmarks> binom_bmk(N_iter_rt);
                arma::vec runtimes(N_iter_rt);
                for(size_t i = 0; i < N_iter_rt; i++) {
                    // 0. thread-safe random generator
                    std::mt19937 rng(base_seed + i);
                    binom_bmk[i] = binom_bmk_subroutine(dim, N, rng);
                    runtimes[i] = binom_bmk[i].runtime;
                }
                write_runtime_bmks(binom_bmk, "./bench/data/BinomLogit_GLM_runtime_bmk.csv");
            }

            // Poisson
            // accuracy benchmarks
            std::cout << "Working on dim = " << dim << ", N = " << N << "\n";
            {
                std::vector<ModelBenchmarks> pois_bmk(N_iter);
                #pragma omp parallel for
                for(size_t i = 0; i < N_iter; i++) {
                    // 0. thread-safe random generator
                    std::mt19937 rng(base_seed + i);
                    pois_bmk[i] = pois_bmk_subroutine(dim, N, rng);
                }
                write_accu_bmks(pois_bmk, "./bench/data/PoissonLog_GLM_accu_bmk.csv");
            }
            // runtime benchmarks (maybe fewer because of no parallelization (competition for resources by threads))
            {
                size_t N_iter_rt = 10;
                std::vector<ModelBenchmarks> pois_bmk(N_iter_rt);
                arma::vec runtimes(N_iter_rt);
                for(size_t i = 0; i < N_iter_rt; i++) {
                    // 0. thread-safe random generator
                    std::mt19937 rng(base_seed + i);
                    pois_bmk[i] = pois_bmk_subroutine(dim, N, rng);
                    runtimes[i] = pois_bmk[i].runtime;
                }
                write_runtime_bmks(pois_bmk, "./bench/data/PoissonLog_GLM_runtime_bmk.csv");
            }
        }
        i++;
    }
}