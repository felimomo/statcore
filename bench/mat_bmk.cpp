#include <armadillo>
#include <chrono>
#include <fstream>
#include <string> 
#include "statcore/matmult.h"


// TBD: factor this out to source, import it here
struct TimedMatMult { arma::mat result; double seconds; };

struct MatMultBmk {
    std::vector<int>  dim;
    std::vector<int>  rep;
    std::vector<std::string> method;
    std::vector<double> runtime;
    std::vector<double> error;
};

// auto [sum, seconds] = time_it([&] { return agg::KahanSum(v); }); 
//--> automatically unpacks output into two named vars
// (return type is a struct)
TimedMatMult time_mm (auto&& f) { 
    // f will be a lambda that takes no arguments 
    // (arguments provided to sum fns as scope variables passed to the lambda)
    auto t1 = std::chrono::steady_clock::now();
    auto result_mat = f();
    auto t2 = std::chrono::steady_clock::now();
    return TimedMatMult{result_mat, std::chrono::duration<double>(t2 - t1).count()};
}

int main() {
    //
    // benchmark properties
    std::vector<int> dimensions = { 256, 512, 1024, 2048 };
    std::vector<int> n_rep = { 50, 30, 20, 10 };
    //
    // io 
    std::string file_path = "bench/data/matmult_benchmarks.csv";
    { // open in limited scope
        std::ofstream outFile(file_path);
        if (!outFile) {
            std::cerr << "Error opening file!" << std::endl;
            return 0;
        }
        outFile << "dim,rep,method,error,runtime\n";
    }
    //
    // bmk loop
    for(int i = 0; auto const& dim : dimensions) {
        std::cout << "Sampling random matrices...\n" << std::flush;
        arma::cube mat_A_lib = arma::randn<arma::cube>(dim, dim, n_rep[i]); 
        arma::cube mat_B_lib = arma::randn<arma::cube>(dim, dim, n_rep[i]); 
        std::cout << "Starting benchmarks for dim = " << dim  << "\n" << std::flush;

        for(int rep = 0; rep < n_rep[i]; rep++) {
            arma::mat A = mat_A_lib.slice(rep);
            arma::mat B = mat_B_lib.slice(rep);            
            auto [ijk_C, ijk_t] = time_mm([A,B]{ return matmult::ijk_mult(A,B); });
            auto [ikj_C, ikj_t] = time_mm([A,B]{ return matmult::ikj_mult(A,B); });
            auto [jki_C, jki_t] = time_mm([A,B]{ return matmult::jki_mult(A,B); });
            auto [manopt_C, manopt_t] = time_mm([A,B]{ return matmult::manopt_mult(A,B); });
            auto [b128_C, b128_t] = time_mm([A,B]{ return matmult::sq_mat_block_mult(A,B,128); });
            auto [b64_C, b64_t] = time_mm([A,B]{ return matmult::sq_mat_block_mult(A,B,64); });
            auto [b32_C, b32_t] = time_mm([A,B]{ return matmult::sq_mat_block_mult(A,B,32); });
            auto [simd128_C, simd128_t] = time_mm([A,B]{ return matmult::sq_mat_simd_block_mult(A,B,128); });
            auto [simd64_C, simd64_t] = time_mm([A,B]{ return matmult::sq_mat_simd_block_mult(A,B,64); });
            auto [simd32_C, simd32_t] = time_mm([A,B]{ return matmult::sq_mat_simd_block_mult(A,B,32); });
            //
            arma::mat arma_C = A * B;
            //
            const double ijk_err =    arma::norm(arma_C - ijk_C, 2);
            const double ikj_err =    arma::norm(arma_C - ikj_C, 2);
            const double jki_err =    arma::norm(arma_C - jki_C, 2);
            const double manopt_err = arma::norm(arma_C - manopt_C, 2);
            const double b128_err =   arma::norm(arma_C - b128_C, 2);
            const double b64_err =    arma::norm(arma_C - b64_C, 2);
            const double b32_err =    arma::norm(arma_C - b32_C, 2);
            const double simd128_err =arma::norm(arma_C - simd128_C, 2);
            const double simd64_err = arma::norm(arma_C - simd64_C, 2);
            const double simd32_err = arma::norm(arma_C - simd32_C, 2);
            //
            // io scope:
            {
                std::ofstream outFile(file_path, std::ios::app); // 2nd arg: flag to write at bottom of file
                outFile << std::setprecision(std::numeric_limits<double>::max_digits10);
                // each method: one line
                outFile << dim << "," << rep<< "," << "ijk"<< ","
                        << ijk_err<< "," << ijk_t << "\n";
                outFile << dim<< "," << rep<< "," << "ikj"<< ","
                        << ikj_err<< "," << ikj_t << "\n";
                outFile << dim<< "," << rep<< "," << "jki"<< ","
                        << jki_err<< "," << jki_t << "\n";
                outFile << dim<< "," << rep<< "," << "ManOpt"<< ","
                        << manopt_err<< "," << manopt_t << "\n";
                outFile << dim<< "," << rep<< "," << "block128"<< ","
                        << b128_err<< "," << b128_t << "\n";
                outFile << dim<< "," << rep<< "," << "block64"<< ","
                        << b64_err<< "," << b64_t << "\n";
                outFile << dim<< "," << rep<< "," << "block32"<< ","
                        << b32_err<< "," << b32_t << "\n";
                outFile << dim<< "," << rep<< "," << "simd128"<< ","
                        << simd128_err<< "," << simd128_t << "\n"; 
                outFile << dim<< "," << rep<< "," << "simd64"<< ","
                        << simd64_err<< "," << simd64_t << "\n";
                outFile << dim<< "," << rep<< "," << "simd32"<< ","
                        << simd32_err<< "," << simd32_t << "\n";
            }
        }


        i++;
    }
}