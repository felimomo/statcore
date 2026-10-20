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
    std::vector<int> dimensions = { 256, 400, 512, 800, 1024, 1500, 2048, 2500 };
    std::vector<int> n_rep =      { 30,  30,  30,  20,  10,   10,   5,    5 };
    //
    // io 
    std::string file_path = "bench/data/matmult_benchmarks.csv";
    { // open in limited scope
        std::ofstream outFile(file_path);
        if (!outFile) {
            std::cerr << "Error opening file!" << std::endl;
            return 0;
        }
        outFile << "dim,rep,method,runtime\n";
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

            auto [bloc64_64_C, bloc64_64_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,64,64,false); });
            auto [simd64_64_C, simd64_64_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,64,64,true); });

            auto [bloc64_32_C, bloc64_32_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,64,32,false); });
            auto [simd64_32_C, simd64_32_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,64,32,true); });

            auto [bloc32_64_C, bloc32_64_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,32,64,false); });
            auto [simd32_64_C, simd32_64_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,32,64,true); });

            auto [bloc32_32_C, bloc32_32_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,32,32,false); });
            auto [simd32_32_C, simd32_32_t] = time_mm([A,B] {return matmult::tiled_sq_mult(A,B,32,32,true); });

            auto [claude_C, claude_t] = time_mm([A,B] {
                        return matmult::tiled_sq_mult_claude<matmult::reg_blocked_mult>(A, B); 
                });

            auto [ref_C, ref_t] = time_mm([&A, &B]() -> arma::mat { return A * B; });
            // avoid Arma lazy eval which was the default with:
            // time_mm([A,B] {return A * B; });
            auto [naive_C, naive_t] = time_mm([A,B] {return matmult::ijk_mult(A, B); });

            //
            // io scope:
            {
                std::ofstream outFile(file_path, std::ios::app); // 2nd arg: flag to write at bottom of file
                outFile << std::setprecision(std::numeric_limits<double>::max_digits10);
                // each method: one line

                // 64 target C block, 64 contracted dim
                outFile << dim<< "," << rep<< "," << "bloc64_64"<< ","
                        << bloc64_64_t << "\n";
                outFile << dim<< "," << rep<< "," << "simd64_64"<< ","
                        << simd64_64_t << "\n";
                
                // 64 target C block, 32 contracted dim
                outFile << dim<< "," << rep<< "," << "bloc64_32"<< ","
                        << bloc64_32_t << "\n";
                outFile << dim<< "," << rep<< "," << "simd64_32"<< ","
                        << simd64_32_t << "\n";

                // 32 target C block, 64 contracted dim
                outFile << dim<< "," << rep<< "," << "bloc32_64"<< ","
                        << bloc32_64_t << "\n";
                outFile << dim<< "," << rep<< "," << "simd32_64"<< ","
                        << simd32_64_t << "\n";

                // 32 target C block, 32 contracted dim
                outFile << dim<< "," << rep<< "," << "bloc32_32"<< ","
                        << bloc32_32_t << "\n";
                outFile << dim<< "," << rep<< "," << "simd32_32"<< ","
                        << simd32_32_t << "\n";

                // armadillo native and naive
                outFile << dim<< "," << rep<< "," << "armadillo"<< ","
                        << ref_t << "\n";
                outFile << dim<< "," << rep<< "," << "naive"<< ","
                        << naive_t << "\n";

                // claude claude_t
                outFile << dim<< "," << rep<< "," << "claude_4x4"<< ","
                        << claude_t << "\n";

            }
        }


        i++;
    }
}