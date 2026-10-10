#include <armadillo>
#include <chrono>
#include <fstream>
#include <string> 
#include "statcore/matmult.h"


// TBD: factor this out to source, import it here
struct TimedMatMult { arma::mat result; double seconds; };

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
    std::vector<int> dimensions = { 256, 512, 1024, 2048, 4096 };
    int n_rep = {50, 50, 50, 25, 25}

    for(int i = 0; auto const& dim : dimensions) {
        arma::cube mat_A_lib = arma::randn<arma::cube>(dim, dim, n_rep[i]); 
        arma::cube mat_B_lib = arma::randn<arma::cube>(dim, dim, n_rep[i]); 
        for(int rep = 0; rep < n_rep[i]; rep++) {
            A = mat_A_lib.slice(rep);
            B = mat_B_lib.slice(rep);            
            [ijk_C, ijk_t] = time_mm([A,B]{ return matmult::ijk_mult(A,B); });
            [ikj_C, ikj_t] = time_mm([A,B]{ return matmult::ikj_mult(A,B); });
            [jki_C, jki_t] = time_mm([A,B]{ return matmult::jki_mult(A,B); });
            [manopt_C, manopt_t] = time_mm([A,B]{ return matmult::manopt_mult(A,B); });
            [b128_C, b128_t] = time_mm([A,B]{ return matmult::sq_mat_block_mult(A,B,128); });
            [b64_C, b64_t] = time_mm([A,B]{ return matmult::sq_mat_block_mult(A,B,64); });
            [b32_C, b32_t] = time_mm([A,B]{ return matmult::sq_mat_block_mult(A,B,32); });
            [simd128_C, simd128_t] = time_mm([A,B]{ return matmult::sq_mat_simd_block_mult(A,B,128); });
            [simd64_C, simd64_t] = time_mm([A,B]{ return matmult::sq_mat_simd_block_mult(A,B,64); });
            [simd32_C, simd32_t] = time_mm([A,B]{ return matmult::sq_mat_simd_block_mult(A,B,32); });
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
        }


        i++;
    }
}