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
            C 

        }


        i++;
    }
}