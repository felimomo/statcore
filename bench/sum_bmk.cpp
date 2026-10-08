#include <armadillo>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <ranges>
#include <string> 
#include <vector>
#include "statcore/agg.h"

// time functions
struct TimedResult { double result; double seconds; };
// struct FunBmk {
//     arma::uvec dim;
//     arma::uvec rep;
//     arma::vec value;
//     arma::vec time;
// };

// goal call:
// auto [sum, seconds] = time_it([&] { return agg::KahanSum(v); }); 
//--> automatically unpacks output into two named vars
// (return type is a struct)
TimedResult time_it (auto&& f) { 
    // f will be a lambda that takes no arguments 
    // (arguments provided to sum fns as scope variables passed to the lambda)
    auto t1 = std::chrono::steady_clock::now();
    auto result = f();
    auto t2 = std::chrono::steady_clock::now();
    return TimedResult{result, std::chrono::duration<double>(t2 - t1).count()};
}

// FunBmk bmk_it(auto&& f, int n, int dim) {
//     arma::vec values(n);
//     arma::vec ground_truths(n);
//     arma::vec runtimes(n);
//     arma::uvec rep = arma::regspace<arma::uvec>(1, n); // fill rep as {1, 2, 3, ...}
//     for (int i = 0; i<n; i++) {
//         arma::vec randv = arma::randn<arma::vec>(dim);
//         std::vector<double> x = arma::conv_to<std::vector<double>>::from(randv);
//         auto [res, t] = time_it([&] { return f(x); });
//         ground_truths[i] = arma::accu(randv)
//         values[i] = res;
//         runtimes[i] = t;
//     }
//     return FunBmk{
//         n,
//         rep,
//         ground_truths,
//         values,
//         runtimes
//     };
// }

void bmk_io(std::string method, int dim, const arma::vec& values, const arma::vec& runtimes) {
    std::string file_path = "bench/data/sum_benchmarks.csv";
    std::ofstream outFile(file_path, std::ios::app); // 2nd arg: flag to write at bottom of file
    if (!outFile) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    
    outFile << std::setprecision(std::numeric_limits<double>::max_digits10); // to differentiate methods
    if( std::filesystem::is_empty(file_path) ) {
        outFile << "method,rep,dim,value,runtime\n";
    }

    for (std::size_t i = 0; auto const& v: values) { // c++20 syntax: (init; item: range)
        outFile << method << ","
                << i+1 << ","
                << dim << ","
                << v << ","
                << runtimes[i] << "\n";
        i++; // very important to not forget this itsy-bitsy little line!
    }
    // closing manually not needed because destructor is called automatically
    // when we go out of scope (the function returns or throws).
}



int main(){
    //one scope per method for memory safety

    std::vector<int> dimensions = {
        128, 
        128 * 4 , 
        128 * 4 * 4, 
        128 * 4 * 4 * 4, 
        128 * 4 * 4 * 4 * 4,
        128 * 4 * 4 * 4 * 4 * 4,
        128 * 4 * 4 * 4 * 4 * 4 * 4
    };
    std::vector<int> n_reps = {
        1024,
        512,
        256,
        128,
        64,
        32,
        16
    };   

    
    for(std::size_t i = 0; auto const& dim: dimensions){
        // generate vectors
        arma::mat vec_lib = arma::randu<arma::mat>(dim, n_reps[i]);  

        // Kahan
        {  
            arma::vec values(n_reps[i]);
            arma::vec runtimes(n_reps[i]);
            for(int j=0; j<n_reps[i]; j++){
                std::vector<double> x = arma::conv_to<std::vector<double>>::from(vec_lib.col(j));
                auto [res, t] = time_it([&] { return agg::KahanSum(x); });
                values[j] = res;
                runtimes[j] = t;
            }
            bmk_io("kahan", dim, values, runtimes);
        }

        // Pairwise
        {  
            arma::vec values(n_reps[i]);
            arma::vec runtimes(n_reps[i]);
            for(int j=0; j<n_reps[i]; j++){
                std::vector<double> x = arma::conv_to<std::vector<double>>::from(vec_lib.col(j));
                auto [res, t] = time_it([&] { return agg::pairwiseSum(x); });
                values[j] = res;
                runtimes[j] = t;
            }
            bmk_io("pairwise", dim, values, runtimes);
        }
        
        // Armadillo
        {  
            arma::vec values(n_reps[i]);
            arma::vec runtimes(n_reps[i]);
            for(int j=0; j<n_reps[i]; j++){
                auto [res, t] = time_it([&] { return arma::accu(vec_lib.col(j)); });
                values[j] = res;
                runtimes[j] = t;
            }
            bmk_io("armadillo", dim, values, runtimes);
        }

        i++;
    }


}