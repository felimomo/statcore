#ifndef AGG_H
#define AGG_H

#include <vector>
#include <span>

namespace agg {
    template <std::floating_point T>
    class Welford {
        private:
            T mean_ = T{0};
            T M2_ = T{0};
            T var_ = T{0};
            std::size_t n_ = 0;
        public:
            Welford() = default;
            T update_moments(T x);
            void merge(const Welford& w);
    };

    template <std::floating_point T>
    Welford<T> combine(Welford<T> a, Welford<T> b);

    template <std::floating_point T>
    T KahanSum (std::vector<T> v);

    template <std::floating_point T>
    T pairwiseSpan(std::span<const T> s);

    template <std::floating_point T>
    T pairwiseSum(const std::vector<T>& v);

    template <std::floating_point T>
    T logSumExp(const std::vector<T>& v);

    template <std::floating_point T>
    T log_1pexp(T x);
}

#endif