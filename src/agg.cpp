#include <algorithm>
#include <concepts>
#include <cmath>
#include <span>
#include <vector>

namespace constants {
    inline constexpr std::size_t PAIRWISE_MIN = 64;
}

namespace agg {

template <std::floating_point T>
class Welford {
    public:
        // Welford () = default; // no need for this, already in header
        T update_moments(T x) {
            T last_mean = mean_;
            n_++;
            mean_ = mean_ + (x - mean_) / n_;
            M2_ = M2_ + (x - last_mean) * (x - mean_);
            if (n_ > 1) { var_ = M2_ / (n_ - 1); }
            return mean_;
        }
        void merge(const Welford& w){
            T delta = mean_ - w.mean_;
            T N = n_ + w.n_;
            mean_ = mean_ + delta * w.n_ / N;
            M2_ = M2_ + w.M2_ + (delta * delta) * (n_ * w.n_ / N);
            n_ = N;
            var_ = M2_ / n_;
        }

    private:
        T mean_ = T{0};
        T M2_ = T{0};
        T var_ = T{0};
        std::size_t n_ = 0;
};

// probably will not actually need it:
template <std::floating_point T>
Welford<T> combine(Welford<T> a, Welford<T> b){ 
    a.merge(b);
    return a;
 } 

 template <std::floating_point T>
 T KahanSum (const std::vector<T>& v) {
    T c = 0.0;
    T total = 0.0;
    for (const auto& el : v ){
        T tmp1 = el - c;
        T tmp2 = total + tmp1; // precision lost in tmp1 due to sum >> tmp1
        c = (tmp2 - total) - tmp1; //singles out precision lost in tmp1
        total = tmp2;
    };
    return total;
}

// The following pattern seems to be generally "helper + driver".
// helper is the span-based function that does the background computations
// on the type best-fit to the function (spans help the recursion).
//
// The driver then is the interface between the actual type we want (std::vector)
// and the type native to the helper.
//
// Alternative names: "wrapper + recursive helper" and "worker / wrapper"
//
// this wrapping seems to be needed because of templating? For 'regular' types,
// an std::vector can be converted down to a span at compile time, however
// the compiler does not perform conversions on template types.

// function takes a span argument, in order to 
template <std::floating_point T>
T pairwiseSpan(std::span<const T> s) {
    if (s.size() <= constants::PAIRWISE_MIN) {
        T total = 0;
        for (T el : s) total += el;
        return total;
    }
    std::size_t m = s.size() / 2;
    return pairwiseSpan(s.first(m)) + pairwiseSpan(s.subspan(m));
}

// perform pairwise sum on vector
template <std::floating_point T>
T pairwiseSum(const std::vector<T>& v) {
    return pairwiseSpan(std::span<const T>(v));
}

template <std::floating_point T>
T logSumExp(const std::vector<T>& v) { // vectorize
    T vmax = std::ranges::max(v);
    std::vector<T> safe_exp_v(v.size());
    std::transform( //vector-level transformation (not sure if it's actually vectorized)
        v.begin(), 
        v.end(), \
        safe_exp_v.begin(), // data destination (copies transformed data at safe_v)
        [vmax](T x) { return std::exp(x - vmax); } //lambda expression, lambda needs access to vmax local var
    );
    return std::log(pairwiseSum(safe_exp_v)) + vmax;
}

template <std::floating_point T>
T log_1pexp(T x) {
    return std::log(1 + std::exp(x));
}

#define AGG_INSTANTIATE(T)                                      \
    template class Welford<T>;                                  \
    template Welford<T> combine<T>(Welford<T>, Welford<T>);     \
    template T KahanSum<T>(const std::vector<T>&);              \
    template T pairwiseSpan<T>(std::span<const T>);             \
    template T pairwiseSum<T>(const std::vector<T>&);           \
    template T logSumExp<T>(const std::vector<T>&);             \
    template T log_1pexp<T>(T);

AGG_INSTANTIATE(float)
AGG_INSTANTIATE(double)
AGG_INSTANTIATE(long double)
#undef AGG_INSTANTIATE

} // namespace agg