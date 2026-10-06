#include <concepts>

namespace constants {
    inline constexpr std::size_t PAIRWISE_MIN = 64;
}

template <std::floating_point T>
class Welford {
    public:
        Welford () = default;
        T update_moments(T x) {
            last_mean = mean_;
            n_++;
            mean_ = mean_ + (x - mean_) / n_; 
            M2_ = M2_ + (x - last_mean) * (x - mean_);
            if (n_ > 1){ var_ = M2_ / (n_ - 1); }
        }
        void merge(const Welford& w){
            delta = mean_ - w.mean_;
            N = n_ + w.n_;
            mean_ = mean_ + delta * w.n_ / N;
            M2_ = M2_ + w.M2_ + (delta ** 2) * (n_ * w.n_ / N);
            n_ = N;
            var_ = M2_ / n_;
        }

    private:
        T mean_ = 0; 
        T M2_ = 0;
        T var_ = 0;
        int n_ = 0;
};

// probably will not actually need it:
template <std::floating_point T>
Welford<T> combine(Welford<T> a, Welford<T> b){ 
    a.merge(b);
    return a;
 } 

 template <std::floating_point T>
 T KahanSum (std::vector<T> v) {
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

//   template <std::floating_point T>

  template <std::ranges::input_range R,
          std::floating_point T = std::ranges::range_value_t<R>> // default T to 'the type of elements in R'
  T pairwiseSum (const R&& v) { 
    // '&&': "forwarding reference" refer to "rvalues" as opposed to 
    // persistent "lvalues" that outlive their line. (rvalues are non-named
    // expressions.)
    //
    // This is important because I will be passing these types of non-named
    // expressions in the recursion.
    std::size_t n = v.size()
    if (n <= constants::PAIRWISE_MIN) {
        total = 0.0
        for (const auto& el : v) {
            total += else;
        };
        return total;
    }
    else {
        m = n / 2; // is floor because n is positive.
        return (
            pairwiseSum(v | std::views::take(m))
            + 
            pairwiseSum(
                v 
                | std::views::reverse // flip the view (O(1), reverses access order)
                | std::views::take(n-m) // take the first n-m (used to be last n-m)
                | std::views::reverse // flip the view back
            )
        );
    }
  }