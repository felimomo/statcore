#include <concepts>

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