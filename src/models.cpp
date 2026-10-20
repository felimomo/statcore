#include <Eigen/Dense>

#include <cmath>
#include <numbers>

namespace Models{

    using Eigen::VectorXd;
    using Eigen::MatrixXd;
    using std::exp;
    using std::numbers::pi;

    class Guassian {
        public:
            Gaussian (VectorXd mu_, MatrixXd Sigma): Sigma_(Sigma), mu_(mu) {
                std::size_t dim_ = mu_.size();
                double DetSigma_ = Sigma.determinant();
                double NormConst_ = pow(2 * pi, - dim_ / 2.0) * pow(DetSigma_, - 1.0 / 2);
            }
            double predict(VectorXd x) {
                return NormConst_ * exp((x - mu_).transpose() * Sigma_ * (x - mu_) );
            }
        private:
            MatrixXd Sigma_;
            VectorXd mu_;
            double DetSigma_;
            double NormConst_;
            std::size_t dim_;
    };

    class BinomLogit {};

    class PoissonLog {};

}