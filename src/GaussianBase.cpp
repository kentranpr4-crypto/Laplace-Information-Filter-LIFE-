#include "GaussianBase.h"

#include <cmath>
#include <stdexcept>
#include <boost/math/distributions/chi_squared.hpp>

double GaussianBase::chi2inv(double p, double nu)
{
    if (!(p > 0.0 && p < 1.0))
    {
        throw std::domain_error("GaussianBase::chi2inv: probability must lie in (0,1)");
    }
    if (!(nu > 0.0))
    {
        throw std::domain_error("GaussianBase::chi2inv: degrees of freedom must be positive");
    }
    boost::math::chi_squared_distribution<double> dist(nu);
    return boost::math::quantile(dist, p);
}

double GaussianBase::probabilityFromNSigma(double nSigma)
{
    if (!(nSigma > 0.0))
    {
        throw std::domain_error("GaussianBase::probabilityFromNSigma: nSigma must be positive");
    }
    return std::erf(nSigma / std::sqrt(2.0));
}

double GaussianBase::mahalanobisThresholdSq(double nSigma, double dim)
{
    return chi2inv(probabilityFromNSigma(nSigma), dim);
}
