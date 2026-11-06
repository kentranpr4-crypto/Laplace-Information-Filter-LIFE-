#ifndef GAUSSIANBASE_H
#define GAUSSIANBASE_H

/**
 * @brief Common, non-templated base of the Gaussian density classes.
 *
 * Holds the statistical helpers shared by GaussianInfo: chi-squared quantiles (Boost.Math) and the
 * mapping from an "n-sigma" confidence level to the Mahalanobis gate used for compatibility tests.
 */
class GaussianBase
{
public:
    virtual ~GaussianBase() = default;

    /**
     * @brief Quantile (inverse CDF) of the chi-squared distribution.
     * @param p   probability in (0, 1)
     * @param nu  degrees of freedom (> 0)
     * @return x such that P(X <= x) = p for X ~ chi^2(nu)
     * @throws std::domain_error for p outside (0,1) or nu <= 0
     */
    static double chi2inv(double p, double nu);

    /**
     * @brief Central probability mass of an n-sigma interval of a 1-D Gaussian: erf(nSigma / sqrt(2)).
     *
     * For nSigma = 3 this is 0.9973002..., i.e. the same mass as a 3-sigma band of a scalar Gaussian.
     */
    static double probabilityFromNSigma(double nSigma);

    /**
     * @brief Squared Mahalanobis threshold of the n-sigma confidence region in dim dimensions.
     *
     * Returns chi2inv(probabilityFromNSigma(nSigma), dim). For dim = 1 this equals nSigma^2 exactly.
     * A sample x is inside the region iff (x - mu)^T P^{-1} (x - mu) <= this value.
     */
    static double mahalanobisThresholdSq(double nSigma, double dim);
};

#endif
