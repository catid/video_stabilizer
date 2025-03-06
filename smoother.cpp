#include "smoother.hpp"

/**
 * 1D Total Variation (L1) smoothing using a simple (iterative)
 * projection approach. This tries to solve:
 *
 *    minimize   sum_i |x_i - data_i| + lambda * sum_i |x_{i+1} - x_i|
 *
 * for x_i, given data_i.  The parameter 'lambda' controls smoothness:
 *  - lambda = 0   -> no smoothing, result = data
 *  - bigger lambda -> more smoothing
 *
 * This is a standard problem; there are more efficient or advanced
 * ways to do this. For demonstration, we implement a simple approach.
 *
 * Returns a vector of smoothed values of the same size as input.
 */
static std::vector<double> tvl1_smooth(const std::vector<double>& data, double lambda, int iterations = 100)
{
    const size_t N = data.size();
    if (N == 0) return {};

    // Initialize x = data
    std::vector<double> x = data;

    // We'll do a simple iterative scheme:
    for(int iter = 0; iter < iterations; ++iter)
    {
        // 1) Proximal step towards data (L1 fidelity):
        for(size_t i = 0; i < N; i++)
        {
            // The "proximal" step for L1 is basically a soft-threshold,
            // but because we have an absolute difference cost, the best
            // local step is to move x_i a bit closer to data_i.
            // We'll do a simple relaxation:
            double alpha = 0.5; // a small relaxation factor
            x[i] = (1.0 - alpha) * x[i] + alpha * data[i];
        }

        // 2) Total variation shrinkage across edges:
        // We'll shrink differences x[i+1]-x[i]. Each iteration:
        for(size_t i = 0; i + 1 < N; i++)
        {
            double diff = x[i+1] - x[i];
            double mag  = std::fabs(diff);
            if (mag > lambda)
            {
                // shrink the difference by lambda, equally distributed
                double shrink = (mag - lambda) / mag * 0.5; 
                x[i]   += diff * shrink;
                x[i+1] -= diff * shrink;
            }
            else
            {
                // If |diff| <= lambda, clamp them to midpoint
                double mid = 0.5*(x[i] + x[i+1]);
                x[i] = mid;
                x[i+1] = mid;
            }
        }
    }

    return x;
}

L1SmootherCenter::L1SmootherCenter(int lagBehind, int lagAhead, double lambda)
: m_lagBehind(lagBehind)
, m_lagAhead(lagAhead)
, m_lambda(lambda)
, m_nextToFinalize(0)
{
}

bool L1SmootherCenter::update(const SimilarityTransform& meas,
            SimilarityTransform& outFinalized)
{
    // 1) Append new measurement
    m_measurements.push_back(meas);
    const int newestIndex = (int)m_measurements.size() - 1;

    // 2) Check if the "m_nextToFinalize" is within a fully-known window
    // We need at least (m_nextToFinalize + lagAhead) <= newestIndex
    // i.e. the future frames we want are available.
    if (m_nextToFinalize + m_lagAhead > newestIndex) {
        return false;
    }

    // We'll finalize the frame at index = m_nextToFinalize.

    // 2a) Build the sub-range we want to smooth
    int startIndex = std::max(0, m_nextToFinalize - m_lagBehind);
    int endIndex   = m_nextToFinalize + m_lagAhead;

    // 2b) Slice the measurements
    std::vector<double> Avec, Bvec, TXvec, TYvec;
    for (int i = startIndex; i <= endIndex; i++)
    {
        const auto &m = m_measurements[i];
        Avec.push_back(m.A);
        Bvec.push_back(m.B);
        TXvec.push_back(m.TX);
        TYvec.push_back(m.TY);
    }

    // 2c) Smooth each parameter independently
    auto A_smooth  = tvl1_smooth(Avec,  m_lambda);
    auto B_smooth  = tvl1_smooth(Bvec,  m_lambda);
    auto TX_smooth = tvl1_smooth(TXvec, m_lambda);
    auto TY_smooth = tvl1_smooth(TYvec, m_lambda);

    // 2d) The "middle" one in that sub-range is index:
    //     (m_nextToFinalize - startIndex)
    int middle = m_nextToFinalize - startIndex;

    SimilarityTransform sm;
    sm.A  = A_smooth[middle];
    sm.B  = B_smooth[middle];
    sm.TX = TX_smooth[middle];
    sm.TY = TY_smooth[middle];

    outFinalized = sm;

    // 2e) We have now finalized m_nextToFinalize
    m_nextToFinalize++;

    return true; // We did produce a finalized transform
}
