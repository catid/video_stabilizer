#include "smoother.hpp"

// -------------------- RunningStat helpers --------------------
// O(1) sliding‑window linear‑regression accumulator. The implementation
// is deliberately kept very small as we want it to be header‑only except
// for these two methods.

void L1SmootherCenter::RunningStat::push(double y, int window)
{
    // If the buffer is already full, remove the oldest sample first so
    // that after the push we are still at (window) elements.
    if (static_cast<int>(buf.size()) == window) {
        const double y_old = buf.front();
        buf.pop_front();

        // Remove its contribution.  Note that after popping, all
        // remaining samples shift their index by −1, hence Σ i*y becomes
        // Σ (i‑1)*y = Σ i*y - Σ y  (where the new Σ refers to the values
        // still in the window).  This single subtraction keeps the
        // statistics consistent in O(1).
        sum_y  -= y_old;
        sum_xy -= sum_y;
        --n;
    }

    // Append the new value at index = n (current size).
    sum_y  += y;
    sum_xy += static_cast<double>(n) * y;
    buf.push_back(y);
    ++n;
}

double L1SmootherCenter::RunningStat::predict_last() const
{
    // Return 0 for empty sequence.
    if (n == 0) return 0.0;
    // For a single sample the best prediction is the sample itself.
    if (n == 1) return buf.back();

    // Pre‑computed closed‑form sums for x = 0..n‑1.
    const double sum_x  = 0.5 * n * (n - 1);                 // Σ x
    const double sum_xx = (static_cast<double>(n - 1) * n * (2 * n - 1)) / 6.0; // Σ x²

    const double denom  = n * sum_xx - sum_x * sum_x;
    if (std::fabs(denom) < 1e-12) return buf.back(); // Degenerate case

    const double slope     = (n * sum_xy - sum_x * sum_y) / denom;
    const double intercept = (sum_y - slope * sum_x) / n;

    return slope * static_cast<double>(n - 1) + intercept;
}

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



L1SmootherCenter::L1SmootherCenter(int lagBehind, int lagAhead,
                                   double lambda_tx, double lambda_rot)
    : m_lagBehind(lagBehind)
    , m_lagAhead(lagAhead)
    , m_lambda_tx(lambda_tx)
    , m_lambda_rot(lambda_rot < 0 ? 0.5 * lambda_tx : lambda_rot)
{
    m_windowSize = std::max(1, m_lagBehind + m_lagAhead);
}

bool L1SmootherCenter::update(const SimilarityTransform& meas,
                              SimilarityTransform& outFinalized)
{
    // 1) Feed the new measurement to the running statistics. This is an
    //    O(1) operation regardless of the window size.
    m_statA .push(meas.A , m_windowSize);
    m_statB .push(meas.B , m_windowSize);
    m_statTX.push(meas.TX, m_windowSize);
    m_statTY.push(meas.TY, m_windowSize);

    // (Optional) keep the raw measurements for external inspection / tests.
    m_measurements.push_back(meas);

    // 2) We can only return a smoothed value once we have accumulated at
    //    least (windowSize) samples – otherwise the linear‑prediction would
    //    be based on too little context and the behaviour would deviate
    //    from the previous implementation.
    if (m_statA.n < m_windowSize)
        return false;

    // 3) Predict the value for the newest (last) sample using the statistics.
    SimilarityTransform sm;
    sm.A  = m_statA .predict_last();
    sm.B  = m_statB .predict_last();
    sm.TX = m_statTX.predict_last();
    sm.TY = m_statTY.predict_last();

    // 4) Step‑detection: if the newest measurement deviates from the
    //    prediction by more than the configured thresholds, propagate it
    //    unfiltered so that intentional camera motion is preserved.
    if (std::fabs(meas.TX - sm.TX) > m_lambda_tx) sm.TX = meas.TX;
    if (std::fabs(meas.TY - sm.TY) > m_lambda_tx) sm.TY = meas.TY;
    if (std::fabs(meas.A  - sm.A ) > m_lambda_rot) sm.A  = meas.A;
    if (std::fabs(meas.B  - sm.B ) > m_lambda_rot) sm.B  = meas.B;

    outFinalized = sm;
    return true;
}
