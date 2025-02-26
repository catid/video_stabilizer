#include "stabilizer.hpp"

VideoStabilizer::VideoStabilizer(int lag)
    : ukf(lag)
{
    m_lag = lag;
}

static bool isIdentity(const SimilarityTransform &t, double eps = 1e-12)
{
    return (std::fabs(t.A)  < eps &&
            std::fabs(t.B)  < eps &&
            std::fabs(t.TX) < eps &&
            std::fabs(t.TY) < eps);
}

cv::Mat VideoStabilizer::processFrame(const cv::Mat& inputFrame, int crop_pixels)
{
    // Increment frame index
    ++m_frameIndex;

    // Keep a copy of the new input frame
    m_frameBuffer.push_back(inputFrame.clone());

    // 1) Compute the measurement from the previous frame to current
    SimilarityTransform currentMeas;
    bool success = aligner.AlignNextFrame(inputFrame, currentMeas);

    // 2) Update the filter. The transform returned here is
    //    the *earliest measurement* that the UKF has just “finalized,”
    //    i.e. it has been smoothed by lag_ future measurements.
    bool reset = !success;
    SimilarityTransform earliestSmoothed = ukf.update(currentMeas, reset);

    // If alignment failed, we reset our local accum/buffers
    if (reset) {
        m_accum = SimilarityTransform(); // identity
    }

    // 3) We still push the *new* measurement onto our local buffer
    m_measurementBuffer.push_back(currentMeas);

    // 4) Check if the UKF has finalized an old measurement.
    //    Option A: Test if earliestSmoothed is non-identity.
    //    Option B (original code): Use m_measurementBuffer.size() > m_lag
    bool hasFinalized = (m_measurementBuffer.size() > (size_t)m_lag);

    cv::Mat outputFrame; // will be empty if nothing is finalized

    if (hasFinalized)
    {
        // The earliest measurement that corresponds to earliestSmoothed
        SimilarityTransform earliestMeas = m_measurementBuffer.front();
        m_measurementBuffer.pop_front();

        // Compute the “jitter” between the raw measurement and
        // the final smoothed result for that measurement:
        //    T_jitter = T_meas ∘ (T_smoothed)^(-1)
        //SimilarityTransform jitter = earliestMeas.compose(earliestSmoothed.inverse());
        SimilarityTransform jitter = earliestMeas;

        // Compose into the “accum” to keep track of net drift
        SimilarityTransform newAccum = m_accum.compose(jitter);

        // Optionally check for large displacement => partial or full reset
        double displacement = newAccum.maxCornerDisplacement(
                                  inputFrame.cols, inputFrame.rows);
        const double min_disp = 24.0, max_disp = 64.0;
        double decay = 1.0;
        if (displacement > max_disp) {
            // Hard reset
            reset = true;
            m_accum = SimilarityTransform(); // identity
        } else if (displacement > min_disp) {
            double f = (displacement - min_disp) / (max_disp - min_disp);
            f = std::max(0.0, std::min(1.0, f));
            decay = 0.95 * (1.0 - f) + 0.5 * f;
        }

        // Apply decay factor
        m_accum.TX *= decay;
        m_accum.TY *= decay;
        m_accum.A  *= decay;
        m_accum.B  *= decay;

        // 5) Now that we have the final accum for that earliest measurement,
        //    pop the corresponding earliest frame and warp it
        if (!m_frameBuffer.empty())
        {
            cv::Mat frameToStabilize = m_frameBuffer.front();
            m_frameBuffer.pop_front();

            // The actual correction is the inverse of newAccum
            SimilarityTransform correction = newAccum.inverse();
            cv::Mat stabilized = warpBySimilarityTransform(
                                     frameToStabilize, correction);

            // Optional crop
            if (crop_pixels > 0) {
                cv::Rect roi(crop_pixels, crop_pixels,
                             stabilized.cols - 2 * crop_pixels,
                             stabilized.rows - 2 * crop_pixels);
                stabilized = stabilized(roi);
            }

            outputFrame = stabilized;
        }
    }

    // If no measurement was finalized this round, outputFrame remains empty
    return outputFrame;
}
