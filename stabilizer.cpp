#include "stabilizer.hpp"

VideoStabilizer::VideoStabilizer(int lag)
    : ukf(lag)
{
    m_lag = lag;
}

cv::Mat VideoStabilizer::processFrame(const cv::Mat& inputFrame, int crop_pixels)
{
    // Bump our frame index
    ++m_frameIndex;

    // 1) Store the new input frame so we can stabilize and output it later
    m_frameBuffer.push_back(inputFrame.clone());

    // 2) Measure the transform from the *previous* frame to this frame
    SimilarityTransform currentMeas;
    bool success = aligner.AlignNextFrame(inputFrame, currentMeas);
    if (!success)
    {
        //std::cerr << "Frame " << m_frameIndex << ": alignment failed # " << (++alignFailures) << std::endl;
    }

    // 4) Give the measurement to the UKF to get a *smoothed* estimate
    bool reset = !success;
    SimilarityTransform cameraMotion = ukf.update(currentMeas, reset);

    if (reset) {
        m_measurementBuffer.clear();
        m_accum = SimilarityTransform(); // identity
    }

    // 5) Check if we can finalize the earliest measurement yet.
    //    For a lag-L filter, once we have L+1 measurements in the UKF,
    //    the earliest one in the buffer is no longer going to change.
    //    So if smoothedEstBuffer has size > L, the front is finalized.
    m_measurementBuffer.push_back(currentMeas);
    bool hasFinalized = (m_measurementBuffer.size() > (size_t)m_lag);

    cv::Mat outputFrame; // empty unless we finalize

    if (hasFinalized)
    {
        // The earliest measurement and its final smoothed estimate
        SimilarityTransform earliestMeas = m_measurementBuffer.front();
        m_measurementBuffer.pop_front();

        // Compute the "jitter" = MeasuredTransform ∘ (cameraMotion)^(-1)
        SimilarityTransform jitter = earliestMeas.compose(cameraMotion.inverse());

        // Compute the new one
        SimilarityTransform newAccum = m_accum.compose(jitter);

        // Optionally check for large displacement → partial or full reset:
        double displacement = newAccum.maxCornerDisplacement(inputFrame.cols, inputFrame.rows);
        const double min_disp = 24.0, max_disp = 64.0;
        double decay = 1.0;
        if (displacement > max_disp) {
            // Hard reset
            reset = true;
            m_accum = SimilarityTransform(); // identity
        }
        else if (displacement > min_disp) {
            // partial decay
            double f = (displacement - min_disp) / (max_disp - min_disp);
            f = std::max(0.0, std::min(1.0, f));
            decay = 0.95 * (1.0 - f) + 0.5 * f;
        }

        // Apply decay factor
        m_accum.TX *= decay;
        m_accum.TY *= decay;
        m_accum.A  *= decay;
        m_accum.B  *= decay;

        // 6) Now that we know the final accum for that earliest measurement,
        //    we can pop the corresponding earliest frame and warp it:
        if (!m_frameBuffer.empty())
        {
            cv::Mat frameToStabilize = m_frameBuffer.front();
            m_frameBuffer.pop_front();

            // The actual correction to *apply* is the inverse of the accum:
            SimilarityTransform correction = newAccum.inverse();
            cv::Mat stabilized = warpBySimilarityTransform(frameToStabilize, correction);

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

    // If we did not finalize anything this time, outputFrame will be empty.
    return outputFrame;
}
