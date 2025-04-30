#include "stabilizer.hpp"

VideoStabilizer::VideoStabilizer(const VideoStabilizerParams& params)
    : m_l1Smoother(params.lag, params.smoother_memory, params.lambda)
{
    m_params = params;
}

cv::Mat VideoStabilizer::processFrame(const cv::Mat& inputFrame)
{
    // 1) Increment our frame index
    ++m_frameIndex;

    // 2) Store the incoming frame for delayed output
    m_frameBuffer.push_back(inputFrame.clone());

    // 3) Measure transform from the previous frame to current
    SimilarityTransform currentMeas;
    bool success = aligner.AlignNextFrame(inputFrame, currentMeas, m_params.aligner);

    // Ignore zoom?
    //currentMeas.A = 0.0;

#if 0
    if (!success) {
        std::cout << "Alignment failed for frame " << m_frameIndex << std::endl;
    }
#endif

    // 4) Update the smoother, which returns the earliest measurement that is
    //    now fully “smoothed” after seeing lag_ future measurements.
    bool reset = !success;
    SimilarityTransform earliestSmoothed;
    if (m_params.enable_smoother) {
        m_l1Smoother.update(currentMeas, earliestSmoothed);
    }

    // If alignment fails, we reset accum (like original code).
    if (reset) {
        m_accum = SimilarityTransform(); // identity
    }

    // 5) Store the new measurement (for future finalization)
    m_measurementBuffer.push_back(currentMeas);

    // 6) Check if we have finalized an old measurement
    //    i.e. the smoother has done “lag” updates since that old measurement
    bool hasFinalized = (m_measurementBuffer.size() > (size_t)m_params.lag);

    cv::Mat outputFrame; // empty unless we finalize

    if (hasFinalized)
    {
        // 6a) Pop the earliest measurement from the queue
        SimilarityTransform earliestMeas = m_measurementBuffer.front();
        m_measurementBuffer.pop_front();

        SimilarityTransform jitter;
        if (m_params.enable_smoother) {
            jitter = earliestMeas.compose( earliestSmoothed.inverse() );
        } else {
            jitter = earliestMeas;  // <-- (A) purely raw measurement
        }

        // 6c) Compose into newAccum
        SimilarityTransform newAccum = m_accum.compose(jitter);

        // 6d) Check displacement for partial or full reset
        double displacement = newAccum.maxCornerDisplacement(
                                  inputFrame.cols, inputFrame.rows);

        double decay = 1.0;
        if (displacement > m_params.max_disp) {
            decay = m_params.max_decay;
        } else if (displacement > m_params.min_disp) {
            double f = (displacement - m_params.min_disp) / (m_params.max_disp - m_params.min_disp);
            f = std::max(0.0, std::min(1.0, f));
            decay = m_params.min_decay * (1.0 - f) + m_params.max_decay * f;
        } else {
            decay = m_params.min_decay;
        }

        newAccum.TX *= decay;
        newAccum.TY *= decay;
        newAccum.A  *= decay;
        newAccum.B  *= decay;

        m_accum = newAccum;

        // 6e) Pop the corresponding earliest frame
        if (!m_frameBuffer.empty())
        {
            cv::Mat frameToStabilize = m_frameBuffer.front();
            m_frameBuffer.pop_front();

            // 6f) Warp it by newAccum.inverse()
            SimilarityTransform correction = newAccum.inverse();
            cv::Mat stabilized = warpBySimilarityTransform(
                                     frameToStabilize, correction);

            // 7) Optional crop
            if (m_params.crop_pixels > 0) {
                cv::Rect roi(
                    m_params.crop_pixels, m_params.crop_pixels,
                    stabilized.cols  - 2*m_params.crop_pixels,
                    stabilized.rows  - 2*m_params.crop_pixels
                );
                stabilized = stabilized(roi);
            }

            outputFrame = stabilized;
        }
    }

    // If nothing was finalized, outputFrame is empty
    return outputFrame;
}
