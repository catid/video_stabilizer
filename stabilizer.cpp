#include "stabilizer.hpp"

VideoStabilizer::VideoStabilizer()
{
    // We start with no residual jitter
    m_accum = SimilarityTransform(); // identity
}

cv::Mat VideoStabilizer::processFrame(const cv::Mat& inputFrame, int crop_pixels)
{
    ++m_frameIndex;

    // Push the incoming frame into the buffer
    m_frameBuffer.push_back(inputFrame);

    // Attempt to measure the transform from the previous to the current frame
    SimilarityTransform transform;
    bool success = aligner.AlignNextFrame(inputFrame, transform);
    if (!success) {
        ++alignFailures;
        std::cerr << "Frame " << m_frameIndex << ": Alignment failed # " << alignFailures << std::endl;
        reset = true;
        m_accum = SimilarityTransform();
    } else {
        //std::cout << "Frame " << m_frameIndex 
        //          << ": Alignment OK => " << transform.toString() << std::endl;

        // Run it through the UKF to get a *one-frame-delayed* estimate
        // FIXME: UKF is not working properly.
        //SimilarityTransform transformEst = ukf.update(transform, reset);
        // (At the *very first* call, transformEst might be identity or uninitialized.)

        //  => accum = accum ∘ diff
        m_accum = m_accum.compose(transform);
    }

    // The transform we will actually apply to a frame is "accum^-1" if we want
    // to “undo” the accumulated jitter. That is, if accum is the net “jitter,”
    // we want to warp the frames by the inverse, to keep them stable.
    SimilarityTransform correction = m_accum.inverse();

    double displacement = m_accum.maxCornerDisplacement(inputFrame.cols, inputFrame.rows);
    double decay = 1.0;

    const double min_displacement = 24.0;
    const double max_displacement = 64.0;
    if (displacement < min_displacement) {
        decay = 1.0;
    } else if (displacement > max_displacement) {
        reset = true;
        m_accum = SimilarityTransform();
    } else {
        // 0.95 at min_displacement, 0.5 at max_displacement
        double f = (displacement - min_displacement) / (max_displacement - min_displacement);
        if (f < 0.0) {
            f = 0.0;
        } else if (f > 1.0) {
            f = 1.0;
        }
        decay = 0.95 * (1.0 - f) + 0.5 * f;
    }

    m_accum.TX *= decay;
    m_accum.TY *= decay;
    m_accum.A *= decay;
    m_accum.B *= decay;

#if 0
    // Now we produce output for the *previous* frame in our buffer
    // only if we have at least 2 frames in the buffer:
    //  - the one we are measuring transforms for (the current)
    //  - the one we want to stabilize and output (the previous).
    cv::Mat processedFrame = inputFrame.clone(); // fallback if not enough frames
    if (m_frameBuffer.size() > 1) {
        // Grab the oldest frame from the buffer. This is the frame we are “delaying.”
        cv::Mat frameToStabilize = m_frameBuffer.front();
        m_frameBuffer.pop_front(); // remove from the queue

        // FIXME #4: “apply” the accumulated transform (its inverse) to that frame
        processedFrame = warpBySimilarityTransform(frameToStabilize, correction);
    }

    // Debug
    //std::cout << "UKF (delayed) Est: " << transformEst.toString() << "\n"
    //          << "  => accum now: " << m_accum.toString() << std::endl;
#else
    auto processedFrame = warpBySimilarityTransform(inputFrame, correction);
#endif

    if (crop_pixels > 0) {
        cv::Rect roi(
            crop_pixels,
            crop_pixels,
            processedFrame.cols - 2 * crop_pixels,
            processedFrame.rows - 2 * crop_pixels);

        processedFrame = processedFrame(roi);
    }

    return processedFrame;
}
