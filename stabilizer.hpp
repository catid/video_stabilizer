#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <deque>

#include "ukf.hpp"
#include "alignment.hpp"

class VideoStabilizer {
public:
    VideoStabilizer(int lag = 3);
    /**
     * Process one frame and return the stabilized frame if available;
     * otherwise return an empty cv::Mat until enough frames have arrived.
     */
    cv::Mat processFrame(const cv::Mat& inputFrame, int crop_pixels = 16);

protected:
    VideoAligner            aligner;  ///< alignment class
    CameraMotionUKF_FixedLag ukf;     ///< The Unscented Kalman Filter (fixed lag)
    int                    m_frameIndex = 0;
    int                    alignFailures = 0;
    int                    m_lag = 0;

    // For each frame i, we store a measured transform from frame i-1 to i
    // (i.e. T_i) once alignment is done.  We'll push them into the UKF.
    std::deque<SimilarityTransform> m_measurementBuffer;

    // We also keep around all input frames until it's time to pop them
    // (once the UKF is done smoothing that portion of the timeline).
    std::deque<cv::Mat> m_frameBuffer;

    SimilarityTransform m_accum;
};
