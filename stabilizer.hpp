#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <deque>

#include "ukf.hpp"
#include "alignment.hpp"

struct VideoStabilizerParams {
    VideoAlignerParams aligner;

    // Lag: number of frames to delay before smoothing
    int lag = 0;

    // Enable UKF: if false, just use the aligner
    bool enable_ukf = false;

    // Crop pixels: if > 0, crop the output image by this amount
    int crop_pixels = 16;

    // Displacement thresholds for decay and full reset
    double min_disp = 24.0, max_disp = 64.0;
    double min_decay = 0.95, max_decay = 0.5;
};

class VideoStabilizer {
public:
    VideoStabilizer(const VideoStabilizerParams& params = VideoStabilizerParams());
    /**
     * Process one frame and return the stabilized frame if available;
     * otherwise return an empty cv::Mat until enough frames have arrived.
     */
    cv::Mat processFrame(const cv::Mat& inputFrame);

protected:
    VideoStabilizerParams m_params;
    VideoAligner            aligner;  ///< alignment class
    CameraMotionUKF_FixedLag ukf;     ///< The Unscented Kalman Filter (fixed lag)
    int                    m_frameIndex = 0;
    int                    alignFailures = 0;

    // For each frame i, we store a measured transform from frame i-1 to i
    // (i.e. T_i) once alignment is done.  We'll push them into the UKF.
    std::deque<SimilarityTransform> m_measurementBuffer;

    // We also keep around all input frames until it's time to pop them
    // (once the UKF is done smoothing that portion of the timeline).
    std::deque<cv::Mat> m_frameBuffer;

    SimilarityTransform m_accum;
};
