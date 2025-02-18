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
    VideoStabilizer();
    cv::Mat processFrame(const cv::Mat& inputFrame, int crop_pixels = 16);

protected:
    VideoAligner       aligner;  ///< Your alignment class
    CameraMotionUKF    ukf;      ///< The Unscented Kalman Filter
    bool               reset = true;
    int                m_frameIndex = 0;
    int alignFailures = 0;

    // An accumulated transform to hold “residual” jitter.
    // We will apply a slow decay so it doesn’t blow up.
    SimilarityTransform m_accum;

    // A small buffer to hold frames so we can output them one-frame delayed.
    std::deque<cv::Mat> m_frameBuffer;
};
